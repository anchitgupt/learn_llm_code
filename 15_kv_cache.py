"""
#15 in the "NN to tiny LLM" ladder.  (fixing the KV-cache problem)

Every generate() we've written so far has a hidden flaw. To produce one new
token it reruns the WHOLE sequence through the network:

    logits, _ = self(idx[:, -BLOCK_SIZE:])   # recompute keys/values for ALL tokens

So generating T tokens costs ~T**2/2 token-forwards. The keys and values for the
earlier tokens never change once computed, yet we recompute them every step.

The fix is the KV CACHE: store each layer's past keys (K) and values (V), and at
each step feed in only the ONE new token, attending its query over the cached
K/V. That makes generation O(T) instead of O(T**2). This is also where #14's GQA
finally pays off: a grouped-query model caches FEWER kv-heads, so the cache is
smaller.

This script reuses the trained #14 model (modern_gpt.pt), implements a cache-based
decoder, and proves it produces the EXACT same tokens as the naive loop while
doing far less work.

Run with:  python 15_kv_cache.py   (requires modern_gpt.pt from running #14 first)
"""

import importlib.util
import os
import time

import torch
import torch.nn.functional as F

# Reuse the #14 model, its building blocks, and the tokenizer.
spec = importlib.util.spec_from_file_location("modern_gpt", "14_modern_gpt.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

DEVICE = mg.DEVICE
BLOCK = mg.BLOCK_SIZE
tg = mg.tg


@torch.no_grad()
def naive_generate(model, prompt, n_new):
    """The old way: re-encode the whole context every single step."""
    idx = torch.tensor([prompt], device=DEVICE)
    token_forwards = 0
    for _ in range(n_new):
        ctx = idx[:, -BLOCK:]
        token_forwards += ctx.shape[1]          # every token re-processed
        logits, _ = model(ctx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx[0].tolist(), token_forwards


@torch.no_grad()
def _step(model, caches, token_id, pos):
    """Run ONE token through the layers at absolute position `pos`, updating
    the per-layer K/V caches, and return the next-token logits."""
    B = 1
    groups = mg.N_HEAD // mg.N_KV_HEAD
    x = model.token_emb(torch.tensor([[token_id]], device=DEVICE))   # (1, 1, C)
    cos, sin = model.cos[pos:pos + 1], model.sin[pos:pos + 1]

    for i, block in enumerate(model.blocks):
        a = block.attn
        h = block.attn_norm(x)
        q = a.q_proj(h).view(B, 1, mg.N_HEAD, mg.HEAD_DIM).transpose(1, 2)
        k = a.k_proj(h).view(B, 1, mg.N_KV_HEAD, mg.HEAD_DIM).transpose(1, 2)
        v = a.v_proj(h).view(B, 1, mg.N_KV_HEAD, mg.HEAD_DIM).transpose(1, 2)
        q, k = mg.apply_rope(q, cos, sin), mg.apply_rope(k, cos, sin)

        # Append this token's K/V to the cache, then attend over ALL of it.
        if caches[i]["k"] is None:
            caches[i]["k"], caches[i]["v"] = k, v
        else:
            caches[i]["k"] = torch.cat([caches[i]["k"], k], dim=2)
            caches[i]["v"] = torch.cat([caches[i]["v"], v], dim=2)
        K, V = caches[i]["k"], caches[i]["v"]            # (1, n_kv, pos+1, hd)

        Kr = K.repeat_interleave(groups, dim=1)          # GQA: share kv across heads
        Vr = V.repeat_interleave(groups, dim=1)
        att = (q @ Kr.transpose(-2, -1)) / mg.HEAD_DIM ** 0.5     # (1, nh, 1, pos+1)
        att = F.softmax(att, dim=-1)                     # no mask: cache is all past
        o = (att @ Vr).transpose(1, 2).contiguous().view(B, 1, mg.N_HEAD * mg.HEAD_DIM)
        x = x + a.o_proj(o)
        x = x + block.ffn(block.ffn_norm(x))

    return model.head(model.norm(x))[:, -1, :]           # (1, vocab)


@torch.no_grad()
def cached_generate(model, prompt, n_new):
    """The fix: prefill the prompt into per-layer caches, then emit one token
    at a time, processing only the newest token each step."""
    caches = [{"k": None, "v": None} for _ in model.blocks]
    tokens = list(prompt)
    token_forwards = 0
    pos = 0

    logits = None
    for t in prompt:                                     # prefill the prompt
        logits = _step(model, caches, t, pos); pos += 1; token_forwards += 1
    for _ in range(n_new):                               # then generate
        nxt = int(logits.argmax(dim=-1))
        tokens.append(nxt)
        logits = _step(model, caches, nxt, pos); pos += 1; token_forwards += 1
    return tokens, token_forwards


def cache_bytes(n_tokens, kv_heads, dtype_bytes=2):
    """Total KV-cache size: 2 (K&V) x layers x kv_heads x head_dim x tokens."""
    return 2 * mg.N_LAYER * kv_heads * mg.HEAD_DIM * n_tokens * dtype_bytes


if __name__ == "__main__":
    if not os.path.exists(mg.CKPT_PATH):
        raise SystemExit("No modern_gpt.pt — run `python 14_modern_gpt.py` first.")
    model = mg.ModernGPT().to(DEVICE)
    model.load_state_dict(torch.load(mg.CKPT_PATH, map_location=DEVICE)["model"])
    model.eval()

    prompt = tg.encode("ROMEO:")                         # a real prompt -> readable output
    N = BLOCK - len(prompt) - 1                           # stay within the trained context
    naive_ids, naive_fw = naive_generate(model, prompt, N)
    cached_ids, cached_fw = cached_generate(model, prompt, N)

    print("=== Correctness ===")
    print(f"naive and cached produce identical tokens: {naive_ids == cached_ids}")
    print(f"sample: {tg.decode(cached_ids)!r}\n")

    print("=== Work done (token-forwards through the layers) ===")
    print(f"naive : {naive_fw:5d}   (~T^2/2 — recomputes the whole context each step)")
    print(f"cached: {cached_fw:5d}   (T — prefill + one new token per step)")
    print(f"-> {naive_fw / cached_fw:.0f}x less work for {N} new tokens\n")

    print("=== Wall-clock (30 runs) ===")
    t0 = time.perf_counter(); [naive_generate(model, prompt, N) for _ in range(30)]
    t_naive = time.perf_counter() - t0
    t0 = time.perf_counter(); [cached_generate(model, prompt, N) for _ in range(30)]
    t_cached = time.perf_counter() - t0
    print(f"naive : {t_naive:.2f}s")
    print(f"cached: {t_cached:.2f}s   ({t_naive / t_cached:.1f}x faster)\n")

    print("=== KV-cache memory @ full context (fp16) ===")
    gqa = cache_bytes(BLOCK, mg.N_KV_HEAD)
    mha = cache_bytes(BLOCK, mg.N_HEAD)
    print(f"GQA ({mg.N_KV_HEAD} kv-heads): {gqa/1024:.1f} KB")
    print(f"full multi-head ({mg.N_HEAD} heads): {mha/1024:.1f} KB  "
          f"-> GQA is {mha/gqa:.0f}x smaller")
    print("\nThis cache is what Google's TurboQuant compresses (next: #16).")
