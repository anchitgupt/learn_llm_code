# Build an LLM from Scratch 🧠→🤖

A hands-on learning ladder: **17 small, runnable Python scripts** that take you
from a 60-line neural network all the way to a **2026-architecture, instruction-
and preference-tuned** GPT-style language model — down to a **quantized KV cache**.
Each script teaches **one** new concept, prints output you can actually read, and
is heavily commented.

Nothing here is a black box. By the end, every core idea inside a frontier LLM
(GPT-4, Claude, …) is something you've built and run yourself. The only thing
separating this from the real thing is **scale** — more parameters, more data,
more compute.

> 📖 **Read it as a website:** the whole ladder is also a single static page in
> [`docs/`](docs/index.html) — narrative, real terminal output, and full source
> for every step. See [Deploy to GitHub Pages](#deploy-to-github-pages) below.

---

## Quick start

```bash
# one-time setup (creates a local virtual env with numpy + torch)
python3 -m venv .venv
.venv/bin/pip install numpy torch

# then run any script
.venv/bin/python 00_neural_network.py
.venv/bin/python 09_tiny_gpt.py
```

> The first run of `09_tiny_gpt.py` trains for ~2 min and saves `tiny_gpt.pt`.
> Every run after that — and scripts #10 and #12 — **load that file instantly**,
> so you never retrain unless you delete the checkpoint.

---

## The ladder

Run them in order. Each builds directly on the last.

| # | Script | New concept | The "aha" |
|---|--------|-------------|-----------|
| 0 | `00_neural_network.py` | forward pass + **backprop** | a net learns XOR from scratch |
| 1 | `01_bigram_counts.py` | "predict the next token" | a language model is just counting |
| 2 | `02_bigram_nn.py` | softmax + **cross-entropy** | gradient descent finds the same answer counting did |
| 3 | `03_tokenizer.py` | char-level + **BPE** | how text becomes numbers (and back) |
| 4 | `04_mlp_lm.py` | **embeddings** + context window | names finally start looking real |
| 5 | `05_autograd.py` | a mini **micrograd** | how backprop automates itself |
| 6 | `06_switch_to_pytorch.py` | **PyTorch** | all the hand-coded gradients collapse to `loss.backward()` |
| 7 | `07_self_attention.py` | **attention** (Q/K/V + causal mask) | tokens look at each other |
| 8 | `08_transformer_block.py` | multi-head + **residuals** + layernorm | the stackable unit of a transformer |
| 9 | `09_tiny_gpt.py` | **a real GPT** | trained on Shakespeare, it generates text ✅ |
| 10 | `10_sampling.py` | temperature / top-k / top-p | one model, many "writers" |
| 11 | `11_train_loop.py` | resumable checkpoints + LR schedule | production-grade training |
| 12 | `12_finetune.py` | **instruction tuning** | base model → assistant 🤖 |
| 13 | `13_dpo.py` | **DPO** preference tuning | align without a reward model or RL (the modern RLHF) |
| 14 | `14_modern_gpt.py` | **RoPE · RMSNorm · SwiGLU · GQA** | the 2026 architecture — better loss, fewer params |
| 15 | `15_kv_cache.py` | **KV cache** | fix O(T²) generation — identical output, ~31× less work |
| 16 | `16_kv_quant.py` | **TurboQuant** KV-cache quantization | 3-bit cache via the rotation trick (Google, ICLR 2026) |

---

## The story in three acts

**Act 1 — From neural net to language (1–3).**
A neural net is a function that learns from examples (#0). Point it at "what
character comes next?" and you have a language model (#1). Whether you *count*
the answer or *learn* it by gradient descent, you land in the same place (#2).
First you have to turn text into numbers — that's tokenization (#3).

**Act 2 — The deep-learning toolkit (4–6).**
Represent each token as a learned vector (an *embedding*) and look at several
previous tokens at once, and generated text gets dramatically better (#4).
Understand how gradients compute themselves (#5), then hand the job to PyTorch
so you never derive a gradient by hand again (#6).

**Act 3 — The Transformer (7–9).**
*Attention* lets every token decide which earlier tokens matter (#7). Wrap it
with a feed-forward layer, residuals, and layernorm into a reusable block (#8),
stack the blocks, add position embeddings, and you have a **real GPT** (#9).

**Act 4 — Make it usable (10–12).**
Control its creativity with sampling (#10), train it like a pro (#11), and
fine-tune it on instruction→response pairs so it *follows commands* instead of
just continuing text — the leap from GPT-3 to ChatGPT (#12).

**Act 5 — State of the art, 2026 (13–14).**
Align it with **DPO** — the modern alternative to RLHF that needs no reward
model and no RL loop (#13). Then rebuild the GPT with the components every
frontier open model now uses — **RoPE, RMSNorm, SwiGLU, and GQA** — and watch it
beat the classic design with *fewer* parameters (#14).

**Act 6 — Inference efficiency (15–16).**
Fix the hidden O(T²) flaw in generation with a real **KV cache** (#15), then
**compress** that cache to ~3 bits with the rotation trick behind Google's
**TurboQuant** (#16) — the difference between a model that's *correct* and one
that's *deployable* at long context.

---

## Files the scripts create

| File | Made by | What it is |
|------|---------|------------|
| `input.txt` | (downloaded) | ~1MB of Shakespeare — the training corpus |
| `tiny_gpt.pt` | #9 | the trained **base** model (load-if-exists) |
| `train_ckpt.pt` / `train_best.pt` | #11 | resumable training state / best-val weights |
| `tiny_gpt_chat.pt` | #12 | the **instruction-tuned** model |
| `tiny_gpt_dpo.pt` | #13 | the **DPO-aligned** model |
| `modern_gpt.pt` | #14 | the **2026-architecture** model |

To force any model to retrain, just delete its `.pt` file.

---

## Deploy to GitHub Pages

The website lives in [`docs/`](docs/) as a single self-contained `index.html`
(regenerate it any time with `python build_site.py`). To publish it:

```bash
git init && git add . && git commit -m "LLM-from-scratch ladder + site"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Then in the repo on GitHub: **Settings → Pages → Build and deployment → Source:
Deploy from a branch → `main` / `docs`**. (No build step or workflow needed —
the site is plain static HTML.)

Your site goes live at `https://<you>.github.io/<repo>/`. (The `.pt`, `.venv`,
and `input.txt` files are git-ignored, so only code + site get pushed.)

---

## Where to go next

- **Scale it up** — increase `N_EMBD`, `N_LAYERS`, `BLOCK_SIZE` in `09_tiny_gpt.py`
  (or `14_modern_gpt.py`) and watch the output sharpen.
- **Word-level GPT** — feed #3's BPE tokenizer into #9 so it models tokens, not
  just characters.
- **Reasoning RL** — the step *after* DPO: **GRPO / RLVR**, training on verifiable
  rewards. It's the 2026 frontier for reasoning models.
- **Mixture-of-Experts / MLA** — route each token to a few specialist MLPs, or
  compress the KV cache further (DeepSeek-style) for cheaper long context.

---

*Built step by step, one runnable script at a time. The architecture is GPT's;
the scale is a laptop's.* 🏔️
