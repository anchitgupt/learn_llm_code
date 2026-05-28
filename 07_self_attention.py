"""
#7 in the "NN to tiny LLM" ladder.  (the crucial mechanism)

ATTENTION is the idea that makes transformers — and therefore modern LLMs —
work. The MLP (#4/#6) crammed a fixed window of characters into one vector and
treated every position the same. Attention instead lets every token look back
over ALL previous tokens and decide, for itself, which ones are relevant right
now.

The mechanism, per token:
  - Query (Q): "what am I looking for?"
  - Key   (K): "what do I offer?"
  - Value (V): "what information do I pass on if attended to?"

A token compares its Query against every other token's Key (a dot product) to
get attention SCORES, softmaxes them into WEIGHTS, then takes a weighted sum of
the Values. That weighted sum is the token's new, context-aware representation.

CAUSAL MASKING: in a language model a token may only attend to itself and
EARLIER tokens (it must not peek at the future it's trying to predict). We
enforce this by setting future scores to -infinity before the softmax.

This file implements one attention head from scratch in PyTorch and prints the
attention weights so you can SEE which tokens attend to which.

Run with:  python 07_self_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class SelfAttentionHead(nn.Module):
    """A single causal self-attention head."""

    def __init__(self, n_embd, head_size):
        super().__init__()
        # Three linear projections turn each token's embedding into Q, K, V.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x, return_weights=False):
        # x: (B, T, C) = (batch, time/sequence length, embedding dim)
        B, T, C = x.shape
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Attention scores: how much each token (query) matches every other
        # token (key). Scale by sqrt(head_size) to keep softmax well-behaved.
        scores = q @ k.transpose(-2, -1) / self.head_size ** 0.5  # (B, T, T)

        # Causal mask: forbid attending to FUTURE positions.
        mask = torch.tril(torch.ones(T, T))            # lower-triangular 1s
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)            # (B, T, T), rows sum to 1
        out = weights @ v                              # (B, T, head_size)

        if return_weights:
            return out, weights
        return out


if __name__ == "__main__":
    # A toy sequence of 5 tokens, each a 4-dim embedding, single batch.
    B, T, C = 1, 5, 4
    x = torch.randn(B, T, C)

    head = SelfAttentionHead(n_embd=C, head_size=4)
    out, weights = head(x, return_weights=True)

    print(f"input  shape: {tuple(x.shape)}  (batch, tokens, embd)")
    print(f"output shape: {tuple(out.shape)}  (each token now context-aware)\n")

    print("Attention weights — row i = how token i distributes attention over")
    print("tokens 0..4. Note the lower-triangular shape: no token sees the")
    print("future, and each row sums to 1.\n")
    w = weights[0]
    header = "        " + "".join(f"tok{j}  " for j in range(T))
    print(header)
    for i in range(T):
        row = "  ".join(f"{w[i, j]:.2f}" for j in range(T))
        print(f"  tok{i}  {row}")

    print("\nRead it: token 0 can only attend to itself (1.00). Token 4 spreads")
    print("attention across all 5 tokens. The upper triangle is exactly 0 —")
    print("that's the causal mask doing its job.")
