"""
#6 in the "NN to tiny LLM" ladder.

The SAME MLP language model from #4 (mlp_lm.py) — embeddings + context window +
hidden layer — but written in PyTorch. Compare the two files side by side:

  - The forward pass is almost identical to our NumPy version.
  - ALL the hand-coded backprop from #4 collapses into one line: loss.backward()
    (PyTorch's autograd, exactly the engine you built in #5, doing the work).
  - Parameter updates become an optimizer: optimizer.step().

This is the toolkit we'll use to build the transformer. Once you trust that
this matches #4, we never hand-derive a gradient again.

Run with:  python switch_to_pytorch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

WORDS = [
    "emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper",
    "liam", "noah", "oliver", "william", "james", "benjamin", "lucas", "henry",
    "anna", "anya", "maria", "elena", "nina", "sara", "leo", "max", "ella",
    "grace", "chloe", "zoe", "lily", "aria", "ruby", "ivy", "luna", "nora",
]

CHARS = ["."] + sorted("abcdefghijklmnopqrstuvwxyz")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)

BLOCK_SIZE = 3
EMB_DIM = 8
HIDDEN = 64

torch.manual_seed(0)


def build_dataset(words):
    xs, ys = [], []
    for word in words:
        context = [0] * BLOCK_SIZE
        for ch in list(word) + ["."]:
            target = STOI[ch]
            xs.append(context)
            ys.append(target)
            context = context[1:] + [target]
    return torch.tensor(xs), torch.tensor(ys)


class MLP(nn.Module):
    """Exactly #4's architecture, expressed with PyTorch building blocks."""

    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(VOCAB, EMB_DIM)            # the embedding table
        self.fc1 = nn.Linear(BLOCK_SIZE * EMB_DIM, HIDDEN)  # hidden layer
        self.fc2 = nn.Linear(HIDDEN, VOCAB)              # output scores (logits)

    def forward(self, x):
        emb = self.C(x)                  # (N, BLOCK_SIZE, EMB_DIM)
        flat = emb.view(emb.shape[0], -1)  # flatten the context
        h = torch.tanh(self.fc1(flat))
        logits = self.fc2(h)
        return logits


def train(model, X, Y, epochs=2000, lr=0.1):
    # The optimizer holds the parameters and applies the gradient-descent step.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        logits = model(X)
        # cross_entropy = softmax + negative-log-likelihood, in one call.
        loss = F.cross_entropy(logits, Y)

        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # <-- autograd computes EVERY gradient for us
        optimizer.step()        # <-- update all parameters

        if epoch % 200 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")


@torch.no_grad()
def generate(model, max_len=20):
    context = [0] * BLOCK_SIZE
    out = []
    for _ in range(max_len):
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1).item()
        if idx == 0:
            break
        out.append(ITOS[idx])
        context = context[1:] + [idx]
    return "".join(out)


if __name__ == "__main__":
    X, Y = build_dataset(WORDS)
    n_params = sum(p.numel() for p in MLP().parameters())
    print(f"dataset: {len(X)} examples | model: {n_params} parameters\n")

    model = MLP()
    train(model, X, Y)

    print("\nGenerated names:")
    for _ in range(12):
        print("  " + generate(model))
