"""
#2 in the "NN to tiny LLM" ladder.

Same bigram task as #1, but instead of COUNTING transitions we LEARN them with
a neural network trained by gradient descent — exactly the machinery from your
XOR net (neural_network.py), now applied to language.

The remarkable result: a single-layer net trained this way converges to the
SAME probabilities the counting model found. This proves the two views are one
idea. Counting is the closed-form answer; gradient descent is the general tool
that keeps working once models get too complex to count.

Key new concepts vs #1:
  - one-hot encoding   : represent a character as a vector
  - softmax            : turn raw network outputs into probabilities
  - cross-entropy loss : the standard loss for "predict the next token"

Run with:  python bigram_nn.py
"""

import numpy as np

WORDS = [
    "emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper",
    "liam", "noah", "oliver", "william", "james", "benjamin", "lucas", "henry",
    "anna", "anya", "maria", "elena", "nina", "sara", "leo", "max", "ella",
]

CHARS = ["."] + sorted("abcdefghijklmnopqrstuvwxyz")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)  # 27


def build_dataset(words):
    """Flatten all words into (input_char, target_char) index pairs."""
    xs, ys = [], []
    for word in words:
        symbols = ["."] + list(word) + ["."]
        for current, nxt in zip(symbols, symbols[1:]):
            xs.append(STOI[current])
            ys.append(STOI[nxt])
    return np.array(xs), np.array(ys)


def one_hot(indices, n):
    """Turn a list of indices into rows of 0s with a single 1 (the 'on' char)."""
    out = np.zeros((len(indices), n))
    out[np.arange(len(indices)), indices] = 1
    return out


def softmax(logits):
    """Turn raw scores ('logits') into probabilities that sum to 1 per row."""
    logits = logits - logits.max(axis=1, keepdims=True)   # stability trick
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train(xs, ys, epochs=200, lr=20.0):
    """Train a single weight matrix W mapping input char -> next-char scores.

    W has shape (27, 27): one column of scores per possible next character.
    Because the input is one-hot, `X @ W` simply selects the relevant row of W,
    so this net is literally learning a probability table — the same table #1
    counted, but discovered through gradient descent.
    """
    rng = np.random.default_rng(0)
    W = rng.normal(size=(VOCAB, VOCAB))
    X = one_hot(xs, VOCAB)          # (N, 27)
    n = len(xs)

    for epoch in range(epochs):
        # --- Forward pass ---
        logits = X @ W              # (N, 27) raw scores for each next char
        probs = softmax(logits)     # (N, 27) probabilities

        # --- Cross-entropy loss: -log(probability assigned to the TRUE char) ---
        loss = -np.mean(np.log(probs[np.arange(n), ys] + 1e-9))

        # --- Backward pass ---
        # Gradient of softmax + cross-entropy is beautifully simple:
        # (predicted prob - 1 for the true class), averaged over the batch.
        d_logits = probs.copy()
        d_logits[np.arange(n), ys] -= 1
        d_logits /= n
        d_W = X.T @ d_logits

        # --- Gradient descent update ---
        W -= lr * d_W

        if epoch % 20 == 0:
            print(f"epoch {epoch:4d}  loss {loss:.4f}")

    return W


def generate(W, rng, max_len=20):
    """Sample a new word from the trained net, one character at a time."""
    index = 0
    out = []
    for _ in range(max_len):
        logits = one_hot([index], VOCAB) @ W
        probs = softmax(logits)[0]
        index = rng.choice(VOCAB, p=probs)
        if index == 0:
            break
        out.append(ITOS[index])
    return "".join(out)


if __name__ == "__main__":
    xs, ys = build_dataset(WORDS)
    W = train(xs, ys)

    print("\nGenerated names:")
    rng = np.random.default_rng(0)
    for _ in range(100):
        print("  " + generate(W, rng))
