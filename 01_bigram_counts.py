"""
#1 in the "NN to tiny LLM" ladder.

The simplest possible language model: a BIGRAM model built from pure counts.
No neural network, no gradients — just "how often does character Y follow
character X?" We tally that into a table, turn counts into probabilities,
and sample new words one character at a time.

This is the conceptual seed of every LLM: predict the next token given the
previous one(s). Everything later just makes this prediction smarter.

Run with:  python bigram_counts.py
"""

import numpy as np

# A tiny training corpus. Real models use millions of words; we use a handful
# of names so you can eyeball what's happening. '.' is a special marker for
# both START and END of a word.
WORDS = [
    "emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper",
    "liam", "noah", "oliver", "william", "james", "benjamin", "lucas", "henry",
    "anna", "anya", "maria", "elena", "nina", "sara", "leo", "max", "ella",
]


def build_counts(words):
    """Count every (current_char -> next_char) pair across all words."""
    # 27 symbols: '.' (index 0) plus 'a'..'z' (1..26).
    chars = ["."] + sorted("abcdefghijklmnopqrstuvwxyz")
    stoi = {c: i for i, c in enumerate(chars)}        # char  -> index
    itos = {i: c for c, i in stoi.items()}            # index -> char

    counts = np.zeros((27, 27), dtype=np.int64)
    for word in words:
        # Wrap each word with start/end markers: ".emma."
        symbols = ["."] + list(word) + ["."]
        for current, nxt in zip(symbols, symbols[1:]):
            counts[stoi[current], stoi[nxt]] += 1
    return counts, stoi, itos


def to_probabilities(counts):
    """Convert raw counts into a probability for each row.

    +1 is 'smoothing': it stops any transition from being impossible (prob 0),
    which would otherwise break sampling and infinite-loss situations.
    """
    probs = counts.astype(np.float64) + 1
    probs /= probs.sum(axis=1, keepdims=True)   # each row sums to 1
    return probs


def generate(probs, itos, rng, max_len=20):
    """Sample one new word, character by character, until we hit '.'."""
    index = 0                       # always start at '.'
    out = []
    for _ in range(max_len):
        # Pick the next character according to the learned probabilities.
        index = rng.choice(27, p=probs[index])
        if index == 0:              # hit the end marker
            break
        out.append(itos[index])
    return "".join(out)


if __name__ == "__main__":
    counts, stoi, itos = build_counts(WORDS)
    probs = to_probabilities(counts)

    # Peek at what it learned: what most often follows the start marker '.'?
    start_row = probs[0]
    top = np.argsort(start_row)[::-1][:5]
    print("Most likely first letters:")
    for i in top:
        print(f"  '{itos[i]}'  {start_row[i]:.2%}")

    print("\nGenerated names:")
    rng = np.random.default_rng(0)
    for _ in range(10):
        print("  " + generate(probs, itos, rng))
