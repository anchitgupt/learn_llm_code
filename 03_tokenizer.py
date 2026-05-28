"""
#3 in the "NN to tiny LLM" ladder.

A model only understands numbers, so we need a TOKENIZER: a reversible map
between text and a sequence of integer IDs.

  encode("hi")  -> [7, 8]
  decode([7,8]) -> "hi"

Two approaches here:
  1. CharTokenizer  — one ID per character. Simple, what #1/#2 used implicitly.
  2. BPETokenizer   — Byte-Pair Encoding: start from characters, then greedily
                      MERGE the most frequent adjacent pair into a new token,
                      over and over. This is (essentially) what GPT-2/3/4 use.
                      It keeps the vocabulary small while letting common chunks
                      like "th", "ing", " the" become single tokens.

Run with:  python tokenizer.py
"""

from collections import Counter


# --------------------------------------------------------------------------
# 1. Character-level tokenizer
# --------------------------------------------------------------------------
class CharTokenizer:
    """One token per unique character in the training text."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# --------------------------------------------------------------------------
# 2. Byte-Pair Encoding (BPE) tokenizer
# --------------------------------------------------------------------------
class BPETokenizer:
    """Learn a vocabulary by repeatedly merging the most common adjacent pair.

    Start with raw bytes (0..255) so ANY text is representable. Then run
    `num_merges` rounds; each round finds the most frequent neighbouring pair
    of token IDs and assigns it a brand-new ID.
    """

    def __init__(self):
        self.merges = {}   # (id_a, id_b) -> new_id, in the order learned
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> raw bytes

    def _pair_counts(self, ids):
        return Counter(zip(ids, ids[1:]))

    def _merge(self, ids, pair, new_id):
        """Replace every occurrence of `pair` in `ids` with `new_id`."""
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text, num_merges=20):
        ids = list(text.encode("utf-8"))          # bytes -> initial token IDs
        for k in range(num_merges):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)     # most frequent adjacent pair
            new_id = 256 + k
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            # Record what bytes this new token expands to (for decoding).
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        # Apply merges in the SAME order they were learned.
        for pair, new_id in self.merges.items():
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    text = "the theme of the theater is the thunder"

    print("=== CharTokenizer ===")
    ct = CharTokenizer(text)
    ids = ct.encode("the theme")
    print("vocab size:", ct.vocab_size)
    print("encode('the theme') ->", ids)
    print("decode back        ->", repr(ct.decode(ids)))

    print("\n=== BPETokenizer ===")
    bpe = BPETokenizer().train(text, num_merges=10)
    ids = bpe.encode("the theme")
    print("vocab size:", bpe.vocab_size, "(256 bytes + learned merges)")
    print("encode('the theme') ->", ids)
    print("decode back        ->", repr(bpe.decode(ids)))
    print(f"'the theme' is {len('the theme')} chars but only {len(ids)} BPE tokens")

    print("\nLearned merges (most common pairs became single tokens):")
    for pair, new_id in bpe.merges.items():
        piece = bpe.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {pair} -> {new_id}   = {piece!r}")
