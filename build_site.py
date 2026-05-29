"""
Static-site generator for the "Build an LLM from Scratch" learning ladder.

Reads the real source of every numbered script and emits a single, self-contained
docs/index.html (inline CSS + JS) that is ready to deploy on GitHub Pages.

Run with:  python build_site.py
"""

import html
import json
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).parent
DOCS = ROOT / "docs"
REPO = "https://github.com/anchitgupt/learn_llm_code"   # for "open notebook" links


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def read_source(filename: str) -> str:
    path = ROOT / filename
    return path.read_text(encoding="utf-8") if path.exists() else f"# {filename} not found"


# --------------------------------------------------------------------------
# Content model: one entry per rung of the ladder.
#   body = list of HTML paragraph strings (already trusted markup)
#   out  = captured terminal output (escaped at render time)
# --------------------------------------------------------------------------
STEPS = [
    dict(n="00", phase="Foundation", file="00_neural_network.py",
         title="A neural network",
         tag="forward pass + backprop, learning XOR",
         body=[
             "Every LLM is, at bottom, a neural network trained by gradient descent. So we start at the true atom: a two-layer net that learns <strong>XOR</strong> &mdash; the classic problem a single linear layer <em>cannot</em> solve.",
             "It contains the three ideas that never leave: a <strong>forward pass</strong> (predict), a <strong>loss</strong> (measure wrongness), and <strong>backpropagation</strong> (nudge every weight downhill). Written in pure NumPy so nothing hides.",
         ],
         out="""Final predictions:
  [0 0] -> 0.009  (rounded: 0)
  [0 1] -> 0.984  (rounded: 1)
  [1 0] -> 0.986  (rounded: 1)
  [1 1] -> 0.019  (rounded: 0)"""),
    dict(n="01", phase="From neural net to language", file="01_bigram_counts.py",
         title="The simplest language model",
         tag="predict the next character by counting",
         body=[
             "A language model answers one question: <em>given what came before, what comes next?</em> The simplest possible version doesn't even use a network &mdash; it just <strong>counts</strong> which character tends to follow which, then samples from those frequencies.",
             "The generated names are mostly gibberish, and that's the lesson: looking at only <em>one</em> previous character has no memory of the wider word. Every later step exists to fix this.",
         ],
         out="""Most likely first letters:
  'a'  9.62%   'm'  7.69%   'e'  7.69%

Generated names:
  nea
  po
  sxormxu
  tzybzxvkfux"""),
    dict(n="02", phase="From neural net to language", file="02_bigram_nn.py",
         title="The same model, learned",
         tag="softmax + cross-entropy via gradient descent",
         body=[
             "Now we solve the <em>same</em> bigram task with a neural network trained by gradient descent &mdash; and it converges to the same answer the counting found. Two different methods, one truth.",
             "This introduces the permanent LLM toolkit: <strong>softmax</strong> (scores &rarr; probabilities) and <strong>cross-entropy loss</strong> (the universal next-token objective). Counting is the closed form; gradient descent is the general tool that keeps working when models get too complex to count.",
         ],
         out="""epoch    0  loss 3.8114
epoch  100  loss 1.6104
epoch  180  loss 1.5931

Generated names:
  maria   liliar   oler   naria"""),
    dict(n="03", phase="From neural net to language", file="03_tokenizer.py",
         title="Turning text into numbers",
         tag="char-level and Byte-Pair Encoding (BPE)",
         body=[
             "Models only understand numbers, so we need a reversible map between text and integer IDs. We build a character tokenizer, then a tiny <strong>BPE</strong> &mdash; the algorithm GPT actually uses, which greedily merges the most frequent adjacent pair into a new token, over and over.",
             "Watch it learn <code>t,h &rarr; 'th' &rarr; 'the' &rarr; ' the'</code>. Fewer tokens per word means more real text fits in the context window &mdash; the whole reason BPE beats characters.",
         ],
         out="""Learned merges (most common pairs become tokens):
  (t, h)   -> 'th'
  'th' + e -> 'the'
  ' ' + the -> ' the'
'the theme' is 9 chars but only 1 BPE token"""),
    dict(n="04", phase="The deep-learning toolkit", file="04_mlp_lm.py",
         title="Embeddings & context",
         tag="the Bengio MLP — names start looking real",
         body=[
             "The leap. Each character becomes a small <strong>learned vector</strong> (an embedding), and the model looks at a <strong>window</strong> of previous characters at once. Suddenly the output is full of real names: <code>lucas, isabella, liam, ella</code>.",
             "Embeddings are the single most important idea in modern LLMs &mdash; every token in GPT is one. The architecture here is just the XOR net from #00 with <em>look-up &rarr; flatten &rarr; feed</em> bolted onto the front.",
         ],
         out="""epoch    0  loss 3.2892
epoch 1800  loss 0.7440

Generated names:
  lucas   isabella   liam   anna
  ruby    aria       ella   olioer"""),
    dict(n="05", phase="The deep-learning toolkit", file="05_autograd.py",
         title="Automatic differentiation",
         tag="a mini-micrograd — how backprop automates itself",
         body=[
             "We've hand-derived gradients three times. PyTorch does it automatically &mdash; and here's how, in miniature. Every <code>Value</code> remembers its parents and a local rule for one step of the chain rule. Calling <code>.backward()</code> walks the graph in reverse and assembles the whole derivative.",
             "The proof: the auto-computed gradient matches a numerical estimate exactly. You write only the forward math; the backward pass builds itself. That's all PyTorch's autograd is.",
         ],
         out="""forward output: Value(data=0.7064)

gradients (computed automatically):
  w1.grad = 1.0019
numeric check of w1.grad: 1.0019  (matches)"""),
    dict(n="06", phase="The deep-learning toolkit", file="06_switch_to_pytorch.py",
         title="Switching to PyTorch",
         tag="all the hand-coded backprop collapses to one line",
         body=[
             "The same #04 model, rewritten in PyTorch. Thirty lines of manual gradient math become a single <code>loss.backward()</code>; the update loop becomes <code>optimizer.step()</code>; softmax+log becomes <code>F.cross_entropy</code>.",
             "From here we never hand-derive a gradient again &mdash; which is exactly what frees us to build something as intricate as a transformer.",
         ],
         out="""dataset: 192 examples | model: 3571 parameters
epoch 1800  loss 0.6855

Generated names:
  benjamin   sophia   olivia   anya"""),
    dict(n="07", phase="The Transformer", file="07_self_attention.py",
         title="Self-attention",
         tag="the mechanism that makes transformers work",
         body=[
             "The heart of it all. Each token emits a <strong>Query</strong> (what I seek), a <strong>Key</strong> (what I offer), and a <strong>Value</strong> (what I pass on). Dot-product Query&middot;Key gives relevance scores &rarr; softmax &rarr; a weighted sum of Values. Every token gets a new, <em>context-aware</em> representation.",
             "The <strong>causal mask</strong> (the lower triangle below) forbids peeking at the future. Unlike the MLP's fixed window, attention scales to any context and <em>learns</em> what to focus on.",
         ],
         out="""Attention weights (row i attends over tokens 0..i):
        tok0  tok1  tok2  tok3  tok4
  tok0  1.00  0.00  0.00  0.00  0.00
  tok1  0.48  0.52  0.00  0.00  0.00
  tok4  0.12  0.20  0.20  0.20  0.29"""),
    dict(n="08", phase="The Transformer", file="08_transformer_block.py",
         title="The transformer block",
         tag="multi-head attention + MLP + residuals + norm",
         body=[
             "One head is the atom; the block is the molecule GPT stacks. <strong>Multi-head attention</strong> runs several heads in parallel, a <strong>feed-forward</strong> MLP lets each token think, and <strong>residual connections</strong> + <strong>layernorm</strong> make deep stacks trainable.",
             "Shape in equals shape out &mdash; which is precisely why you can stack 4, 12, or 96 of them. GPT-3 is 96 of these in a row.",
         ],
         out="""input  shape: (2, 6, 32)
output shape: (2, 6, 32)   <- identical, so blocks stack
4 heads, each of size 8 | 12,608 parameters
after stacking 3 blocks: (2, 6, 32)"""),
    dict(n="09", phase="The Transformer", file="09_tiny_gpt.py",
         title="A real GPT",
         tag="trained on Shakespeare — it generates text",
         body=[
             "Everything converges. Token + <strong>position embeddings</strong> &rarr; a stack of blocks &rarr; an output head, trained on ~1MB of Shakespeare. From a 4.36 loss of pure noise, it learns &mdash; entirely on its own &mdash; the play format, character names, line breaks, and Shakespearean cadence.",
             "This is a genuine, complete GPT (465K params). The architecture is identical to GPT-4; only the scale differs. The trained weights are saved to <code>tiny_gpt.pt</code> so every later step loads instantly.",
         ],
         out="""iter    0  train loss 4.365
iter 3000  train loss 1.493  val loss 1.674

=== Generated Shakespeare ===
KING XI:
Beliescapes not, if teach his such Lord,
ROMEO:
Whall suf yeten me assoous, being, where,"""),
    dict(n="10", phase="Make it usable", file="10_sampling.py",
         title="Sampling strategies",
         tag="temperature, top-k, top-p — one model, many writers",
         body=[
             "A trained model outputs a probability for every next token; <em>how</em> you pick from them transforms the text without touching a single weight. <strong>Greedy</strong> loops forever; <strong>temperature</strong> trades safety for creativity; <strong>top-k</strong> and <strong>top-p</strong> hit the sweet spot.",
             "This is exactly the <code>temperature</code> / <code>top_p</code> knob you set when calling an LLM API. The model is frozen &mdash; you're reshaping how it samples from itself.",
         ],
         out="""GREEDY  (gets stuck):
  the world of the world of the world...

TEMPERATURE 1.5  (wild, creative):
  thyremh, discops: LeWlooke me hand

TOP-P 0.9  (coherent and varied):
  KING HENRY VI: There shall seek so your way"""),
    dict(n="11", phase="Make it usable", file="11_train_loop.py",
         title="Production training",
         tag="resumable checkpoints + LR warmup/decay",
         body=[
             "A bare loop won't survive a multi-day run. This adds <strong>resumable checkpoints</strong> (model + optimizer + step), a <strong>learning-rate schedule</strong> (warmup then cosine decay), <strong>gradient clipping</strong>, and <strong>best-model tracking</strong>.",
             "Run it once, stop, run it again &mdash; it picks up exactly where it left off. That's how real training survives crashes and restarts.",
         ],
         out="""no checkpoint found - starting fresh
iter    0  lr 3.0e-05  train 4.365
iter 1500  train 1.599  val 1.757

# next run:
resumed from train_ckpt.pt at iter 1500"""),
    dict(n="12", phase="Make it usable", file="12_finetune.py",
         title="Instruction tuning",
         tag="base model → assistant (the ChatGPT leap)",
         body=[
             "A base model only continues text. The step that made ChatGPT is <strong>instruction tuning</strong>: fine-tune on (instruction, response) pairs, with <strong>loss masking</strong> so the model is trained only on the <em>answer</em>, not the question.",
             "Same weights, one nudge &mdash; but now it follows commands instead of rambling. Ask the base model to 'say hello' and it quotes Shakespeare; the tuned model replies politely.",
         ],
         out="""BASE model, ask 'say hello':
  -> 'I will not the world and the words,'

FINE-TUNED model:
  Q: say hello
  A: good morrow to you, friend.
  Q: what is love
  A: love is a gentle madness."""),
    dict(n="13", phase="State of the art · 2026", file="13_dpo.py",
         title="Preference tuning with DPO",
         tag="the modern alternative to RLHF",
         body=[
             "Instruction tuning only shows the model <em>good</em> answers. <strong>Preference tuning</strong> teaches it to prefer a chosen response over a rejected one. The 2026 standard is <strong>DPO</strong> (Direct Preference Optimization): where classic RLHF needs three models and a reinforcement-learning loop, DPO does it with a single, simple loss.",
             "Starting from the #12 model and a frozen reference copy, DPO widens the chosen-vs-rejected gap from +54 to +96 &mdash; with no reward model and no RL. This is essentially how modern open models are aligned today.",
         ],
         out="""BEFORE DPO:  chosen-minus-rejected gap: +54.22
training (chosen > rejected)...
step  400  loss 0.014
AFTER  DPO:  chosen-minus-rejected gap: +96.06"""),
    dict(n="14", phase="State of the art · 2026", file="14_modern_gpt.py",
         title="The 2026 architecture",
         tag="RoPE · RMSNorm · SwiGLU · GQA",
         body=[
             "Our #09 GPT is the classic 2017 transformer. Frontier open models (Llama, Mistral, Qwen, DeepSeek, Gemma) have converged on four upgrades: <strong>RoPE</strong> rotary positions, <strong>RMSNorm</strong>, <strong>SwiGLU</strong> gated feed-forwards, and <strong>GQA</strong> grouped-query attention.",
             "Same training data and loop as #09 &mdash; but it reaches a <strong>better validation loss (1.634 vs 1.674) with fewer parameters (418K vs 465K)</strong>. That efficiency is why the whole field switched. This is, in miniature, a 2026 model.",
         ],
         out="""device: cpu | parameters: 418,848
RoPE + RMSNorm + SwiGLU + GQA (4 query heads share 2 kv heads)
iter 2000  train 1.399  val 1.634   <- beats #09 with fewer params

=== Generated Shakespeare (modern architecture) ===
JULIET:
At, grones for your royalty, making, I'"""),
    dict(n="15", phase="Inference efficiency · 2026", file="15_kv_cache.py",
         title="The KV cache",
         tag="fix O(T²) generation — cache past keys/values",
         body=[
             "Every <code>generate()</code> so far hid a flaw: to produce one token it re-encodes the <em>entire</em> context, recomputing keys and values that never change. That's <strong>O(T²)</strong> work. The <strong>KV cache</strong> stores each layer's past K/V and feeds in only the new token each step — turning generation into <strong>O(T)</strong>.",
             "Proven on the #14 model: <strong>identical output</strong> to the naive loop with ~31× less work. And this is where #14's <strong>GQA</strong> finally pays off — a grouped-query model caches fewer kv-heads, so the cache itself is smaller.",
         ],
         out="""=== Correctness ===
naive and cached produce identical tokens: True
sample: 'ROMEO:\\nI shall be so seem the souls of the coursed...'

=== Work done (token-forwards through the layers) ===
naive : 1938   (~T^2/2 — recomputes the whole context each step)
cached:   63   (T — one new token per step)   -> 31x less work

=== KV-cache memory @ full context (fp16) ===
GQA (2 kv-heads): 48 KB   vs   full multi-head: 96 KB  (2x smaller)"""),
    dict(n="16", phase="Inference efficiency · 2026", file="16_kv_quant.py",
         title="Compressing the cache · TurboQuant",
         tag="3-bit KV cache via the rotation trick",
         body=[
             "At long context the KV cache — not the weights — is the memory wall. Google's <strong>TurboQuant</strong> (ICLR 2026, <a href=\"https://arxiv.org/abs/2504.19874\">arXiv:2504.19874</a>) shrinks it to <strong>~3 bits/value</strong> with near-zero loss. The obstacle is <em>outlier channels</em>; the trick is to <strong>rotate</strong> each vector by a random orthogonal matrix before quantizing — preserving dot products (so attention scores are unchanged) while spreading the outliers so few bits suffice.",
             "Shown honestly: a rotation only helps heavy-tailed data, so on our tiny near-Gaussian model it's <em>neutral</em> — but on the outlier structure real LLMs have, rotate-then-quantize cuts error <strong>1.8×</strong> and lifts attention fidelity from <strong>0.67 → 0.85</strong>, at 5.3× compression.",
         ],
         out="""=== Why outliers matter: rotation only helps heavy-tailed data ===
tiny model's real keys: near-Gaussian  ->  rotation ~neutral

=== With realistic outlier channels (ratio 21) ===
key reconstruction error @ 3-bit:
  naive quantization   : 0.566
  rotate-then-quantize  : 0.313   (1.8x lower)
attention-output fidelity vs fp (cosine sim):
  naive quantization   : 0.6658
  rotate-then-quantize  : 0.8502
memory: fp16 16 bits -> 3-bit = 5.3x smaller"""),
]


# Deeper theory per rung: an optional key formula + a few explanatory paragraphs.
# Keyed by step number; rendered after the intro paragraphs of each step.
THEORY = {
    "00": dict(
        eq="a = &sigma;(W&middot;x + b)&emsp;&middot;&emsp;&sigma;&prime;(z) = &sigma;(z)(1 &minus; &sigma;(z))&emsp;&middot;&emsp;w &larr; w &minus; &eta;&nbsp;&part;L/&part;w",
        cap="a neuron: weighted sum &rarr; nonlinearity; trained by gradient descent on the loss.",
        paras=[
            "A neuron computes a weighted sum of its inputs and passes it through a nonlinear <em>activation</em>. Stacking a hidden layer of these between input and output makes the network a <strong>universal function approximator</strong> &mdash; with enough hidden units it can represent any continuous function. The nonlinearity is essential: without it, stacked linear layers collapse into a single linear map.",
            "Learning means minimizing a <strong>loss</strong> (here, mean-squared error) by gradient descent. <strong>Backpropagation</strong> is just the chain rule applied in reverse: compute how the loss depends on each weight, layer by layer from output back to input, then nudge every weight a small step <em>against</em> its gradient. XOR is the canonical test because it is <em>not linearly separable</em> &mdash; a single layer cannot solve it, but a hidden layer learns the intermediate features that can.",
        ]),
    "01": dict(
        eq="P(next | cur) = count(cur, next) / &Sigma; count(cur, &middot;)",
        cap="the maximum-likelihood bigram: probabilities are just normalized counts.",
        paras=[
            "Language modeling is the task of estimating P(next token | previous tokens). A <strong>bigram</strong> model makes the <em>Markov assumption</em> that the next token depends only on the current one. Under that assumption the maximum-likelihood estimate has a closed form: count each pair and normalize each row to a probability distribution.",
            "<strong>Smoothing</strong> (adding 1 to every count, a.k.a. Laplace smoothing) prevents any transition from having probability zero &mdash; otherwise an unseen pair would make the whole sequence impossible and the log-loss infinite. Generation is then just repeated sampling from each conditional distribution. The fatal limit: a context of one token has no memory of the wider word, which is exactly what later steps fix.",
        ]),
    "02": dict(
        eq="softmax(z)&#7522; = exp(z&#7522;) / &Sigma;&#11388; exp(z&#11388;)&emsp;&middot;&emsp;L = &minus;log p(target)&emsp;&middot;&emsp;&part;L/&part;z = p &minus; y",
        cap="cross-entropy = negative log-likelihood; its gradient through softmax is simply p &minus; y.",
        paras=[
            "Here the bigram is <em>learned</em> instead of counted. The model maps a one-hot input through a weight matrix to <strong>logits</strong>, which <strong>softmax</strong> turns into a probability distribution. Training minimizes <strong>cross-entropy</strong> &mdash; the negative log-probability assigned to the true next token. Minimizing cross-entropy is identical to maximizing likelihood, the very objective the counting model solves in closed form &mdash; which is why the two converge to the same answer.",
            "The gradient of softmax-plus-cross-entropy is famously clean: <code>predicted &minus; actual</code>. Each row of the weight matrix is effectively learning the log-probabilities of the next token (a per-row logistic regression). The point isn't that gradient descent beats counting here &mdash; it ties it &mdash; but that gradient descent is the <em>general</em> tool that keeps working once the model is too complex to count.",
        ]),
    "03": dict(
        eq="merge* = argmax over adjacent pairs of  count(a, b)&emsp;&rarr;&emsp;new token",
        cap="BPE greedily merges the most frequent adjacent pair, over and over.",
        paras=[
            "A tokenizer is a reversible map between text and integer IDs, and its design is a trade-off between <strong>vocabulary size</strong> and <strong>sequence length</strong>. Character-level: tiny vocabulary, very long sequences, but never an unknown token. Word-level: short sequences, but a huge vocabulary and an out-of-vocabulary problem for any new word.",
            "<strong>Byte-Pair Encoding</strong> finds the sweet spot. Starting from raw bytes/characters, it repeatedly merges the most frequent adjacent pair into a new subword token. Frequent chunks like <code>the</code> or <code>ing</code> become single tokens while rare words still decompose into pieces &mdash; an information-theoretic win, giving common patterns short codes. Modern LLMs use byte-level BPE for exactly this robustness, which is why fewer tokens per word means more text fits in the context window.",
        ]),
    "04": dict(
        eq="e = C[context]&emsp;&middot;&emsp;h = tanh(W&#8321;&middot;flatten(e) + b&#8321;)&emsp;&middot;&emsp;logits = W&#8322;&middot;h",
        cap="Bengio's neural LM: look up embeddings for a window, then an MLP predicts the next token.",
        paras=[
            "This is the 2003 Bengio neural language model, and it introduces the idea that powers everything after it: the <strong>embedding</strong>. Each token is mapped to a dense, learned vector instead of a one-hot. Because similar tokens can learn similar vectors, the model shares statistical strength across contexts &mdash; directly fighting the <em>curse of dimensionality</em> that makes count-based n-grams blow up as the context grows.",
            "The second idea is a fixed <strong>context window</strong>: concatenate the embeddings of the previous <em>n</em> tokens and feed them through an MLP (the same tanh network as #00) to predict the next one. Now the prediction depends on several tokens jointly, so generated names look real. The remaining limitation &mdash; a <em>fixed</em>, small window &mdash; is what attention later removes.",
        ]),
    "05": dict(
        eq="reverse-mode autodiff:&emsp;&part;L/&part;u = &Sigma; (&part;L/&part;v)&middot;(&part;v/&part;u)&emsp;over children v of u",
        cap="every value remembers its parents and a local derivative; .backward() applies the chain rule in reverse.",
        paras=[
            "Hand-deriving gradients doesn't scale, so frameworks use <strong>reverse-mode automatic differentiation</strong>. As the forward pass runs, it records a <em>computational graph</em>: every intermediate value remembers which values produced it and a local rule for its derivative. This is not numerical approximation nor symbolic algebra &mdash; it's exact, computed by composing local derivatives.",
            "The backward pass walks that graph from the output back to the inputs in topological order, multiplying and accumulating local derivatives (the chain rule) to get <code>&part;loss/&part;parameter</code> for every parameter at once. Remarkably, the whole backward pass costs about the same as one forward pass &mdash; which is what makes training networks with billions of parameters feasible.",
        ]),
    "06": dict(
        eq="loss.backward()&emsp;&equiv;&emsp;the autograd graph of #05, on tensors&emsp;&middot;&emsp;optimizer.step()",
        cap="PyTorch = tensors + autograd + modules + optimizers, GPU-accelerated.",
        paras=[
            "PyTorch packages the ideas so far into reusable machinery: <strong>tensors</strong> (n-dimensional arrays that track gradients), <strong>nn.Module</strong> (parameter bookkeeping), <strong>optimizers</strong>, and GPU acceleration. A single <code>loss.backward()</code> runs the same reverse-mode autodiff you built in #05, but over whole tensors; <code>optimizer.step()</code> applies the update rule to every parameter.",
            "The optimizer matters: plain SGD uses one global learning rate, while <strong>Adam/AdamW</strong> keep per-parameter running estimates of the gradient's mean and variance to adapt each step size &mdash; far more robust for deep networks. This abstraction is exactly what frees the next steps to build something as intricate as a transformer without ever writing a gradient by hand.",
        ]),
    "07": dict(
        eq="Attention(Q, K, V) = softmax( QK&#7488; / &radic;d&#8342; + mask )&middot;V",
        cap="content-based weighted average; scaling by &radic;d&#8342; keeps the softmax gradients stable.",
        paras=[
            "Attention is a <strong>content-based weighted average</strong>. Each token produces a <em>query</em> (what it's looking for), a <em>key</em> (what it offers), and a <em>value</em> (what it passes on). The dot product of a query with every key gives relevance scores; softmax turns those into weights that sum to one; the output is the weighted sum of values. Scaling scores by &radic;d&#8342; keeps them from growing with dimension, which would otherwise saturate the softmax and kill its gradient.",
            "Two properties make it powerful. It has a <strong>global receptive field</strong> &mdash; any token can attend to any earlier token in a single step, unlike the fixed window of an MLP or the slow recurrence of an RNN. And it is <em>permutation-equivariant</em> (it sees a set, not a sequence), which is precisely why transformers must add positional information separately. The <strong>causal mask</strong> sets future scores to &minus;&infin; so a token can never attend to what it's trying to predict.",
        ]),
    "08": dict(
        eq="x = x + MHA(LN(x))&emsp;&middot;&emsp;x = x + FFN(LN(x))&emsp;(pre-norm residual)",
        cap="attention mixes across tokens; the FFN mixes across features; residuals + norm keep deep stacks trainable.",
        paras=[
            "A transformer block combines four ideas. <strong>Multi-head attention</strong> runs several attention operations in parallel low-dimensional subspaces, letting different heads specialize (one tracks the previous letter, another the start of the word), then concatenates them. The position-wise <strong>feed-forward network</strong> (two linear layers around a nonlinearity, ~4&times; wider) gives each token nonlinear processing on its own.",
            "The other two ideas make depth <em>trainable</em>. <strong>Residual connections</strong> (<code>x = x + sublayer(x)</code>) create identity shortcuts so gradients flow through dozens of layers without vanishing, and each block only has to learn a small correction. <strong>LayerNorm</strong> normalizes each token's vector to a stable scale; applying it <em>before</em> each sublayer (pre-norm) trains more stably than after. The slogan: attention mixes information <em>across</em> tokens, the FFN processes <em>each</em> token &mdash; gather, then think, repeated N times.",
        ]),
    "09": dict(
        eq="h = blocks( E_tok[x] + E_pos )&emsp;&middot;&emsp;logits = h&middot;W&#7488;&emsp;&middot;&emsp;L = &minus;&Sigma; log p(next | context)",
        cap="decoder-only GPT: token + position embeddings, N causal blocks, next-token cross-entropy.",
        paras=[
            "A GPT is a <strong>decoder-only transformer</strong>. Tokens become embeddings; because attention is order-blind, a learned <strong>positional embedding</strong> is added so the model knows <em>where</em> each token sits. The sum flows through N stacked causal blocks, a final norm, and a linear head that produces a probability distribution over the vocabulary for every position.",
            "Training uses <strong>next-token cross-entropy</strong> with <em>teacher forcing</em>: thanks to the causal mask, the model predicts the next token at <em>every</em> position simultaneously in one forward pass, and the loss averages over all of them &mdash; extremely efficient. Generation reverses this: sample a token, append it, repeat (autoregression). This is the entire GPT recipe; <em>scaling laws</em> say the loss falls predictably as you add parameters, data, and compute &mdash; the only real difference between this and GPT-4.",
        ]),
    "10": dict(
        eq="p &prop; exp(logits / T)&emsp;&middot;&emsp;top-k: keep k highest&emsp;&middot;&emsp;top-p: smallest set with &Sigma;p &ge; p",
        cap="decoding shapes the model's own distribution; the weights never change.",
        paras=[
            "Once trained, the model gives a distribution over the next token; <strong>decoding</strong> decides how to pick from it. Greedy (argmax) maximizes each local step but produces repetitive, globally sub-optimal text, and beam search &mdash; great for translation &mdash; degenerates for open-ended generation. So we sample, but shape the distribution first.",
            "<strong>Temperature</strong> T divides the logits before softmax: T&nbsp;&lt;&nbsp;1 sharpens toward the top choices (safer, more repetitive), T&nbsp;&gt;&nbsp;1 flattens them (wilder, more diverse), and T&rarr;0 recovers greedy. <strong>Truncation</strong> then trims the unreliable tail &mdash; <em>top-k</em> keeps the k most likely tokens, <em>top-p</em> (nucleus) keeps the smallest set whose probability mass exceeds p, adapting how many options it considers to how confident the model is. None of this touches the weights; it's purely how you read the same model.",
        ]),
    "11": dict(
        eq="lr(t): warmup &rarr; cosine decay&emsp;&middot;&emsp;clip &Vert;g&Vert; &le; c&emsp;&middot;&emsp;AdamW: decoupled weight decay",
        cap="the engineering that lets a run survive hours, crashes, and bad batches.",
        paras=[
            "Real training needs more than a loop. A <strong>learning-rate schedule</strong> &mdash; a linear <em>warmup</em> followed by <em>cosine decay</em> &mdash; starts small to avoid instability when early gradients are large and poorly conditioned, then anneals to refine the solution. <strong>Gradient clipping</strong> caps the gradient norm so a single pathological batch can't blow the weights to infinity.",
            "<strong>Checkpointing</strong> the model, optimizer state, and step count makes a run <em>resumable</em> &mdash; essential for multi-day training and crash recovery &mdash; while tracking the best validation loss guards against overfitting. <strong>AdamW</strong> decouples weight decay from the gradient update, a small change that consistently improves generalization. These are the unglamorous details that separate a toy loop from a real training run.",
        ]),
    "12": dict(
        eq="maximize  P(response | instruction)&emsp;&middot;&emsp;loss on response tokens only (mask the prompt)",
        cap="supervised fine-tuning turns a text-continuer into an instruction-follower.",
        paras=[
            "A pretrained <strong>base model</strong> models P(text) &mdash; it <em>continues</em> whatever you give it, it doesn't obey. <strong>Supervised fine-tuning (SFT)</strong> keeps training that same model on curated (instruction, response) pairs wrapped in a fixed template, teaching the conditional distribution P(response | instruction). It's still ordinary next-token cross-entropy &mdash; only the data has changed.",
            "The crucial trick is <strong>loss masking</strong>: gradients are computed only on the <em>response</em> tokens (the prompt's labels are ignored), so the model learns to <em>answer</em> a question rather than to generate more questions. This single step &mdash; one nudge on top of a base model &mdash; is what turned GPT-3 into InstructGPT and, ultimately, ChatGPT.",
        ]),
    "13": dict(
        eq="L = &minus;log &sigma;( &beta;[ (log &pi;_&theta;(y_w) &minus; log &pi;_ref(y_w)) &minus; (log &pi;_&theta;(y_l) &minus; log &pi;_ref(y_l)) ] )",
        cap="DPO: one classification loss over (chosen y_w, rejected y_l) pairs &mdash; no reward model, no RL.",
        paras=[
            "Classic <strong>RLHF</strong> aligns a model in three stages: collect human preferences, train a separate <em>reward model</em>, then optimize the policy against it with reinforcement learning (PPO) under a KL penalty. It works but is heavy and unstable &mdash; three models and an RL loop.",
            "<strong>DPO</strong>'s insight is that the RLHF objective has a closed-form optimum linking the reward to the log-ratio between the policy and a frozen <em>reference</em> model. That lets you skip the reward model and the RL entirely and optimize preferences <em>directly</em> with a simple binary-classification loss on (chosen, rejected) pairs. The coefficient &beta; controls how far the policy may drift from the reference (an implicit KL constraint). Same alignment effect, far simpler and more stable &mdash; today's stacks chain SFT &rarr; DPO &rarr; GRPO/RLVR for reasoning.",
        ]),
    "14": dict(
        eq="RoPE: rotate(q,k) by m&theta;&emsp;&middot;&emsp;RMSNorm(x)=x&middot;g/&radic;(mean&nbsp;x&sup2;+&epsilon;)&emsp;&middot;&emsp;SwiGLU=(SiLU(xW&#8321;)&odot;xW&#8323;)W&#8322;",
        cap="the four upgrades behind Llama / Mistral / Qwen / DeepSeek.",
        paras=[
            "Four refinements separate a 2017 transformer from a 2026 one. <strong>RoPE</strong> encodes position by <em>rotating</em> the query and key vectors by an angle proportional to their position; because a rotation's dot product depends only on the <em>relative</em> offset, the model generalizes to longer contexts and needs no learned position table. <strong>RMSNorm</strong> drops LayerNorm's mean-subtraction and bias, normalizing by the root-mean-square alone &mdash; cheaper, equally stable.",
            "<strong>SwiGLU</strong> replaces the ReLU feed-forward with a <em>gated</em> unit: one branch decides &lsquo;how much&rsquo; and multiplies the other, giving more expressive power per parameter. <strong>GQA</strong> (grouped-query attention) lets several query heads share one set of key/value heads, interpolating between full multi-head (best quality) and multi-query (smallest cache). Together they reach a lower loss with <em>fewer</em> parameters &mdash; which is exactly why the whole field adopted them.",
        ]),
    "15": dict(
        eq="memory = 2 &middot; L &middot; n_kv &middot; d_head &middot; T&emsp;&middot;&emsp;work:  na&iuml;ve &asymp; T&sup2;/2  &rarr;  cached = T",
        cap="past keys/values never change, so cache them instead of recomputing.",
        paras=[
            "During autoregressive decoding, the keys and values of all previous tokens are <em>fixed</em> once computed &mdash; yet the naive loop recomputes them at every step, costing O(T&sup2;) for a T-token generation. The <strong>KV cache</strong> stores each layer's past K and V and simply appends the new token's K/V each step, so a step is O(1) projection plus attention over the cache &mdash; making the whole generation O(T).",
            "The cache grows linearly with context length and, at long contexts, becomes the dominant consumer of inference memory (often larger than the weights). This is where #14's <strong>GQA</strong> pays off concretely: fewer key/value heads means a proportionally smaller cache. The output is bit-for-bit identical to the naive loop &mdash; it's the same mathematics, just not recomputed.",
        ]),
    "16": dict(
        eq="rotate by orthogonal Q  (&langle;Qa, Qb&rangle; = &langle;a, b&rangle;)  &rarr;  quantize  &rarr;  unrotate",
        cap="a rotation preserves dot products while spreading outliers, so low bits suffice.",
        paras=[
            "Since cache memory is proportional to bits-per-value, low-bit <strong>quantization</strong> shrinks it directly. But per-tensor low-bit quantization fails on heavy-tailed data: a few <em>outlier coordinates</em> stretch the min&ndash;max range, forcing every ordinary value onto a coarse grid. Real transformer K/V are full of such outlier channels.",
            "<strong>TurboQuant</strong>'s trick is to rotate each vector by a random orthogonal (or Hadamard) matrix <em>before</em> quantizing. A rotation preserves lengths and &mdash; crucially &mdash; inner products, so attention scores are unchanged if queries are rotated the same way; meanwhile, by concentration of measure, it spreads the outliers' energy across all coordinates, eliminating them so low bits quantize cleanly. PolarQuant plus a residual-correction stage reaches ~3 bits, ~6&times; smaller and ~8&times; faster. The honest caveat (shown in the output): a Gaussian is rotation-invariant, so the trick only helps when outliers actually exist &mdash; which, in real models, they do.",
        ]),
}


VISUALS = {
    "00": [
        dict(file="assets/visuals/00-1-xor-corners.webp",
             title="XOR corners",
             caption="The matching labels sit on opposite corners, so one straight split cannot solve the task.",
             alt="Four glowing XOR corner points with crossed class structure."),
        dict(file="assets/visuals/00-2-hidden-features.webp",
             title="Hidden features",
             caption="A hidden layer bends the input space into features a final neuron can separate.",
             alt="Small neural network bending input signals through a hidden layer."),
        dict(file="assets/visuals/00-3-output-prediction.webp",
             title="Output prediction",
             caption="Forward pass, loss, and backprop nudge the weights until the XOR outputs are correct.",
             alt="Output neuron receiving weighted hidden signals and resolving binary predictions."),
    ],
    "01": [
        dict(file="assets/visuals/01-1-pair-stream.webp",
             title="Pair extraction",
             caption="The model reads adjacent character pairs and asks which symbol tends to come next.",
             alt="Character-like tiles flowing into adjacent pair arrows."),
        dict(file="assets/visuals/01-2-count-matrix.webp",
             title="Count table",
             caption="Every current-character row becomes a distribution after counts are normalized.",
             alt="Glowing square transition matrix filling with count marks."),
        dict(file="assets/visuals/01-3-sample-path.webp",
             title="Sampling path",
             caption="Generation repeatedly samples from the row for the current character.",
             alt="Curving sampling path stepping through a matrix of character states."),
    ],
    "02": [
        dict(file="assets/visuals/02-1-one-hot-weights.webp",
             title="One-hot input",
             caption="A one-hot character selects the row of weights that scores possible next characters.",
             alt="One-hot vector feeding a glowing weight matrix."),
        dict(file="assets/visuals/02-2-softmax-probs.webp",
             title="Softmax probabilities",
             caption="Logits become probabilities; cross-entropy rewards the true next token.",
             alt="Logit bars flowing through a softmax funnel into probability bars."),
        dict(file="assets/visuals/02-3-descent-to-counts.webp",
             title="Learned counts",
             caption="Gradient descent moves the weights toward the same transition pattern counting found.",
             alt="Optimization path descending toward a transition distribution."),
    ],
    "03": [
        dict(file="assets/visuals/03-1-raw-tokens.webp",
             title="Raw symbols",
             caption="Text first becomes small reversible pieces: characters or bytes.",
             alt="Many small text-like tiles split into raw tokens."),
        dict(file="assets/visuals/03-2-pair-merges.webp",
             title="Frequent merges",
             caption="BPE repeatedly merges the most frequent adjacent pair into a larger token.",
             alt="Repeated adjacent token pairs combining into larger glowing chunks."),
        dict(file="assets/visuals/03-3-compressed-context.webp",
             title="Shorter context",
             caption="Common chunks use fewer tokens, so more real text fits into the model window.",
             alt="Compressed token chunks fitting into a context-window rail."),
    ],
    "04": [
        dict(file="assets/visuals/04-1-embedding-lookup.webp",
             title="Embedding lookup",
             caption="Token IDs index rows in an embedding table instead of staying one-hot.",
             alt="Token ID tiles looking up dense vector columns in an embedding table."),
        dict(file="assets/visuals/04-2-context-window.webp",
             title="Context window",
             caption="Several previous embeddings are packed together so the model sees local history.",
             alt="Multiple embedding vectors aligned inside a fixed context window."),
        dict(file="assets/visuals/04-3-mlp-prediction.webp",
             title="MLP prediction",
             caption="The flattened context feeds an MLP that predicts the next character distribution.",
             alt="Context vectors flowing through layered neural network panels to output bars."),
    ],
    "05": [
        dict(file="assets/visuals/05-1-forward-graph.webp",
             title="Forward graph",
             caption="Each scalar operation records the values that produced it.",
             alt="Scalar computation graph flowing from inputs to a final output node."),
        dict(file="assets/visuals/05-2-local-rules.webp",
             title="Local rules",
             caption="Every operation stores a tiny derivative rule for its immediate parents.",
             alt="Computation nodes with small derivative-rule modules attached."),
        dict(file="assets/visuals/05-3-backward-gradients.webp",
             title="Reverse pass",
             caption="Backward walks the graph in reverse, accumulating gradients at the parameters.",
             alt="Amber gradient arrows flowing backward through a computation graph."),
    ],
    "06": [
        dict(file="assets/visuals/06-1-manual-gradients.webp",
             title="Manual gradients",
             caption="The hand-derived derivative wiring grows messy as models get larger.",
             alt="Dense manual gradient wiring around tensor-like blocks."),
        dict(file="assets/visuals/06-2-tensor-autograd.webp",
             title="Tensor autograd",
             caption="PyTorch records tensor operations and runs reverse-mode autodiff over the graph.",
             alt="Tensor blocks flowing through reusable layers with an autograd graph."),
        dict(file="assets/visuals/06-3-optimizer-step.webp",
             title="Optimizer update",
             caption="After backward, the optimizer applies the parameter update in one step.",
             alt="Optimizer loop updating model parameter blocks after backward signals."),
    ],
    "07": [
        dict(file="assets/visuals/07-1-qkv-streams.webp",
             title="Q, K, V streams",
             caption="Each token emits what it seeks, what it offers, and what it will pass on.",
             alt="Token columns emitting query, key, and value streams."),
        dict(file="assets/visuals/07-2-causal-mask.webp",
             title="Causal mask",
             caption="The lower triangle lets each token attend only to itself and the past.",
             alt="Lower-triangular attention matrix with future positions blocked."),
        dict(file="assets/visuals/07-3-weighted-values.webp",
             title="Weighted values",
             caption="Attention weights mix value vectors into context-aware token representations.",
             alt="Weighted value vectors merging into updated token representations."),
    ],
    "08": [
        dict(file="assets/visuals/08-1-attention-heads.webp",
             title="Many heads",
             caption="Multiple attention heads inspect the same sequence from different subspaces.",
             alt="Parallel attention heads running across token rows."),
        dict(file="assets/visuals/08-2-residual-norm.webp",
             title="Residual and norm",
             caption="Skip paths and normalization keep the block trainable while sublayers do useful work.",
             alt="Transformer sublayers wrapped by residual arcs and normalization rings."),
        dict(file="assets/visuals/08-3-stacked-blocks.webp",
             title="Stackable block",
             caption="Shape in equals shape out, so GPT can stack the same block repeatedly.",
             alt="Identical transformer blocks stacked vertically with matching input and output shapes."),
    ],
    "09": [
        dict(file="assets/visuals/09-1-token-position.webp",
             title="Token plus position",
             caption="Token identity and position are combined before the transformer stack.",
             alt="Token tiles and position rhythm combining into model input."),
        dict(file="assets/visuals/09-2-gpt-stack.webp",
             title="GPT stack",
             caption="Causal transformer blocks learn next-token patterns from the training corpus.",
             alt="Stacked transformer blocks receiving a flowing manuscript-like dataset."),
        dict(file="assets/visuals/09-3-next-token-output.webp",
             title="Next-token head",
             caption="Generation samples one next token, appends it, and repeats.",
             alt="Output probability beams selecting abstract next-token tiles."),
    ],
    "10": [
        dict(file="assets/visuals/10-1-greedy-loop.webp",
             title="Greedy loop",
             caption="Always taking the top token is locally safe but can fall into repetition.",
             alt="Tallest probability bar looping back into repeated token choices."),
        dict(file="assets/visuals/10-2-temperature-shape.webp",
             title="Temperature",
             caption="Temperature sharpens or flattens the distribution before sampling.",
             alt="Probability bars reshaped from sharp to broad by heated particle flow."),
        dict(file="assets/visuals/10-3-topk-topp-filter.webp",
             title="Top-k and top-p",
             caption="Truncation removes the unreliable tail while keeping enough variety.",
             alt="Probability distribution with tail candidates filtered behind a cutoff gate."),
    ],
    "11": [
        dict(file="assets/visuals/11-1-training-loop.webp",
             title="Training loop",
             caption="Batches, loss, gradients, and optimizer updates form the reliable training cycle.",
             alt="Closed loop of batches flowing through model, loss, gradients, and update."),
        dict(file="assets/visuals/11-2-lr-and-clipping.webp",
             title="Schedule and clipping",
             caption="Warmup, decay, and gradient clipping prevent unstable updates.",
             alt="Learning-rate curve beside a clipped gradient spike."),
        dict(file="assets/visuals/11-3-checkpoint-resume.webp",
             title="Checkpointing",
             caption="Model, optimizer, and step state are saved so training can resume exactly.",
             alt="Checkpoint storage blocks preserving model and optimizer state."),
    ],
    "12": [
        dict(file="assets/visuals/12-1-instruction-pairs.webp",
             title="Instruction pairs",
             caption="Supervised examples show the model how an instruction should be answered.",
             alt="Prompt and response cards flowing into a base language model."),
        dict(file="assets/visuals/12-2-loss-mask.webp",
             title="Loss mask",
             caption="Only response tokens send loss backward; prompt tokens are context.",
             alt="Prompt side masked while answer side sends training signal backward."),
        dict(file="assets/visuals/12-3-assistant-path.webp",
             title="Assistant behavior",
             caption="The tuned model follows the requested answer path instead of simply continuing text.",
             alt="Helpful response path glowing brighter than open-ended continuation paths."),
    ],
    "13": [
        dict(file="assets/visuals/13-1-chosen-rejected.webp",
             title="Preference pair",
             caption="DPO trains on a chosen response and a rejected response from the same prompt.",
             alt="Two response paths, one bright chosen path and one dim rejected path."),
        dict(file="assets/visuals/13-2-policy-reference.webp",
             title="Policy and reference",
             caption="The trainable policy is compared against a frozen reference copy.",
             alt="Two model blocks side by side emitting comparable probability streams."),
        dict(file="assets/visuals/13-3-preference-gap.webp",
             title="Wider gap",
             caption="Optimization increases the chosen-vs-rejected log-probability margin.",
             alt="Chosen response probability rising while rejected probability falls."),
    ],
    "14": [
        dict(file="assets/visuals/14-1-rope-rotation.webp",
             title="RoPE",
             caption="Rotary position geometry makes attention depend on relative offsets.",
             alt="Query and key vectors rotating around circular position geometry."),
        dict(file="assets/visuals/14-2-norm-swiglu.webp",
             title="RMSNorm and SwiGLU",
             caption="Normalization stabilizes the stream while gated feed-forwards add efficient capacity.",
             alt="Normalized residual stream passing through a glowing gated feed-forward unit."),
        dict(file="assets/visuals/14-3-gqa-sharing.webp",
             title="GQA",
             caption="Several query heads share fewer key/value heads, shrinking the inference cache.",
             alt="Grouped query heads sharing a smaller set of key and value heads."),
    ],
    "15": [
        dict(file="assets/visuals/15-1-naive-recompute.webp",
             title="Naive recompute",
             caption="Without a cache, generation repeatedly re-encodes the whole growing context.",
             alt="Repeated triangular recomputation work for a growing token sequence."),
        dict(file="assets/visuals/15-2-cache-append.webp",
             title="Append K/V",
             caption="The cache stores past keys and values, then appends only the new token's pair.",
             alt="Layered key-value cache shelves receiving one new block at a time."),
        dict(file="assets/visuals/15-3-smaller-gqa-cache.webp",
             title="Smaller cache",
             caption="GQA reduces the number of key/value heads that must be stored.",
             alt="Compact grouped-query cache feeding the same output stream."),
    ],
    "16": [
        dict(file="assets/visuals/16-1-outlier-channels.webp",
             title="Outlier channels",
             caption="A few huge coordinates stretch the quantization range and waste low-bit precision.",
             alt="Vector histogram with a few tall amber outlier channels dominating the range."),
        dict(file="assets/visuals/16-2-rotation-spread.webp",
             title="Rotate and spread",
             caption="An orthogonal rotation spreads outlier energy while preserving attention dot products.",
             alt="Outlier vector passing through a rotation plane into a more even distribution."),
        dict(file="assets/visuals/16-3-three-bit-cache.webp",
             title="3-bit cache",
             caption="After rotation, the cache can be stored in very few bits with better fidelity.",
             alt="Compact quantized cache blocks feeding an attention layer."),
    ],
}


CSS = r"""
:root{
  --bg:#0a0c10; --bg2:#0d1016; --panel:#13161f; --panel2:#181c27;
  --ink:#ece8de; --muted:#9aa1ad; --faint:#5d6470;
  --line:rgba(255,255,255,.07); --line2:rgba(255,255,255,.12);
  --accent:#5ef0c0; --accent-dim:#2f8f74; --amber:#f2b65e; --rose:#f08a8a;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
  --serif:'Newsreader',Georgia,serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--serif); font-size:18px; line-height:1.7;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
  background-image:
    linear-gradient(var(--line) 1px,transparent 1px),
    linear-gradient(90deg,var(--line) 1px,transparent 1px);
  background-size:64px 64px; background-position:center top;
}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(120% 80% at 50% -10%,rgba(94,240,192,.10),transparent 60%);}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:0 28px;position:relative;z-index:1}
.mono{font-family:var(--mono)}

/* top bar */
.bar{position:sticky;top:0;z-index:50;backdrop-filter:blur(10px);
  background:rgba(10,12,16,.72);border-bottom:1px solid var(--line)}
.bar .wrap{display:flex;align-items:center;justify-content:space-between;height:62px}
.brand{font-family:var(--mono);font-weight:600;letter-spacing:.02em;font-size:15px}
.brand b{color:var(--accent)}
.bar nav{display:flex;gap:26px;font-family:var(--mono);font-size:13px;letter-spacing:.04em;text-transform:uppercase}
.bar nav a{color:var(--muted)} .bar nav a:hover{color:var(--ink);text-decoration:none}
@media(max-width:760px){.bar nav{display:none}}

/* hero */
.hero{position:relative;padding:96px 0 70px;overflow:hidden}
#net{position:absolute;inset:0;width:100%;height:100%;z-index:0;opacity:.55}
.hero .wrap{position:relative;z-index:2}
.kicker{font-family:var(--mono);font-size:13px;letter-spacing:.34em;text-transform:uppercase;color:var(--accent);margin:0 0 22px}
h1.title{font-family:var(--mono);font-weight:700;line-height:.96;letter-spacing:-.01em;
  font-size:clamp(44px,9vw,104px);margin:0}
h1.title .l2{display:block;color:var(--faint)}
h1.title .l2 em{font-style:normal;color:var(--ink)}
.lede{font-size:clamp(19px,2.4vw,25px);color:#cfd3da;max-width:640px;margin:28px 0 0}
.lede b{color:var(--ink)}
.cta{display:flex;gap:14px;margin-top:38px;flex-wrap:wrap}
.btn{font-family:var(--mono);font-size:14px;letter-spacing:.03em;padding:14px 22px;border-radius:2px;
  border:1px solid var(--line2);color:var(--ink);transition:.18s;background:transparent}
.btn:hover{text-decoration:none;border-color:var(--accent);color:var(--accent)}
.btn.solid{background:var(--accent);color:#06231b;border-color:var(--accent);font-weight:600}
.btn.solid:hover{background:#7ff6cf;color:#06231b}
.stats{display:flex;gap:40px;margin-top:54px;flex-wrap:wrap}
.stat .num{font-family:var(--mono);font-size:34px;font-weight:700;color:var(--accent);line-height:1}
.stat .lab{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-top:8px}

/* section heads */
.section{padding:64px 0;border-top:1px solid var(--line)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.28em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h2.sec{font-family:var(--mono);font-weight:600;font-size:clamp(28px,4vw,40px);margin:0 0 18px;letter-spacing:-.01em}
.sec-lede{color:#cbd0d8;max-width:720px;font-size:20px}

/* acts */
.acts{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:34px}
@media(max-width:820px){.acts{grid-template-columns:1fr}}
.act{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:24px}
.act h3{font-family:var(--mono);font-size:15px;margin:0 0 10px;color:var(--accent)}
.act p{margin:0;color:#c2c7d0;font-size:16px}

/* ladder layout */
.ladder{display:grid;grid-template-columns:230px 1fr;gap:48px;align-items:start}
@media(max-width:900px){.ladder{grid-template-columns:1fr}}
.index{position:sticky;top:86px;font-family:var(--mono);font-size:13px}
@media(max-width:900px){.index{display:none}}
.index .ph{color:var(--faint);text-transform:uppercase;letter-spacing:.14em;font-size:11px;margin:18px 0 8px}
.index a{display:flex;gap:10px;color:var(--muted);padding:4px 0;transition:.15s}
.index a:hover{color:var(--ink);text-decoration:none}
.index a .nn{color:var(--faint)}
.index a.on{color:var(--accent)} .index a.on .nn{color:var(--accent)}

/* step card */
.step{padding:30px 0 6px;scroll-margin-top:80px;opacity:0;transform:translateY(18px);
  transition:opacity .6s ease,transform .6s ease}
.step.in{opacity:1;transform:none}
.phase-div{font-family:var(--mono);font-size:12px;letter-spacing:.26em;text-transform:uppercase;
  color:var(--accent);padding:46px 0 6px;border-top:1px solid var(--line);margin-top:24px}
.phase-div:first-of-type{border-top:0}
.step .head{display:flex;align-items:baseline;gap:18px}
.step .nn{font-family:var(--mono);font-size:46px;font-weight:700;color:var(--faint);line-height:1}
.step h3{font-family:var(--mono);font-weight:600;font-size:25px;margin:0}
.step .tag{font-family:var(--mono);font-size:13px;color:var(--accent);margin:6px 0 0}
.step .file{font-family:var(--mono);font-size:12px;color:var(--muted);margin-left:auto;
  border:1px solid var(--line);padding:5px 10px;border-radius:2px;white-space:nowrap}
.step .body{margin-top:16px;max-width:760px}
.step .body p{margin:0 0 14px}
code{font-family:var(--mono);font-size:.86em;background:rgba(94,240,192,.10);color:var(--accent);
  padding:2px 6px;border-radius:3px}

/* key formula */
.eq{font-family:var(--mono);font-size:14px;line-height:1.7;background:rgba(94,240,192,.05);
  border-left:2px solid var(--accent);padding:13px 18px;margin:18px 0 0;border-radius:0 4px 4px 0;
  color:#d4efe6;overflow-x:auto;white-space:nowrap}
.eq .cap{display:block;font-size:12px;color:var(--muted);margin-top:7px;white-space:normal}

/* theory block */
.theory{margin-top:18px;padding:18px 20px;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;max-width:760px}
.theory .lab{font-family:var(--mono);font-size:11px;letter-spacing:.26em;text-transform:uppercase;
  color:var(--amber);margin:0 0 12px;display:flex;align-items:center;gap:9px}
.theory .lab::before{content:"";width:18px;height:1px;background:var(--amber);display:inline-block}
.theory p{margin:0 0 13px;color:#c6cbd4;font-size:16.5px}
.theory p:last-child{margin-bottom:0}

/* terminal output */
.term{background:#070809;border:1px solid var(--line2);border-radius:6px;margin:18px 0 0;overflow:hidden}
.term .top{display:flex;align-items:center;gap:8px;padding:9px 14px;border-bottom:1px solid var(--line);
  font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.dot{width:9px;height:9px;border-radius:50%;background:#2a2f38}
.dot.g{background:var(--accent)}
.term pre{margin:0;padding:16px 18px;font-family:var(--mono);font-size:13.5px;line-height:1.55;
  color:#bfe9da;overflow-x:auto;white-space:pre}
.term .out-lab{margin-left:auto;color:var(--faint)}

/* code disclosure */
details.src{margin:14px 0 0;border:1px solid var(--line);border-radius:6px;background:var(--panel);overflow:hidden}
details.src summary{cursor:pointer;list-style:none;font-family:var(--mono);font-size:13px;color:var(--muted);
  padding:12px 16px;display:flex;align-items:center;gap:10px;transition:.15s;user-select:none}
details.src summary::-webkit-details-marker{display:none}
details.src summary:hover{color:var(--accent)}
details.src summary .chev{transition:.2s;color:var(--faint)}
details.src[open] summary .chev{transform:rotate(90deg);color:var(--accent)}
details.src[open] summary{border-bottom:1px solid var(--line);color:var(--ink)}
details.src pre{margin:0;max-height:540px;overflow:auto;padding:18px;font-size:13px;line-height:1.6;
  background:var(--bg2)}
details.src code{background:none;color:#d6dae0;padding:0;font-size:13px}

/* prism token colors */
.token.comment{color:#5d6470;font-style:italic}
.token.keyword{color:#f2b65e}
.token.string,.token.triple-quoted-string{color:#9fe6c4}
.token.number,.token.boolean{color:#f08a8a}
.token.function{color:#5ef0c0}
.token.class-name,.token.builtin{color:#8fd0ff}
.token.operator,.token.punctuation{color:#9aa1ad}
.token.decorator{color:#f2b65e}

/* SOTA */
.sota-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:30px}
@media(max-width:820px){.sota-grid{grid-template-columns:1fr}}
.cmp{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.cmp .h{font-family:var(--mono);font-size:13px;letter-spacing:.06em;padding:12px 16px;border-bottom:1px solid var(--line);color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:15px}
td,th{padding:11px 16px;text-align:left;border-bottom:1px solid var(--line)}
th{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);font-weight:400}
tr:last-child td{border-bottom:0}
td.old{color:var(--muted)} td.new{color:var(--accent);font-family:var(--mono);font-size:13.5px}
td.k{font-family:var(--mono);font-size:13px;color:var(--ink)}
.pill{display:inline-block;font-family:var(--mono);font-size:11px;color:var(--amber);border:1px solid rgba(242,182,94,.3);
  padding:2px 8px;border-radius:20px;margin:0 6px 6px 0}
.stack{font-family:var(--mono);font-size:15px;margin-top:24px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.stack .n{background:var(--panel2);border:1px solid var(--line2);padding:10px 16px;border-radius:3px}
.stack .ar{color:var(--accent)}

/* run */
.run pre{background:#070809;border:1px solid var(--line2);border-radius:6px;padding:18px 20px;
  font-family:var(--mono);font-size:14px;color:#bfe9da;overflow-x:auto}
.run .c{color:var(--faint)}
.run .p{color:var(--accent)}

footer{border-top:1px solid var(--line);padding:46px 0 80px;color:var(--muted);font-family:var(--mono);font-size:13px}
footer b{color:var(--ink)}
.sources{font-size:13px;color:var(--muted);margin-top:18px;line-height:1.9}
.sources a{color:var(--accent-dim)}

/* brand as link */
.bar a.brand{color:var(--ink)} .bar a.brand:hover{text-decoration:none}

/* ladder grid (index) */
.phlabel{font-family:var(--mono);font-size:12px;letter-spacing:.26em;text-transform:uppercase;
  color:var(--accent);margin:40px 0 0;padding-top:26px;border-top:1px solid var(--line)}
.phlabel.first{border-top:0;padding-top:0;margin-top:26px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(258px,1fr));gap:14px;margin-top:16px}
.card{position:relative;display:block;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;padding:18px 18px 20px;overflow:hidden;transition:transform .16s,border-color .16s}
.card:hover{border-color:var(--accent);transform:translateY(-3px);text-decoration:none}
.card .thumb{display:block;width:calc(100% + 36px);height:132px;margin:-18px -18px 14px;
  object-fit:cover;border-bottom:1px solid var(--line);background:#070809;filter:saturate(.95)}
.card .cn{font-family:var(--mono);font-size:13px;color:var(--faint)}
.card h3{font-family:var(--mono);font-size:16px;font-weight:600;margin:7px 0;color:var(--ink)}
.card:hover h3{color:var(--accent)}
.card p{margin:0;color:var(--muted);font-size:13.5px;font-family:var(--mono);line-height:1.5}
.card .go{position:absolute;top:16px;right:18px;color:var(--faint);font-family:var(--mono)}
.card:hover .go{color:var(--accent)}

/* step page */
.stephero{padding:54px 0 30px;border-bottom:1px solid var(--line)}
.crumb{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--accent);margin-bottom:20px}
.sh-head{display:flex;align-items:baseline;gap:22px}
.sh-head .nn{font-family:var(--mono);font-size:clamp(44px,8vw,72px);font-weight:700;
  color:var(--faint);line-height:.85}
.sh-head h1{font-family:var(--mono);font-weight:700;font-size:clamp(27px,4.5vw,44px);
  margin:0;letter-spacing:-.01em}
.sh-head .tag{font-family:var(--mono);font-size:14px;color:var(--accent);margin-top:9px}
.stephero .file{display:inline-block;margin-top:20px;font-family:var(--mono);font-size:12px;
  color:var(--muted);border:1px solid var(--line);padding:5px 10px;border-radius:2px}
.nb-btn{display:inline-block;margin:20px 0 0 10px;font-family:var(--mono);font-size:12px;
  color:var(--accent);border:1px solid var(--accent-dim);padding:5px 12px;border-radius:2px}
.nb-btn:hover{background:rgba(94,240,192,.08);text-decoration:none}
.stepwrap{max-width:830px}
.lead p{font-size:18.5px;line-height:1.75;color:#d7dbe2;margin:0 0 15px}
.visuals{margin:30px 0 4px}
.visual-kicker{font-family:var(--mono);font-size:11px;letter-spacing:.24em;text-transform:uppercase;
  color:var(--amber);margin:0 0 12px}
.visual-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:820px){.visual-grid{grid-template-columns:1fr}}
.visual-card{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.visual-card img{display:block;width:100%;height:auto;background:#070809}
.visual-card figcaption{padding:12px 13px 14px;color:#bfc5cf;font-size:14.5px;line-height:1.45}
.visual-card figcaption strong{display:block;font-family:var(--mono);font-size:12px;color:var(--accent);
  margin:0 0 6px;letter-spacing:.05em}
.srch{font-family:var(--mono);font-size:12px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--amber);margin:32px 0 0}
.srch span{color:var(--muted);letter-spacing:0;text-transform:none}
.term pre code{color:#cdd2da}
.pager{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:46px}
.pager a{display:block;padding:18px 20px;border:1px solid var(--line);border-radius:6px;transition:border-color .16s}
.pager a:hover{border-color:var(--accent);text-decoration:none}
.pager .dir{font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--faint)}
.pager .pt{font-family:var(--mono);color:var(--ink);font-size:15px;margin-top:7px}
.pager a:hover .pt{color:var(--accent)}
.pager .next{text-align:right}
@media(max-width:600px){.pager{grid-template-columns:1fr}}

/* animations */
.anim{margin:22px 0 0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:24px 22px}
.anim .alab{font-family:var(--mono);font-size:11px;letter-spacing:.24em;text-transform:uppercase;
  color:var(--accent);margin:0 0 20px;display:flex;align-items:center;gap:9px}
.anim .alab::before{content:"";width:18px;height:1px;background:var(--accent);display:inline-block}
.anim .acap{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin:18px auto 0;
  text-align:center;max-width:640px;line-height:1.65}
.anim video{display:block;max-width:100%;border-radius:6px;margin:0 auto;background:#000}
/* #07 causal-attention grid */
.attn-grid{display:grid;grid-template-columns:repeat(5,32px);gap:7px;justify-content:center}
.attn-grid .c{width:32px;height:32px;border-radius:5px;background:#0f1218;border:1px solid var(--line)}
.attn-grid .c.on{animation:attnpulse 3.6s ease-in-out infinite}
@keyframes attnpulse{0%,72%,100%{background:#0f1218;box-shadow:none}
  36%{background:var(--accent);box-shadow:0 0 12px rgba(94,240,192,.45)}}
/* #10 temperature bars */
.tbars{display:flex;align-items:flex-end;justify-content:center;gap:12px;height:112px}
.tbars .b{width:34px;height:var(--h1);border-radius:3px 3px 0 0;
  background:linear-gradient(180deg,var(--accent),var(--accent-dim));animation:temp 5s ease-in-out infinite}
@keyframes temp{0%,100%{height:var(--h1)}50%{height:var(--h2)}}
/* #15 kv-cache append */
.kv{display:flex;gap:8px;justify-content:center;min-height:44px;align-items:center}
.kv .k{width:34px;height:38px;border-radius:5px;border:1px solid var(--accent-dim);
  background:rgba(94,240,192,.08);display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:12px;color:var(--accent);opacity:0;animation:kvin 6s ease-in-out infinite}
@keyframes kvin{0%{opacity:0;transform:scale(.5)}8%{opacity:1;transform:scale(1)}90%{opacity:1}100%{opacity:0}}
/* #03 bpe merge */
.bpe{position:relative;height:40px;font-family:var(--mono);font-size:22px;text-align:center;color:var(--ink)}
.bpe span{position:absolute;left:0;right:0;top:0;opacity:0;animation:bpe 6s ease-in-out infinite}
@keyframes bpe{0%{opacity:0}4%{opacity:1}28%{opacity:1}33%{opacity:0}100%{opacity:0}}
@media(prefers-reduced-motion:reduce){.anim *{animation:none!important}}
"""


JS = r"""
// hero neural-net animation
(function(){
  const c=document.getElementById('net'); if(!c) return;
  const x=c.getContext('2d'); let W,H,nodes=[],edges=[],pulses=[];
  const layers=[4,6,6,5,3];
  function build(){
    W=c.width=c.offsetWidth*devicePixelRatio; H=c.height=c.offsetHeight*devicePixelRatio;
    nodes=[]; edges=[];
    const pad=W*0.08, gw=(W-pad*2)/(layers.length-1);
    layers.forEach((cnt,li)=>{
      for(let i=0;i<cnt;i++){
        const gy=H/(cnt+1)*(i+1);
        nodes.push({x:pad+gw*li,y:gy,l:li,p:Math.random()*Math.PI*2});
      }
    });
    for(let a=0;a<nodes.length;a++)for(let b=0;b<nodes.length;b++)
      if(nodes[b].l===nodes[a].l+1) edges.push([a,b]);
  }
  function spawn(){
    if(pulses.length<26 && edges.length){
      const e=edges[(Math.random()*edges.length)|0];
      pulses.push({a:e[0],b:e[1],t:0,s:0.012+Math.random()*0.02});
    }
  }
  function draw(){
    x.clearRect(0,0,W,H);
    x.lineWidth=1*devicePixelRatio;
    edges.forEach(e=>{const A=nodes[e[0]],B=nodes[e[1]];
      x.strokeStyle='rgba(255,255,255,0.05)';x.beginPath();x.moveTo(A.x,A.y);x.lineTo(B.x,B.y);x.stroke();});
    pulses.forEach(p=>{const A=nodes[p.a],B=nodes[p.b];p.t+=p.s;
      const px=A.x+(B.x-A.x)*p.t, py=A.y+(B.y-A.y)*p.t;
      const g=x.createRadialGradient(px,py,0,px,py,7*devicePixelRatio);
      g.addColorStop(0,'rgba(94,240,192,0.9)');g.addColorStop(1,'rgba(94,240,192,0)');
      x.fillStyle=g;x.beginPath();x.arc(px,py,7*devicePixelRatio,0,7);x.fill();});
    pulses=pulses.filter(p=>p.t<1);
    const t=Date.now()/900;
    nodes.forEach(n=>{const r=(2.4+Math.sin(t+n.p)*0.9)*devicePixelRatio;
      x.fillStyle='rgba(94,240,192,'+(0.5+Math.sin(t+n.p)*0.3)+')';
      x.beginPath();x.arc(n.x,n.y,r,0,7);x.fill();});
    if(Math.random()<0.32) spawn();
    requestAnimationFrame(draw);
  }
  build(); draw(); addEventListener('resize',build);
})();
"""


def slug(s):
    """Per-script page filename: 09_tiny_gpt.py -> 09_tiny_gpt.html."""
    return s["file"].replace(".py", ".html")


def eq_theory_html(s):
    t = THEORY.get(s["n"])
    parts = []
    if t and t.get("eq"):
        parts.append(f'<div class="eq">{t["eq"]}'
                     f'<span class="cap">{t.get("cap", "")}</span></div>')
    if t and t.get("paras"):
        paras = "".join(f"<p>{p}</p>" for p in t["paras"])
        parts.append(f'<div class="theory"><div class="lab">The theory</div>{paras}</div>')
    return "".join(parts)


def visual_gallery_html(s):
    """Three concept images for a lesson page."""
    visuals = VISUALS.get(s["n"], [])
    if not visuals:
        return ""
    cards = []
    for v in visuals:
        cards.append(
            '<figure class="visual-card">'
            f'<img src="{esc(v["file"])}" alt="{esc(v["alt"])}" loading="lazy" decoding="async">'
            f'<figcaption><strong>{esc(v["title"])}</strong>{esc(v["caption"])}</figcaption>'
            '</figure>')
    return (
        '<section class="visuals" aria-label="Visual walkthrough">'
        '<div class="visual-kicker">Visual walkthrough</div>'
        f'<div class="visual-grid">{"".join(cards)}</div>'
        '</section>')


HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="icon" href="data:,">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="styles.css">
</head>
<body>
"""

PRISM = ('\n<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>'
         '\n<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-clike.min.js"></script>'
         '\n<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>\n')


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _cell_outputs(cell):
    """Collect a code cell's text outputs (stdout, results, errors)."""
    chunks = []
    for o in cell.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            chunks.append("".join(o.get("text", [])))
        elif t in ("execute_result", "display_data"):
            tp = o.get("data", {}).get("text/plain", "")
            chunks.append("".join(tp) if isinstance(tp, list) else tp)
        elif t == "error":
            chunks.append("\n".join(o.get("traceback", [])))
    return _ANSI.sub("", "".join(chunks))


def _panel(label, inner, lit=True):
    dot = '<span class="dot g"></span>' if lit else '<span class="dot"></span>'
    return (f'<div class="term"><div class="top">{dot}<span class="dot"></span>'
            f'<span class="dot"></span><span class="out-lab">{label}</span></div>{inner}</div>')


def render_notebook(s):
    """Render the executed .ipynb as styled code-cell + output blocks (Jupyter-
    style). Returns '' if the notebook doesn't exist yet."""
    p = ROOT / s["file"].replace(".py", ".ipynb")
    if not p.exists():
        return ""
    nb = json.loads(p.read_text(encoding="utf-8"))
    out, n = [], 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        n += 1
        out.append(_panel(f"In [{n}]",
                          f'<pre><code class="language-python">{esc(src)}</code></pre>'))
        otxt = _cell_outputs(cell)
        if otxt.strip():
            out.append(_panel("Output", f"<pre>{esc(otxt.rstrip())}</pre>", lit=False))
    return "\n".join(out)


def _anim_attention():
    cells = []
    for r in range(5):
        for c in range(5):
            if c <= r:
                cells.append(f'<div class="c on" style="animation-delay:{r*0.18 + c*0.12:.2f}s"></div>')
            else:
                cells.append('<div class="c"></div>')
    return (f'<div class="attn-grid">{"".join(cells)}</div>'
            '<div class="acap">Causal attention: each row (a query token) lights up the tokens it '
            'may attend to &mdash; itself and every earlier token. The upper triangle stays dark: '
            'the future is masked.</div>')


def _anim_temperature():
    peak = [16, 32, 96, 46, 24, 15]
    flat = [50, 54, 58, 54, 50, 48]
    bars = "".join(f'<div class="b" style="--h1:{p}px;--h2:{f}px"></div>'
                   for p, f in zip(peak, flat))
    return (f'<div class="tbars">{bars}</div>'
            '<div class="acap">Temperature reshapes the same distribution without touching the '
            'weights: low T is peaked (confident, repetitive), high T is flat (diverse, riskier).</div>')


def _anim_kvcache():
    ks = "".join(f'<div class="k" style="animation-delay:{i*0.8:.1f}s">k{i}</div>' for i in range(6))
    return (f'<div class="kv">{ks}</div>'
            '<div class="acap">The KV cache grows by one entry per generated token &mdash; past '
            'keys/values are kept, never recomputed. That turns O(T&sup2;) decoding into O(T).</div>')


def _anim_bpe():
    stages = ["t &middot; h &middot; e &middot; t &middot; h &middot; e",
              "th &middot; e &middot; th &middot; e",
              "the &middot; the"]
    spans = "".join(f'<span style="animation-delay:{i*2}s">{t}</span>'
                    for i, t in enumerate(stages))
    return (f'<div class="bpe">{spans}</div>'
            '<div class="acap">Byte-Pair Encoding merges the most frequent adjacent pair, again and '
            'again: t,h &rarr; th &rarr; the. Fewer tokens carry the same text.</div>')


ANIM = {
    "03": _anim_bpe(),
    "07": _anim_attention(),
    "10": _anim_temperature(),
    "15": _anim_kvcache(),
}
ANIM_CAP = {
    "14": "RoPE encodes position by rotating each query/key vector by an angle proportional to its "
          "position. A rotation preserves the dot product, so attention scores depend only on the "
          "relative offset between tokens.",
}


def render_anim(s):
    """A motion block for a step: a rendered Manim <video> if present, else a
    lightweight CSS animation, else nothing."""
    vid = DOCS / "assets" / "anim" / f"{s['n']}.mp4"
    if vid.exists():
        inner = (f'<video src="assets/anim/{s["n"]}.mp4" autoplay loop muted playsinline></video>'
                 f'<div class="acap">{ANIM_CAP.get(s["n"], "")}</div>')
    elif s["n"] in ANIM:
        inner = ANIM[s["n"]]
    else:
        return ""
    return f'<div class="anim"><div class="alab">Animation</div>{inner}</div>'


def render_step_page(s, prev, nxt):
    """A full standalone HTML page for one script."""
    src = esc(read_source(s["file"]))
    out = esc(s["out"])
    body = "\n".join(f"<p>{p}</p>" for p in s["body"])

    topnav = '<a href="index.html#ladder">&larr; All steps</a>'
    if prev:
        topnav += f'<a href="{slug(prev)}">&larr; {prev["n"]}</a>'
    if nxt:
        topnav += f'<a href="{slug(nxt)}">{nxt["n"]} &rarr;</a>'

    prev_html = (f'<a class="prev" href="{slug(prev)}"><div class="dir">&larr; Previous</div>'
                 f'<div class="pt">{prev["n"]} &middot; {esc(prev["title"])}</div></a>'
                 if prev else "<span></span>")
    next_html = (f'<a class="next" href="{slug(nxt)}"><div class="dir">Next &rarr;</div>'
                 f'<div class="pt">{nxt["n"]} &middot; {esc(nxt["title"])}</div></a>'
                 if nxt else "<span></span>")

    nb_name = s["file"].replace(".py", ".ipynb")
    nb_url = f'{REPO}/blob/main/{nb_name}'

    nb_html = render_notebook(s)
    if not nb_html:   # fallback to raw output + .py source if no notebook yet
        nb_html = (_panel("Output", f"<pre>{out}</pre>", lit=False)
                   + _panel(esc(s['file']),
                            f'<pre><code class="language-python">{src}</code></pre>'))

    head = HEAD.format(
        title=f"{s['n']} &middot; {esc(s['title'])} &mdash; Build an LLM from Scratch",
        desc=esc(s["tag"]))

    return head + f"""<header class="bar"><div class="wrap">
  <a class="brand" href="index.html">from_scratch<b>.llm</b></a>
  <nav>{topnav}</nav>
</div></header>

<section class="stephero"><div class="wrap">
  <div class="crumb">{esc(s['phase'])}</div>
  <div class="sh-head">
    <span class="nn">{s['n']}</span>
    <div><h1>{esc(s['title'])}</h1><div class="tag">{s['tag']}</div></div>
  </div>
  <span class="file">{esc(s['file'])}</span>
  <a class="nb-btn" href="{nb_url}">&#9658; Open as notebook (with output)</a>
</div></section>

<section class="section"><div class="wrap stepwrap">
  <div class="lead">{body}</div>
  {eq_theory_html(s)}
  {render_anim(s)}
  {visual_gallery_html(s)}
  <div class="srch">The notebook &middot; <span>{nb_name}</span> &mdash; code cells &amp; their output. <a href="{nb_url}">Open &amp; run on GitHub &rarr;</a></div>
  {nb_html}
  <div class="pager">{prev_html}{next_html}</div>
</div></section>

<footer><div class="wrap">
  <a href="index.html" style="color:var(--accent)">&larr; Back to all 17 steps</a>
  &middot; built one runnable script at a time.
</div></footer>
{PRISM}</body>
</html>"""


def render_card(s):
    visual = VISUALS.get(s["n"], [{}])[0].get("file", "")
    thumb = (f'<img class="thumb" src="{esc(visual)}" alt="" loading="lazy" decoding="async">'
             if visual else "")
    return (f'<a class="card" href="{slug(s)}"><span class="go">&rarr;</span>'
            f'{thumb}<div class="cn">{s["n"]}</div><h3>{esc(s["title"])}</h3>'
            f'<p>{s["tag"]}</p></a>')


def render_ladder():
    out, last = [], None
    for s in STEPS:
        if s["phase"] != last:
            if last is not None:
                out.append("</div>")
            cls = "phlabel first" if last is None else "phlabel"
            out.append(f'<div class="{cls}">{esc(s["phase"])}</div><div class="grid">')
            last = s["phase"]
        out.append(render_card(s))
    out.append("</div>")
    return "\n".join(out)


def index_page():
    """Landing page: reuse the existing hero/acts/SOTA/run/footer, but swap the
    inline ladder for a grid of links to the per-script pages."""
    p = PAGE
    p = p.replace("<style>__CSS__</style>",
                  '<link rel="stylesheet" href="styles.css">')
    p = p.replace("Neural network &rarr; LLM &middot; in 15 steps",
                  "Neural network &rarr; LLM &middot; in 17 steps")
    p = p.replace(
        "Read straight through, or jump to any rung. Each shows the idea, the real terminal output, and the full source.",
        "Seventeen rungs, each its own page &mdash; the idea, the key formula, the theory, the real terminal output, and the full source. Every page also links to an executed Jupyter notebook you can run yourself.")
    ladder_old = ('  <div class="ladder" style="margin-top:34px">\n'
                  '    <nav class="index">__INDEX__</nav>\n'
                  '    <div class="steps">__STEPS__</div>\n'
                  '  </div>')
    p = p.replace(ladder_old, render_ladder())
    scripts_old = (
        '<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-clike.min.js"></script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>\n'
        '<script>__JS__</script>')
    p = p.replace(scripts_old, '<script src="app.js"></script>')
    return p


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Build an LLM from Scratch &mdash; the learning ladder</title>
<meta name="description" content="Fifteen small, runnable Python scripts that take you from a neural network to a 2026-architecture, instruction- and preference-tuned GPT. No black boxes.">
<link rel="icon" href="data:,">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
<style>__CSS__</style>
</head>
<body>
<header class="bar"><div class="wrap">
  <div class="brand">from_scratch<b>.llm</b></div>
  <nav>
    <a href="#ladder">The ladder</a>
    <a href="#sota">State of the art</a>
    <a href="#run">Run it</a>
  </nav>
</div></header>

<section class="hero">
  <canvas id="net"></canvas>
  <div class="wrap">
    <p class="kicker">Neural network &rarr; LLM &middot; in 15 steps</p>
    <h1 class="title">BUILD AN LLM<span class="l2">FROM <em>SCRATCH</em></span></h1>
    <p class="lede">Seventeen small, <b>runnable</b> Python scripts that climb from a 60-line neural network to a 2026-architecture, instruction- and preference-tuned GPT &mdash; down to a quantized KV cache. <b>No black boxes</b> &mdash; every idea inside a frontier model, built and run by hand.</p>
    <div class="cta">
      <a class="btn solid" href="#ladder">Start climbing &darr;</a>
      <a class="btn" href="#sota">What's latest (2026)</a>
    </div>
    <div class="stats">
      <div class="stat"><div class="num">17</div><div class="lab">runnable scripts</div></div>
      <div class="stat"><div class="num">0</div><div class="lab">magic / black boxes</div></div>
      <div class="stat"><div class="num">465K</div><div class="lab">params in the GPT</div></div>
      <div class="stat"><div class="num">2026</div><div class="lab">architecture &amp; alignment</div></div>
    </div>
  </div>
</section>

<section class="section"><div class="wrap">
  <p class="eyebrow">The arc</p>
  <h2 class="sec">Three acts, one machine.</h2>
  <p class="sec-lede">Each script teaches exactly one new idea and builds directly on the last. Nothing is imported that you haven't already built yourself.</p>
  <div class="acts">
    <div class="act"><h3>Act I &middot; Language</h3><p>A neural net learns "what comes next." Counting, then gradient descent, then turning text into tokens.</p></div>
    <div class="act"><h3>Act II &middot; Toolkit</h3><p>Embeddings and context windows, how autograd works, and the switch to PyTorch so gradients compute themselves.</p></div>
    <div class="act"><h3>Act III &middot; Transformer</h3><p>Attention, the transformer block, a real GPT on Shakespeare &mdash; then sampling, training, and alignment up to 2026 spec.</p></div>
  </div>
</div></section>

<section class="section" id="ladder"><div class="wrap">
  <p class="eyebrow">The climb</p>
  <h2 class="sec">The ladder.</h2>
  <p class="sec-lede">Read straight through, or jump to any rung. Each shows the idea, the real terminal output, and the full source.</p>
  <div class="ladder" style="margin-top:34px">
    <nav class="index">__INDEX__</nav>
    <div class="steps">__STEPS__</div>
  </div>
</div></section>

<section class="section" id="sota"><div class="wrap">
  <p class="eyebrow">Checked May 2026</p>
  <h2 class="sec">What's latest &mdash; and what we use.</h2>
  <p class="sec-lede">The transformer hasn't been replaced; it's been refined. Two fronts have moved since 2017 &mdash; the architecture, and how models are aligned after pre-training. Steps #13 and #14 bring this ladder up to current spec.</p>

  <div class="sota-grid">
    <div class="cmp">
      <div class="h">Architecture &middot; #09 (2017) &rarr; #14 (2026)</div>
      <table>
        <tr><th>Component</th><th>Classic</th><th>Modern</th></tr>
        <tr><td class="k">Position</td><td class="old">learned absolute</td><td class="new">RoPE</td></tr>
        <tr><td class="k">Norm</td><td class="old">LayerNorm</td><td class="new">RMSNorm</td></tr>
        <tr><td class="k">Feed-forward</td><td class="old">ReLU MLP</td><td class="new">SwiGLU</td></tr>
        <tr><td class="k">Attention</td><td class="old">multi-head</td><td class="new">GQA</td></tr>
        <tr><td class="k">Result</td><td class="old">val 1.674 / 465K</td><td class="new">val 1.634 / 418K</td></tr>
      </table>
    </div>
    <div class="cmp">
      <div class="h">Post-training stack &middot; 2026</div>
      <div style="padding:18px 16px">
        <p style="margin:0 0 14px;color:#c2c7d0;font-size:16px">Every major model this past year &mdash; DeepSeek-R1, Nemotron, GPT-5-class &mdash; uses a modular stack rather than plain RLHF:</p>
        <div class="stack">
          <span class="n">SFT</span><span class="ar">&rarr;</span>
          <span class="n">DPO</span><span class="ar">&rarr;</span>
          <span class="n">GRPO / RLVR</span>
        </div>
        <p style="margin:16px 0 0;color:var(--muted);font-size:15px">
          <span class="pill">SFT &middot; #12</span> instruction following<br>
          <span class="pill">DPO &middot; #13</span> preference alignment, no reward model<br>
          <span class="pill">GRPO</span> RL with verifiable rewards, for reasoning
        </p>
      </div>
    </div>
  </div>

  <p class="sec-lede" style="margin-top:28px;font-size:17px;color:var(--muted)">
    The other front is <strong>inference efficiency</strong> (steps #15&ndash;#16): the KV cache, then quantizing it to ~3 bits with Google's TurboQuant. Beyond that, all variations on what you've built &mdash; Multi-Latent Attention (DeepSeek) compresses the cache further; Mixture-of-Experts routes each token to a few specialist MLPs; QK-Norm adds stability. The atom is unchanged &mdash; it's attention, all the way down.</p>

  <div class="sources">
    <strong style="color:var(--ink)">Sources</strong> &middot; checked May 2026<br>
    <a href="https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison">Raschka &mdash; The Big LLM Architecture Comparison</a><br>
    <a href="https://llm-stats.com/blog/research/post-training-techniques-2026">Post-Training in 2026: GRPO, DAPO, RLVR &amp; Beyond</a><br>
    <a href="https://www.sundeepteki.org/advice/the-complete-guide-to-post-training-llms-how-sft-rlhf-dpo-and-grpo-shape-llms">The Complete Guide to Post-Training LLMs (SFT/RLHF/DPO/GRPO)</a><br>
    <a href="https://arxiv.org/abs/2305.18290">Rafailov et al. &mdash; Direct Preference Optimization (DPO)</a><br>
    <a href="https://arxiv.org/abs/2504.19874">TurboQuant &mdash; 3-bit KV-cache quantization (Google Research, ICLR 2026)</a>
  </div>
</div></section>

<section class="section run" id="run"><div class="wrap">
  <p class="eyebrow">Get going</p>
  <h2 class="sec">Run it yourself.</h2>
  <p class="sec-lede">Pure Python. One-time setup, then run any script. The GPT trains in ~2 minutes on a laptop CPU and caches its weights so every later step loads instantly.</p>
  <pre style="margin-top:26px">
<span class="c"># one-time setup</span>
python3 -m venv .venv
.venv/bin/pip install numpy torch

<span class="c"># climb the ladder</span>
<span class="p">.venv/bin/python</span> 00_neural_network.py  <span class="c"># the atom</span>
<span class="p">.venv/bin/python</span> 09_tiny_gpt.py        <span class="c"># a real GPT on Shakespeare</span>
<span class="p">.venv/bin/python</span> 14_modern_gpt.py      <span class="c"># the 2026 architecture</span></pre>
</div></section>

<footer><div class="wrap">
  Built step by step, one runnable script at a time. <b>The architecture is GPT's; the scale is a laptop's.</b><br>
  17 scripts &middot; neural&nbsp;network &rarr; tiny&nbsp;GPT &rarr; DPO-aligned, 2026-spec model with a quantized KV cache &middot; no black boxes.
</div></footer>

<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-clike.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
<script>__JS__</script>
</body>
</html>"""


def main():
    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    (DOCS / "styles.css").write_text(CSS, encoding="utf-8")
    (DOCS / "app.js").write_text(JS, encoding="utf-8")
    (DOCS / "index.html").write_text(index_page(), encoding="utf-8")

    for i, s in enumerate(STEPS):
        prev = STEPS[i - 1] if i > 0 else None
        nxt = STEPS[i + 1] if i < len(STEPS) - 1 else None
        (DOCS / slug(s)).write_text(render_step_page(s, prev, nxt), encoding="utf-8")

    print(f"wrote docs/index.html + styles.css + app.js")
    print(f"wrote {len(STEPS)} per-script pages: "
          f"{slug(STEPS[0])} ... {slug(STEPS[-1])}")


if __name__ == "__main__":
    main()
