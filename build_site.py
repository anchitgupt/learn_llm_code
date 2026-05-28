"""
Static-site generator for the "Build an LLM from Scratch" learning ladder.

Reads the real source of every numbered script and emits a single, self-contained
docs/index.html (inline CSS + JS) that is ready to deploy on GitHub Pages.

Run with:  python build_site.py
"""

import html
import os
import pathlib

ROOT = pathlib.Path(__file__).parent
DOCS = ROOT / "docs"


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
]


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

// scroll reveal + scrollspy
(function(){
  const steps=[...document.querySelectorAll('.step')];
  const io=new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting)e.target.classList.add('in');});},{threshold:0.12});
  steps.forEach(s=>io.observe(s));
  const links=new Map([...document.querySelectorAll('.index a')].map(a=>[a.getAttribute('href').slice(1),a]));
  const spy=new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting){
    links.forEach(l=>l.classList.remove('on'));
    const a=links.get(e.target.id); if(a)a.classList.add('on');
  }});},{rootMargin:'-45% 0px -50% 0px'});
  steps.forEach(s=>spy.observe(s));
})();
"""


def render_step(s):
    src = esc(read_source(s["file"]))
    out = esc(s["out"])
    body = "\n".join(f"<p>{p}</p>" for p in s["body"])
    return f"""
<article class="step" id="s{s['n']}">
  <div class="head">
    <span class="nn">{s['n']}</span>
    <div>
      <h3>{esc(s['title'])}</h3>
      <div class="tag">{s['tag']}</div>
    </div>
    <span class="file">{esc(s['file'])}</span>
  </div>
  <div class="body">
    {body}
    <div class="term">
      <div class="top"><span class="dot g"></span><span class="dot"></span><span class="dot"></span>
        <span>{esc(s['file'])}</span><span class="out-lab">output</span></div>
      <pre>{out}</pre>
    </div>
    <details class="src">
      <summary><span class="chev">&#9656;</span> View full source &middot; {esc(s['file'])}</summary>
      <pre><code class="language-python">{src}</code></pre>
    </details>
  </div>
</article>"""


def render_index():
    out, last = [], None
    for s in STEPS:
        if s["phase"] != last:
            out.append(f'<div class="ph">{esc(s["phase"])}</div>')
            last = s["phase"]
        out.append(f'<a href="#s{s["n"]}"><span class="nn">{s["n"]}</span> {esc(s["title"])}</a>')
    return "\n".join(out)


def render_steps():
    out, last = [], None
    for s in STEPS:
        if s["phase"] != last:
            out.append(f'<div class="phase-div">{esc(s["phase"])}</div>')
            last = s["phase"]
        out.append(render_step(s))
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Build an LLM from Scratch &mdash; the learning ladder</title>
<meta name="description" content="Fifteen small, runnable Python scripts that take you from a neural network to a 2026-architecture, instruction- and preference-tuned GPT. No black boxes.">
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
    <p class="lede">Fifteen small, <b>runnable</b> Python scripts that climb from a 60-line neural network to a 2026-architecture, instruction- and preference-tuned GPT. <b>No black boxes</b> &mdash; every idea inside a frontier model, built and run by hand.</p>
    <div class="cta">
      <a class="btn solid" href="#ladder">Start climbing &darr;</a>
      <a class="btn" href="#sota">What's latest (2026)</a>
    </div>
    <div class="stats">
      <div class="stat"><div class="num">15</div><div class="lab">runnable scripts</div></div>
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
    Other current ideas worth knowing, all variations on what you've built: Multi-Latent Attention (DeepSeek) compresses the KV cache further; Mixture-of-Experts routes each token to a few specialist MLPs; QK-Norm adds stability. The atom is unchanged &mdash; it's attention, all the way down.</p>

  <div class="sources">
    <strong style="color:var(--ink)">Sources</strong> &middot; checked May 2026<br>
    <a href="https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison">Raschka &mdash; The Big LLM Architecture Comparison</a><br>
    <a href="https://llm-stats.com/blog/research/post-training-techniques-2026">Post-Training in 2026: GRPO, DAPO, RLVR &amp; Beyond</a><br>
    <a href="https://www.sundeepteki.org/advice/the-complete-guide-to-post-training-llms-how-sft-rlhf-dpo-and-grpo-shape-llms">The Complete Guide to Post-Training LLMs (SFT/RLHF/DPO/GRPO)</a><br>
    <a href="https://arxiv.org/abs/2305.18290">Rafailov et al. &mdash; Direct Preference Optimization (DPO)</a>
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
  15 scripts &middot; neural&nbsp;network &rarr; tiny&nbsp;GPT &rarr; DPO-aligned, 2026-spec model &middot; no black boxes.
</div></footer>

<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-clike.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
<script>__JS__</script>
</body>
</html>"""


def main():
    DOCS.mkdir(exist_ok=True)
    page = (PAGE
            .replace("__CSS__", CSS)
            .replace("__INDEX__", render_index())
            .replace("__STEPS__", render_steps())
            .replace("__JS__", JS))
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    kb = len(page.encode("utf-8")) / 1024
    print(f"wrote docs/index.html ({kb:.0f} KB) + docs/.nojekyll")
    print(f"steps embedded: {len(STEPS)}")


if __name__ == "__main__":
    main()
