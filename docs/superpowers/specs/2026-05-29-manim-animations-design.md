# Manim Animations Design

Date: 2026-05-29
Repo: `learn_llm_code`

## Goal

Add one short Manim concept animation for each of the 17 runnable scripts in the
"Build an LLM from Scratch" ladder. Each animation should make the core
computation in that script visible, then the existing static site should embed
the matching video on that lesson page.

This is a first pass focused on coverage, consistency, and a repeatable render
pipeline. The animations are silent, concise, and designed to complement the
existing notebooks, SVG/WebP visuals, terminal output, and source-code panels.

## Scope

In scope:

- One Manim scene for each script from `00_neural_network.py` through
  `16_kv_quant.py`.
- Shared Manim helpers for recurring visual ideas: title bars, code callouts,
  token boxes, matrices, arrows, probability bars, training curves, and small
  architecture diagrams.
- A render script that can render all scenes or a selected scene.
- Web-ready MP4 outputs under `docs/assets/animations/`.
- Site integration in `build_site.py` that adds an animation section only when
  a lesson's video exists.
- README/run instructions for installing Manim, rendering scenes, and rebuilding
  the site.

Out of scope for this first pass:

- Voiceover or subtitles.
- Long-form 1-2 minute lesson videos.
- Replacing the existing SVG/WebP visual walkthroughs.
- Running the training scripts to generate live animation data.
- Publishing rendered videos if the local environment cannot install or run
  Manim during implementation.

## Architecture

Use approach A: a shared Manim library plus one scene class per script.

Planned files:

- `animations/llm_anim/__init__.py`
- `animations/llm_anim/common.py`
- `animations/llm_anim/scenes.py`
- `render_animations.py`

`common.py` owns the reusable visual language. It should centralize the color
palette, font sizes, layout helpers, and small Manim builders so all 17 scenes
feel like one course rather than unrelated videos.

`scenes.py` defines scene classes named by lesson number, for example
`Scene00NeuralNetwork`, `Scene07SelfAttention`, and `Scene16KVQuant`. Each scene
is independent enough to render by itself, but uses helpers from `common.py`.

`render_animations.py` is the stable entry point. It maps lesson numbers to
scene classes and output filenames, validates input, checks whether Manim is
available, creates output directories, invokes Manim, and fails clearly when a
render fails.

`build_site.py` remains the source of lesson ordering and metadata. It should
render an "Animation" section for a lesson only if the matching MP4 is present
under `docs/assets/animations/`.

## Scene Plan

All scenes target roughly 20-40 seconds.

| # | Script | Animation focus |
|---|--------|-----------------|
| 00 | `00_neural_network.py` | XOR points, hidden layer separation, forward/loss/backward weight nudges |
| 01 | `01_bigram_counts.py` | Character pair stream accumulating into a count matrix, then sampling a name |
| 02 | `02_bigram_nn.py` | One-hot input, logits, softmax, cross-entropy, gradient descent toward counts |
| 03 | `03_tokenizer.py` | Raw characters, repeated pair merges, compressed BPE token sequence |
| 04 | `04_mlp_lm.py` | Context window lookup, embeddings flattening, MLP next-character prediction |
| 05 | `05_autograd.py` | Forward computation graph, local derivative rules, reverse gradient flow |
| 06 | `06_switch_to_pytorch.py` | Manual-gradient blocks collapsing into tensor/autograd/optimizer steps |
| 07 | `07_self_attention.py` | Q/K/V projections, causal mask, attention weights, weighted value sum |
| 08 | `08_transformer_block.py` | Parallel heads, residual add/norm, feed-forward, stackable same-shape output |
| 09 | `09_tiny_gpt.py` | Token and position embeddings through GPT blocks to next-token logits |
| 10 | `10_sampling.py` | Probability distribution reshaped by temperature, top-k, and top-p filters |
| 11 | `11_train_loop.py` | Train/eval loop, LR warmup-decay curve, checkpoint save/resume |
| 12 | `12_finetune.py` | Instruction-response template, loss mask over answer tokens, assistant output |
| 13 | `13_dpo.py` | Chosen vs rejected responses, policy/reference comparison, widening margin |
| 14 | `14_modern_gpt.py` | RoPE rotation, RMSNorm/SwiGLU block, GQA shared key/value heads |
| 15 | `15_kv_cache.py` | Naive recomputation versus append-only KV cache, identical output with less work |
| 16 | `16_kv_quant.py` | Outlier channels, rotation spreading values, 3-bit quantized cache buckets |

## Render Flow

Primary commands:

```bash
.venv/bin/pip install manim
.venv/bin/python render_animations.py --scene 07
.venv/bin/python render_animations.py --all
.venv/bin/python build_site.py
```

The render script should support at least:

- `--scene NN` to render one scene.
- `--all` to render every scene.
- `--quality low|medium|high`, defaulting to a development-friendly low or
  medium quality.

Rendered files should use stable names derived from the lesson number and slug,
for example `docs/assets/animations/07_self_attention.mp4`.

## Site Integration

Each lesson page gets a compact animation section after the text/theory area and
before the rendered notebook/source-heavy content. The section should include:

- A short heading such as "Animation".
- A one-sentence caption specific to the lesson.
- A native `<video controls preload="metadata">` element.

If a video is missing, the section is omitted. This keeps the site useful while
individual scenes are still being rendered.

## Error Handling

The render script must:

- Detect a missing `manim` Python package and print a concrete install command.
- Reject unknown scene numbers before invoking Manim.
- Create the animation output directory if needed.
- Exit non-zero if Manim fails to render a selected scene.
- Avoid pretending success if an expected MP4 was not created.

The site generator must:

- Not fail when animation videos are absent.
- Escape captions and paths through existing HTML helpers.
- Produce deterministic HTML from the same set of available assets.

## Validation

Implementation is complete when these checks pass or their blockers are reported
clearly:

- Python import check for the new animation package and scene registry.
- Render at least one cheap scene locally if Manim and its system dependencies
  are available.
- Run `python build_site.py`.
- Serve/open a generated lesson page and confirm the animation section appears
  when the corresponding MP4 exists.
- Confirm the static site still works when no MP4 exists for a lesson.

If Manim cannot be installed or rendered in the local environment, the fallback
completion standard is: import/registry tests pass, the render script fails with
clear install instructions, `build_site.py` succeeds, and README instructions
are exact enough for the user to run locally after installing Manim.

## Open Decisions Resolved

- Use one shared Manim module with 17 scene classes.
- Embed videos into existing lesson pages.
- Make the first pass short and silent.
- Preserve existing visual walkthroughs and notebooks.
- Do not require all videos to exist before rebuilding the site.
