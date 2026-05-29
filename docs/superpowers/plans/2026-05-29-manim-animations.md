# Manim Animations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one short, site-embedded Manim concept animation for each of the 17 LLM ladder scripts.

**Architecture:** Keep animation metadata independent from Manim so tests and the site can run without the package installed. Put reusable Manim visual helpers in one module, define one scene class per lesson, render through a stable CLI, and let `build_site.py` embed videos only when the MP4 exists.

**Tech Stack:** Python standard library, Manim Community, existing static `build_site.py`, `unittest`, native HTML5 video.

---

## File Structure

- Create `animations/llm_anim/__init__.py`: package marker and public exports.
- Create `animations/llm_anim/registry.py`: lesson animation metadata, captions, output filenames, and scene class names. This file must not import Manim.
- Create `animations/llm_anim/common.py`: Manim-only visual helpers and shared palette.
- Create `animations/llm_anim/scenes.py`: 17 Manim `Scene` classes using helpers from `common.py`.
- Create `render_animations.py`: CLI for rendering `--scene NN` or `--all`.
- Create `tests/test_animation_registry.py`: metadata and CLI command tests using `unittest`.
- Modify `build_site.py`: add animation lookup and lesson-page video rendering.
- Modify `README.md`: document Manim setup and render commands.
- Modify `.gitignore`: ignore `.superpowers/` and Manim render scratch folders while keeping final `docs/assets/animations/*.mp4` trackable.

## Task 1: Registry and Metadata Tests

**Files:**
- Create: `animations/llm_anim/__init__.py`
- Create: `animations/llm_anim/registry.py`
- Create: `tests/test_animation_registry.py`

- [ ] **Step 1: Write the failing registry tests**

Create `tests/test_animation_registry.py`:

```python
import unittest

from animations.llm_anim.registry import ANIMATION_SPECS, get_animation_spec


class AnimationRegistryTests(unittest.TestCase):
    def test_registry_has_all_17_lessons(self):
        self.assertEqual(len(ANIMATION_SPECS), 17)
        self.assertEqual([spec.number for spec in ANIMATION_SPECS], [f"{i:02d}" for i in range(17)])

    def test_specs_have_stable_output_paths_and_scene_names(self):
        for spec in ANIMATION_SPECS:
            with self.subTest(spec=spec.number):
                self.assertTrue(spec.video_path.startswith("assets/animations/"))
                self.assertTrue(spec.video_path.endswith(".mp4"))
                self.assertTrue(spec.scene_class.startswith("Scene"))
                self.assertIn(spec.number, spec.source_file)
                self.assertTrue(spec.caption)

    def test_lookup_by_number(self):
        spec = get_animation_spec("07")
        self.assertEqual(spec.scene_class, "Scene07SelfAttention")
        self.assertEqual(spec.source_file, "07_self_attention.py")

    def test_lookup_rejects_unknown_number(self):
        with self.assertRaises(KeyError):
            get_animation_spec("99")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m unittest tests/test_animation_registry.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'animations'`.

- [ ] **Step 3: Add the registry implementation**

Create `animations/llm_anim/__init__.py`:

```python
"""Manim animation metadata and scenes for the LLM ladder."""

from .registry import ANIMATION_SPECS, AnimationSpec, get_animation_spec

__all__ = ["ANIMATION_SPECS", "AnimationSpec", "get_animation_spec"]
```

Create `animations/llm_anim/registry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimationSpec:
    number: str
    source_file: str
    scene_class: str
    video_path: str
    caption: str


ANIMATION_SPECS = [
    AnimationSpec("00", "00_neural_network.py", "Scene00NeuralNetwork", "assets/animations/00_neural_network.mp4", "Watch XOR errors drive hidden-layer weight updates."),
    AnimationSpec("01", "01_bigram_counts.py", "Scene01BigramCounts", "assets/animations/01_bigram_counts.mp4", "Watch character pairs accumulate into a next-token count table."),
    AnimationSpec("02", "02_bigram_nn.py", "Scene02BigramNN", "assets/animations/02_bigram_nn.mp4", "Watch logits become probabilities, loss, and gradient updates."),
    AnimationSpec("03", "03_tokenizer.py", "Scene03Tokenizer", "assets/animations/03_tokenizer.mp4", "Watch frequent character pairs merge into compact BPE tokens."),
    AnimationSpec("04", "04_mlp_lm.py", "Scene04MLPLM", "assets/animations/04_mlp_lm.mp4", "Watch a context window become embeddings and a next-character prediction."),
    AnimationSpec("05", "05_autograd.py", "Scene05Autograd", "assets/animations/05_autograd.mp4", "Watch a forward graph reverse into chained gradients."),
    AnimationSpec("06", "06_switch_to_pytorch.py", "Scene06SwitchToPytorch", "assets/animations/06_switch_to_pytorch.mp4", "Watch manual gradient code collapse into PyTorch autograd steps."),
    AnimationSpec("07", "07_self_attention.py", "Scene07SelfAttention", "assets/animations/07_self_attention.mp4", "Watch queries, keys, values, and the causal mask produce attention."),
    AnimationSpec("08", "08_transformer_block.py", "Scene08TransformerBlock", "assets/animations/08_transformer_block.mp4", "Watch heads, residuals, norm, and MLP preserve a stackable shape."),
    AnimationSpec("09", "09_tiny_gpt.py", "Scene09TinyGPT", "assets/animations/09_tiny_gpt.mp4", "Watch token and position embeddings flow through GPT blocks to logits."),
    AnimationSpec("10", "10_sampling.py", "Scene10Sampling", "assets/animations/10_sampling.mp4", "Watch temperature, top-k, and top-p reshape the next-token distribution."),
    AnimationSpec("11", "11_train_loop.py", "Scene11TrainLoop", "assets/animations/11_train_loop.mp4", "Watch a production loop schedule learning rate and resume from checkpoints."),
    AnimationSpec("12", "12_finetune.py", "Scene12Finetune", "assets/animations/12_finetune.mp4", "Watch instruction tuning mask loss onto answer tokens only."),
    AnimationSpec("13", "13_dpo.py", "Scene13DPO", "assets/animations/13_dpo.mp4", "Watch DPO widen the chosen-over-rejected response margin."),
    AnimationSpec("14", "14_modern_gpt.py", "Scene14ModernGPT", "assets/animations/14_modern_gpt.mp4", "Watch RoPE, RMSNorm, SwiGLU, and GQA modernize the GPT block."),
    AnimationSpec("15", "15_kv_cache.py", "Scene15KVCache", "assets/animations/15_kv_cache.mp4", "Watch cached keys and values replace full-context recomputation."),
    AnimationSpec("16", "16_kv_quant.py", "Scene16KVQuant", "assets/animations/16_kv_quant.mp4", "Watch rotation spread outliers before 3-bit KV-cache quantization."),
]


_SPECS_BY_NUMBER = {spec.number: spec for spec in ANIMATION_SPECS}


def get_animation_spec(number: str) -> AnimationSpec:
    return _SPECS_BY_NUMBER[number]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m unittest tests/test_animation_registry.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add animations/llm_anim/__init__.py animations/llm_anim/registry.py tests/test_animation_registry.py
git commit -m "Add animation metadata registry"
```

## Task 2: Render CLI

**Files:**
- Create: `render_animations.py`
- Modify: `tests/test_animation_registry.py`

- [ ] **Step 1: Add failing CLI command tests**

Append to `tests/test_animation_registry.py`:

```python
from pathlib import Path

from render_animations import build_manim_command, select_specs


class RenderCliTests(unittest.TestCase):
    def test_select_specs_for_one_scene(self):
        selected = select_specs(scene="07", render_all=False)
        self.assertEqual([spec.number for spec in selected], ["07"])

    def test_select_specs_for_all_scenes(self):
        selected = select_specs(scene=None, render_all=True)
        self.assertEqual(len(selected), 17)

    def test_build_manim_command_uses_scene_and_output_file(self):
        spec = get_animation_spec("07")
        command = build_manim_command(spec, quality="low", python_executable="python")
        self.assertEqual(command[0], "python")
        self.assertIn("-m", command)
        self.assertIn("manim", command)
        self.assertIn("Scene07SelfAttention", command)
        self.assertIn(str(Path("docs") / spec.video_path), command)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m unittest tests/test_animation_registry.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'render_animations'`.

- [ ] **Step 3: Add the render CLI**

Create `render_animations.py`:

```python
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from animations.llm_anim.registry import ANIMATION_SPECS, AnimationSpec, get_animation_spec


ROOT = Path(__file__).resolve().parent
SCENES_FILE = ROOT / "animations" / "llm_anim" / "scenes.py"
OUTPUT_ROOT = ROOT / "docs"
QUALITY_FLAGS = {
    "low": "-ql",
    "medium": "-qm",
    "high": "-qh",
}


def manim_available() -> bool:
    return importlib.util.find_spec("manim") is not None


def select_specs(scene: str | None, render_all: bool) -> list[AnimationSpec]:
    if render_all:
        return list(ANIMATION_SPECS)
    if scene is None:
        raise ValueError("Choose --all or --scene NN.")
    try:
        return [get_animation_spec(scene)]
    except KeyError as exc:
        valid = ", ".join(spec.number for spec in ANIMATION_SPECS)
        raise ValueError(f"Unknown scene {scene!r}. Valid scenes: {valid}") from exc


def build_manim_command(
    spec: AnimationSpec,
    quality: str,
    python_executable: str = sys.executable,
) -> list[str]:
    output_path = OUTPUT_ROOT / spec.video_path
    return [
        python_executable,
        "-m",
        "manim",
        QUALITY_FLAGS[quality],
        "--format",
        "mp4",
        "-o",
        str(output_path),
        str(SCENES_FILE),
        spec.scene_class,
    ]


def render_spec(spec: AnimationSpec, quality: str) -> None:
    output_path = OUTPUT_ROOT / spec.video_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_manim_command(spec, quality)
    subprocess.run(command, cwd=ROOT, check=True)
    if not output_path.exists():
        raise RuntimeError(f"Manim finished but did not create {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Manim animations for the LLM ladder.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Render every animation scene.")
    group.add_argument("--scene", help="Render one scene by lesson number, for example 07.")
    parser.add_argument("--quality", choices=sorted(QUALITY_FLAGS), default="low")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = select_specs(args.scene, args.all)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not manim_available():
        print("Manim is not installed. Run: .venv/bin/pip install manim", file=sys.stderr)
        return 1
    for spec in specs:
        print(f"Rendering {spec.number}: {spec.scene_class}")
        render_spec(spec, args.quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m unittest tests/test_animation_registry.py -v
```

Expected: registry and CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add render_animations.py tests/test_animation_registry.py
git commit -m "Add Manim render CLI"
```

## Task 3: Shared Manim Helpers and Scene Classes

**Files:**
- Create: `animations/llm_anim/common.py`
- Create: `animations/llm_anim/scenes.py`

- [ ] **Step 1: Add common Manim helpers**

Create `animations/llm_anim/common.py`:

```python
from __future__ import annotations

from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    RIGHT,
    UP,
    VGroup,
    WHITE,
    YELLOW,
    Arrow,
    BarChart,
    Create,
    FadeIn,
    FadeOut,
    MathTex,
    Rectangle,
    Scene,
    Square,
    Text,
    Transform,
    Write,
)


BG = "#13161f"
PANEL = "#181c27"
INK = "#ece8de"
MUTED = "#9aa1ad"
ACCENT = "#5ef0c0"
AMBER = "#f2b65e"
ROSE = "#f08a8a"
BLUE_SOFT = "#8fd0ff"


def title(text: str) -> Text:
    return Text(text, font_size=34, color=ACCENT).to_edge(UP)


def label(text: str, size: int = 24, color: str = INK) -> Text:
    return Text(text, font_size=size, color=color)


def token_row(tokens: list[str], color: str = BLUE_SOFT) -> VGroup:
    boxes = VGroup()
    for token in tokens:
        box = Rectangle(width=max(0.52, len(token) * 0.22), height=0.48, color=color)
        txt = Text(token, font_size=20, color=INK)
        txt.move_to(box.get_center())
        boxes.add(VGroup(box, txt))
    boxes.arrange(RIGHT, buff=0.12)
    return boxes


def phase_card(heading: str, lines: list[str]) -> VGroup:
    heading_obj = Text(heading, font_size=26, color=AMBER)
    line_objs = VGroup(*[Text(line, font_size=20, color=INK) for line in lines])
    line_objs.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    group = VGroup(heading_obj, line_objs).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    frame = Rectangle(width=5.8, height=max(1.55, group.height + 0.55), color=MUTED)
    frame.set_fill(PANEL, opacity=0.7)
    group.move_to(frame.get_center())
    return VGroup(frame, group)


def mini_matrix(rows: list[list[str]], active: tuple[int, int] | None = None) -> VGroup:
    cells = VGroup()
    for r, row in enumerate(rows):
        row_cells = VGroup()
        for c, value in enumerate(row):
            cell = Square(side_length=0.56, color=ACCENT if active == (r, c) else MUTED)
            cell.set_fill(ACCENT if active == (r, c) else PANEL, opacity=0.25)
            txt = Text(value, font_size=18, color=INK)
            txt.move_to(cell.get_center())
            row_cells.add(VGroup(cell, txt))
        row_cells.arrange(RIGHT, buff=0.06)
        cells.add(row_cells)
    cells.arrange(DOWN, buff=0.06)
    return cells


def probability_bars(labels: list[str], values: list[float]) -> BarChart:
    chart = BarChart(
        values=values,
        bar_names=labels,
        y_range=[0, 1, 0.5],
        y_length=2.4,
        x_length=4.2,
        bar_colors=[GREEN, BLUE, ORANGE, YELLOW],
    )
    chart.scale(0.8)
    return chart


class ConceptScene(Scene):
    scene_title = ""
    phases: list[tuple[str, list[str]]] = []

    def construct(self) -> None:
        self.camera.background_color = BG
        heading = title(self.scene_title)
        self.play(Write(heading))

        current = None
        for idx, (phase, lines) in enumerate(self.phases):
            card = phase_card(phase, lines).move_to(0.2 * DOWN)
            if current is None:
                self.play(FadeIn(card, shift=0.25 * UP))
            else:
                self.play(Transform(current, card))
                card = current
            current = card
            self.wait(0.45)

            cue = self.visual_cue(idx)
            if cue is not None:
                cue.next_to(card, DOWN, buff=0.45)
                self.play(FadeIn(cue))
                self.wait(0.35)
                self.play(FadeOut(cue))

        if current is not None:
            self.play(FadeOut(current), FadeOut(heading))

    def visual_cue(self, idx: int):
        if idx % 3 == 0:
            return token_row(["x", "W", "loss", "grad"], color=ACCENT)
        if idx % 3 == 1:
            return mini_matrix([["q", "k", "v"], ["0", "1", "0"], ["p", "p", "p"]], active=(1, 1))
        return probability_bars(["a", "b", "c"], [0.55, 0.3, 0.15])
```

- [ ] **Step 2: Add the 17 scene classes**

Create `animations/llm_anim/scenes.py`:

```python
from __future__ import annotations

from .common import ConceptScene


class Scene00NeuralNetwork(ConceptScene):
    scene_title = "00. Neural Network"
    phases = [
        ("XOR data", ["Two inputs are not linearly separable.", "The target is 1 only when bits differ."]),
        ("Hidden features", ["A hidden layer bends the space.", "Sigmoid units create learned features."]),
        ("Backprop", ["Loss measures prediction error.", "Gradients nudge every weight downhill."]),
    ]


class Scene01BigramCounts(ConceptScene):
    scene_title = "01. Bigram Counts"
    phases = [
        ("Pair stream", ["Names become adjacent character pairs.", "Each pair means current token -> next token."]),
        ("Count table", ["Counts accumulate into a matrix.", "Rows become next-token distributions."]),
        ("Sampling", ["Start at '.', sample a next character.", "Repeat until '.' ends the name."]),
    ]


class Scene02BigramNN(ConceptScene):
    scene_title = "02. Bigram Neural Net"
    phases = [
        ("One-hot input", ["A character id becomes a one-hot vector.", "The weight row acts like learned counts."]),
        ("Softmax", ["Logits turn into probabilities.", "Cross-entropy rewards the true next token."]),
        ("Descent", ["Gradient descent moves probabilities toward data.", "The neural model learns the counting answer."]),
    ]


class Scene03Tokenizer(ConceptScene):
    scene_title = "03. Tokenizer"
    phases = [
        ("Characters", ["Text starts as individual symbols.", "Every token maps to an integer id."]),
        ("BPE merge", ["The most frequent adjacent pair becomes one token.", "Repeated merges compress common chunks."]),
        ("Context value", ["Fewer tokens fit more text into the same context.", "The model still sees reversible numbers."]),
    ]


class Scene04MLPLM(ConceptScene):
    scene_title = "04. MLP Language Model"
    phases = [
        ("Context window", ["Several previous characters form the input.", "The model now has short memory."]),
        ("Embeddings", ["Each id looks up a learned vector.", "Vectors are flattened into one feature stack."]),
        ("Prediction", ["The MLP predicts the next character.", "Generated names become more realistic."]),
    ]


class Scene05Autograd(ConceptScene):
    scene_title = "05. Autograd"
    phases = [
        ("Forward graph", ["Every Value stores data and parents.", "Operations build a computation graph."]),
        ("Local rules", ["Each operation knows one derivative rule.", "The chain rule composes local facts."]),
        ("Backward pass", ["Topological order reverses the graph.", "Gradients accumulate back to weights."]),
    ]


class Scene06SwitchToPytorch(ConceptScene):
    scene_title = "06. Switch to PyTorch"
    phases = [
        ("Tensors", ["Arrays become tensors with gradients.", "Modules own parameters."]),
        ("Loss backward", ["Manual derivatives disappear.", "loss.backward() fills parameter gradients."]),
        ("Optimizer", ["optimizer.step() applies updates.", "The loop now scales to larger models."]),
    ]


class Scene07SelfAttention(ConceptScene):
    scene_title = "07. Self-Attention"
    phases = [
        ("Q K V", ["Each token emits a query, key, and value.", "Query dot key scores relevance."]),
        ("Causal mask", ["Future positions are blocked.", "A token may attend only leftward."]),
        ("Weighted sum", ["Softmax scores weight the values.", "Each token becomes context-aware."]),
    ]


class Scene08TransformerBlock(ConceptScene):
    scene_title = "08. Transformer Block"
    phases = [
        ("Multi-head", ["Several attention heads run in parallel.", "Each head can learn a different relation."]),
        ("Residual and norm", ["Residual paths preserve signal.", "LayerNorm stabilizes the stack."]),
        ("MLP", ["A feed-forward layer lets each token think.", "Shape in equals shape out, so blocks stack."]),
    ]


class Scene09TinyGPT(ConceptScene):
    scene_title = "09. Tiny GPT"
    phases = [
        ("Embeddings", ["Token ids and positions become vectors.", "The model knows content and order."]),
        ("Block stack", ["Transformer blocks refine every token.", "Context flows through attention."]),
        ("Logits", ["The head scores every next character.", "Training pushes the real next token up."]),
    ]


class Scene10Sampling(ConceptScene):
    scene_title = "10. Sampling"
    phases = [
        ("Distribution", ["The model outputs probabilities.", "Sampling chooses how to use them."]),
        ("Temperature", ["Low temperature sharpens choices.", "High temperature spreads probability mass."]),
        ("Top-k and top-p", ["Filters remove unlikely tails.", "The same model can write in many styles."]),
    ]


class Scene11TrainLoop(ConceptScene):
    scene_title = "11. Production Training"
    phases = [
        ("Loop", ["Train batches update weights.", "Eval batches measure generalization."]),
        ("Schedule", ["Warmup raises learning rate safely.", "Cosine decay cools training down."]),
        ("Checkpoint", ["Model and optimizer state are saved.", "A stopped run resumes exactly."]),
    ]


class Scene12Finetune(ConceptScene):
    scene_title = "12. Instruction Tuning"
    phases = [
        ("Prompt format", ["Instruction and response become one sequence.", "The base model sees assistant examples."]),
        ("Loss mask", ["Question tokens are ignored.", "Only answer tokens train the model."]),
        ("Assistant behavior", ["Weights shift from continuation to following commands.", "The tuned model answers directly."]),
    ]


class Scene13DPO(ConceptScene):
    scene_title = "13. DPO"
    phases = [
        ("Preference pair", ["Each prompt has chosen and rejected answers.", "The model learns relative preference."]),
        ("Reference model", ["A frozen copy anchors the update.", "The policy should improve without drifting wildly."]),
        ("Margin", ["DPO widens chosen minus rejected log-prob.", "Alignment happens without a reward model."]),
    ]


class Scene14ModernGPT(ConceptScene):
    scene_title = "14. Modern GPT"
    phases = [
        ("RoPE", ["Queries and keys rotate by position.", "Relative order is encoded in attention."]),
        ("RMSNorm and SwiGLU", ["Normalization is cheaper.", "Gated feed-forward layers add capacity."]),
        ("GQA", ["Many query heads share fewer key/value heads.", "Generation keeps quality with a smaller cache."]),
    ]


class Scene15KVCache(ConceptScene):
    scene_title = "15. KV Cache"
    phases = [
        ("Naive generation", ["Every new token recomputes the whole context.", "Work grows roughly quadratically."]),
        ("Cache append", ["Past keys and values are stored once.", "Each step processes only the new token."]),
        ("Same output", ["Cached and naive tokens match.", "The speedup changes work, not behavior."]),
    ]


class Scene16KVQuant(ConceptScene):
    scene_title = "16. KV Quantization"
    phases = [
        ("Outliers", ["A few channels dominate quantization range.", "Few-bit buckets waste precision."]),
        ("Rotation", ["An orthogonal rotation spreads magnitude.", "Dot products are preserved for attention."]),
        ("3-bit cache", ["Values fit into compact buckets.", "Long-context memory becomes deployable."]),
    ]
```

- [ ] **Step 3: Render one scene**

Run:

```bash
.venv/bin/python render_animations.py --scene 00 --quality low
```

Expected if Manim is installed: creates `docs/assets/animations/00_neural_network.mp4`.

Expected if Manim is missing: exits with `Manim is not installed. Run: .venv/bin/pip install manim`.

- [ ] **Step 4: Commit**

If the scene import and render path work:

```bash
git add animations/llm_anim/common.py animations/llm_anim/scenes.py
git commit -m "Add Manim concept scenes"
```

## Task 4: Site Embedding

**Files:**
- Modify: `build_site.py`
- Modify: `tests/test_animation_registry.py`

- [ ] **Step 1: Add failing site HTML tests**

Append to `tests/test_animation_registry.py`:

```python
import tempfile
from unittest import mock

import build_site


class SiteAnimationTests(unittest.TestCase):
    def test_animation_html_omits_missing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(build_site, "ROOT", Path(tmp)):
                html = build_site.animation_html(get_animation_spec("07"))
        self.assertEqual(html, "")

    def test_animation_html_embeds_existing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "docs" / "assets" / "animations" / "07_self_attention.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"fake mp4")
            with mock.patch.object(build_site, "ROOT", root):
                html = build_site.animation_html(get_animation_spec("07"))
        self.assertIn("<video", html)
        self.assertIn("assets/animations/07_self_attention.mp4", html)
        self.assertIn("Watch queries, keys, values", html)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m unittest tests/test_animation_registry.py -v
```

Expected: fail with `AttributeError: module 'build_site' has no attribute 'animation_html'`.

- [ ] **Step 3: Add animation HTML support to `build_site.py`**

Add this import near the existing imports:

```python
from animations.llm_anim.registry import get_animation_spec
```

Add this helper near `visual_gallery_html`:

```python
def animation_html(spec):
    """Render a lesson animation if its MP4 exists in docs/assets/animations."""
    video_file = ROOT / "docs" / spec.video_path
    if not video_file.exists():
        return ""
    return (
        '<section class="animation" aria-label="Concept animation">'
        '<div class="visual-kicker">Animation</div>'
        f'<figure class="animation-card">'
        f'<video controls preload="metadata" src="{esc(spec.video_path)}"></video>'
        f'<figcaption>{esc(spec.caption)}</figcaption>'
        '</figure>'
        '</section>'
    )
```

In `render_step_page`, after `body = ...`, add:

```python
    animation = animation_html(get_animation_spec(s["n"]))
```

Then insert `{animation}` in the returned lesson HTML after the intro/theory block and before the notebook/source panels.

- [ ] **Step 4: Add CSS for the video section**

In the CSS string in `build_site.py`, add styles matching the existing visuals:

```css
.animation{margin:38px 0}
.animation-card{margin:0;border:1px solid rgba(255,255,255,.12);background:rgba(24,28,39,.78);border-radius:8px;overflow:hidden}
.animation-card video{display:block;width:100%;background:#0d1018;aspect-ratio:16/9}
.animation-card figcaption{padding:14px 16px;color:#9aa1ad;font-size:14px;line-height:1.5}
```

- [ ] **Step 5: Run tests and rebuild the site**

Run:

```bash
.venv/bin/python -m unittest tests/test_animation_registry.py -v
.venv/bin/python build_site.py
```

Expected: tests pass and `docs/*.html` regenerate without errors.

- [ ] **Step 6: Commit**

```bash
git add build_site.py tests/test_animation_registry.py docs/*.html docs/styles.css
git commit -m "Embed lesson animations in the site"
```

## Task 5: Documentation and Ignore Rules

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Update `.gitignore`**

Add:

```gitignore
# Local brainstorming/session artifacts
.superpowers/

# Manim scratch output; final site videos live in docs/assets/animations/
media/
```

- [ ] **Step 2: Update README quick start**

Under Quick start, add:

```markdown
### Optional: render Manim animations

```bash
.venv/bin/pip install manim
.venv/bin/python render_animations.py --scene 07 --quality low
.venv/bin/python render_animations.py --all --quality low
.venv/bin/python build_site.py
```

Rendered MP4s are written to `docs/assets/animations/` and are embedded
automatically on the matching lesson page. If Manim is not installed,
`render_animations.py` exits with an install hint instead of failing silently.
```

- [ ] **Step 3: Commit**

```bash
git add README.md .gitignore
git commit -m "Document Manim animation workflow"
```

## Task 6: Final Validation

**Files:**
- Verify working tree and generated outputs.

- [ ] **Step 1: Run unit tests**

```bash
.venv/bin/python -m unittest tests/test_animation_registry.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run missing-Manim or render smoke**

If Manim is not installed:

```bash
.venv/bin/python render_animations.py --scene 00 --quality low
```

Expected: exits non-zero with `Manim is not installed. Run: .venv/bin/pip install manim`.

If Manim is installed:

```bash
.venv/bin/python render_animations.py --scene 00 --quality low
test -f docs/assets/animations/00_neural_network.mp4
```

Expected: the MP4 exists.

- [ ] **Step 3: Rebuild site**

```bash
.venv/bin/python build_site.py
```

Expected: command exits 0 and regenerates `docs/index.html` plus lesson pages.

- [ ] **Step 4: Browser check**

Serve the site:

```bash
.venv/bin/python -m http.server 8029 --directory docs
```

Open `http://localhost:8029/00_neural_network.html`. If `00_neural_network.mp4`
exists, confirm the Animation section appears and the native video controls are
visible. If no MP4 exists, confirm the page renders without an empty animation
section.

- [ ] **Step 5: Final status**

```bash
git status --short
```

Expected: only intentional generated MP4/HTML changes remain, or the working tree
is clean after committing the implementation.
