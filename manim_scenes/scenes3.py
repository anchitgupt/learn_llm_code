"""
Remaining Manim scenes so every step is animated: #01 #02 #04 #06 #11 #12 #16.

Render each (then copy mp4 to docs/assets/anim/<NN>.mp4):
  .venv/bin/manim -qm --media_dir /tmp/manim_media -o 01 manim_scenes/scenes3.py Bigram
  ... Softmax(02) Embeddings(04) PyTorch(06) LRSchedule(11) LossMask(12) TurboQuant(16)
"""

import numpy as np
from manim import *

BG = "#0a0c10"
MINT = "#5ef0c0"
AMBER = "#f2b65e"
ROSE = "#f08a8a"
INK = "#ece8de"
MUTED = "#9aa1ad"
PANEL = "#12151c"
BASE = -2.0


def vbar(x, h, color, w=0.7):
    return Rectangle(width=w, height=max(h, 0.01), fill_color=color, fill_opacity=0.9,
                     stroke_width=0).move_to([x, BASE + h / 2, 0])


def tile(ch, x, y, color="#3a4658", txt=INK):
    sq = Square(side_length=0.72, stroke_color=color, stroke_width=2,
                fill_color=PANEL, fill_opacity=1).move_to([x, y, 0])
    return VGroup(sq, Text(ch, color=txt).scale(0.42).move_to(sq))


class Bigram(Scene):
    """#01 — count next chars after a token, then sample."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("Bigram: count which character follows which, then sample",
                     color=INK).scale(0.5).to_edge(UP, buff=0.6)
        self.play(FadeIn(title))
        prompt = tile("h", -5, 0.4, color=MINT, txt=MINT)
        arrow = Arrow([-4.4, 0.4, 0], [-3.2, 0.4, 0], color="#5a6472", buff=0.1, stroke_width=4)
        self.play(FadeIn(prompt), GrowArrow(arrow))
        chars = ["a", "e", "i", "o", "t"]
        cnts = [1.0, 2.6, 0.6, 0.4, 1.6]
        xs = [-1.8, -0.6, 0.6, 1.8, 3.0]
        bars = VGroup(*[vbar(x, h, AMBER) for x, h in zip(xs, cnts)])
        labs = VGroup(*[Text(c, color=MUTED).scale(0.4).move_to([x, BASE - 0.4, 0])
                        for c, x in zip(chars, xs)])
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.12),
                  FadeIn(labs))
        self.wait(0.3)
        self.play(bars[1].animate.set_fill(MINT), Flash(bars[1].get_top(), color=MINT))
        res = tile("e", 4.6, 0.4, color=MINT, txt=MINT)
        self.play(FadeIn(res, shift=UP * 0.2))
        self.wait(0.6)


class Softmax(Scene):
    """#02 — raw scores become probabilities that sum to 1."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("softmax turns raw scores into probabilities (they sum to 1)",
                     color=INK).scale(0.5).to_edge(UP, buff=0.6)
        self.play(FadeIn(title))
        xs = [-3.2, -1.6, 0, 1.6, 3.2]
        raw = [1.4, 2.9, 0.7, 2.1, 0.5]
        s = sum(raw)
        probs = [r / s * 4.2 for r in raw]
        sb = VGroup(*[vbar(x, h, AMBER) for x, h in zip(xs, raw)])
        lab1 = Text("scores (logits)", color=AMBER).scale(0.42).to_edge(DOWN, buff=0.9)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in sb], lag_ratio=0.1), FadeIn(lab1))
        self.wait(0.4)
        pb = VGroup(*[vbar(x, h, MINT) for x, h in zip(xs, probs)])
        lab2 = Text("probabilities  —  Σ = 1", color=MINT).scale(0.42).to_edge(DOWN, buff=0.9)
        self.play(ReplacementTransform(sb, pb), ReplacementTransform(lab1, lab2), run_time=1.4)
        self.wait(0.7)


class Embeddings(Scene):
    """#04 — tokens become learned vectors; a window predicts the next."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("Embeddings: each token becomes a learned vector",
                     color=INK).scale(0.52).to_edge(UP, buff=0.6)
        self.play(FadeIn(title))
        ctx = ["l", "i", "a"]
        xs = [-4.5, -3.4, -2.3]
        tiles = VGroup(*[tile(c, x, 2.0) for c, x in zip(ctx, xs)])
        self.play(LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=0.15))

        def vec(x, vals):
            cells = VGroup(*[Rectangle(width=0.5, height=0.3, stroke_width=1,
                                       stroke_color="#3a4658",
                                       fill_color=interpolate_color(ManimColor(BG), ManimColor(MINT), float(v)),
                                       fill_opacity=1).move_to([x, 0.6 - i * 0.32, 0])
                             for i, v in enumerate(vals)])
            return cells
        vecs = VGroup(*[vec(x, np.random.RandomState(i).rand(3))
                        for i, x in enumerate(xs)])
        arrows = VGroup(*[Arrow([x, 1.6, 0], [x, 1.0, 0], buff=0.05,
                                color="#5a6472", stroke_width=3) for x in xs])
        self.play(*[GrowArrow(a) for a in arrows], LaggedStart(*[FadeIn(v) for v in vecs], lag_ratio=0.15))

        mlp = RoundedRectangle(width=2.4, height=1.1, corner_radius=0.12, stroke_color=MINT,
                               stroke_width=3, fill_color=PANEL, fill_opacity=1).shift(RIGHT * 1.4)
        mlpt = Text("MLP", color=MINT).scale(0.45).move_to(mlp)
        feed = Arrow([-1.9, 0.3, 0], mlp.get_left(), buff=0.1, color="#5a6472", stroke_width=4)
        out = tile("m", 4.4, 0.3, color=AMBER, txt=AMBER)
        outarr = Arrow(mlp.get_right(), [3.9, 0.3, 0], buff=0.1, color=AMBER, stroke_width=4)
        self.play(GrowArrow(feed), FadeIn(mlp), FadeIn(mlpt))
        self.play(GrowArrow(outarr), FadeIn(out, shift=RIGHT * 0.2))
        cap = Text("a window of previous tokens predicts the next one",
                   color=MUTED).scale(0.38).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(cap))
        self.wait(0.6)


class PyTorch(Scene):
    """#06 — hand-coded gradients collapse into loss.backward()."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("PyTorch: the hand-coded gradients collapse to one call",
                     color=INK).scale(0.52).to_edge(UP, buff=0.7)
        self.play(FadeIn(title))
        lines = ["d_logits = probs - y", "d_W2 = h.T @ d_logits",
                 "d_h = (d_logits @ W2.T) * (1 - h**2)", "d_W1 = X.T @ d_h",
                 "np.add.at(d_C, X, d_emb)", "W -= lr * d_W   # for each"]
        col = VGroup(*[Text(l, color=MUTED, font="monospace").scale(0.34) for l in lines])
        col.arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(LEFT * 3.2)
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in col], lag_ratio=0.12))
        self.wait(0.5)
        call = Text("loss.backward()", color=MINT, font="monospace").scale(0.6).shift(RIGHT * 3.0)
        arrow = Arrow(col.get_right(), call.get_left(), buff=0.4, color=AMBER, stroke_width=4)
        self.play(GrowArrow(arrow))
        self.play(col.animate.set_opacity(0.25), FadeIn(call, scale=1.1))
        self.play(Indicate(call, color=MINT, scale_factor=1.1))
        self.wait(0.6)


class LRSchedule(Scene):
    """#11 — learning-rate warmup then cosine decay."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("Learning-rate schedule: warm up, then cosine-decay",
                     color=INK).scale(0.52).to_edge(UP, buff=0.6)
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 1.1, 0.5], x_length=9.5, y_length=4.2,
                  axis_config={"stroke_color": "#3a4658", "include_ticks": False})
        wu = 0.12
        def lr(t):
            if t < wu:
                return t / wu
            p = (t - wu) / (1 - wu)
            return 0.1 + 0.5 * (1 + np.cos(np.pi * p)) * 0.9
        curve = ax.plot(lr, x_range=[0, 1, 0.005], color=AMBER, stroke_width=5)
        xlab = Text("training step", color=MUTED).scale(0.36).next_to(ax.x_axis, DOWN, buff=0.15)
        ylab = Text("learning rate", color=MUTED).scale(0.36).next_to(ax.y_axis, UP, buff=0.1)
        self.play(FadeIn(title), Create(ax), FadeIn(xlab), FadeIn(ylab))
        self.play(Create(curve), run_time=1.6)
        t = ValueTracker(0.0)
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), lr(t.get_value())),
                                        color=MINT, radius=0.1))
        wlab = Text("warmup", color=MINT).scale(0.34).move_to(ax.c2p(wu, 1.05) + UP * 0.25)
        dlab = Text("cosine decay", color=MINT).scale(0.34).move_to(ax.c2p(0.6, lr(0.6)) + UP * 0.5)
        self.play(FadeIn(dot))
        self.play(t.animate.set_value(wu), FadeIn(wlab), run_time=0.8)
        self.play(t.animate.set_value(1.0), FadeIn(dlab), run_time=2.2, rate_func=linear)
        self.wait(0.6)


class LossMask(Scene):
    """#12 — train only on the answer; mask the prompt."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("Instruction tuning: compute loss only on the answer",
                     color=INK).scale(0.52).to_edge(UP, buff=0.7)
        self.play(FadeIn(title))
        prompt = ["QUESTION:", "say", "hi", "ANSWER:"]
        answer = ["hi", "there"]
        seq = prompt + answer
        widths = [1.8, 0.9, 0.8, 1.7, 0.9, 1.3]
        x = -5.2
        tiles = VGroup()
        kinds = []
        for w, word, isans in zip(widths, seq, [False]*4 + [True]*2):
            box = RoundedRectangle(width=w, height=0.7, corner_radius=0.08,
                                   stroke_color=MINT if isans else "#3a4658", stroke_width=2,
                                   fill_color=PANEL, fill_opacity=1)
            box.move_to([x + w / 2, 0, 0])
            t = Text(word, color=INK if isans else MUTED).scale(0.34).move_to(box)
            tiles.add(VGroup(box, t)); kinds.append(isans)
            x += w + 0.18
        self.play(LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=0.1))
        self.wait(0.3)
        # dim the prompt, mark loss on the answer tokens
        dim = VGroup(*[tiles[i] for i in range(len(kinds)) if not kinds[i]])
        ans = [tiles[i] for i in range(len(kinds)) if kinds[i]]
        masklab = Text("prompt — masked (no loss)", color=MUTED).scale(0.36).next_to(dim, DOWN, buff=0.5)
        self.play(dim.animate.set_opacity(0.3), FadeIn(masklab))
        grads = VGroup(*[Arrow(a.get_top() + UP * 0.5, a.get_top() + UP * 0.05,
                               color=MINT, buff=0, stroke_width=4,
                               max_tip_length_to_length_ratio=0.5) for a in ans])
        glab = Text("loss here", color=MINT).scale(0.36).next_to(VGroup(*ans), UP, buff=0.6)
        self.play(LaggedStart(*[GrowArrow(g) for g in grads], lag_ratio=0.2), FadeIn(glab))
        self.wait(0.7)


class TurboQuant(Scene):
    """#16 — rotate to spread outliers, then quantize to a few levels."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("TurboQuant: rotate to spread outliers, then quantize",
                     color=INK).scale(0.5).to_edge(UP, buff=0.6)
        self.play(FadeIn(title))
        xs = [-3.6 + i * 0.9 for i in range(9)]
        outlier = [0.5, 0.4, 3.6, 0.5, 0.4, 0.5, 3.2, 0.4, 0.5]
        spread = [1.5, 1.7, 1.4, 1.6, 1.5, 1.7, 1.5, 1.6, 1.5]
        bars = VGroup(*[vbar(x, h, AMBER) for x, h in zip(xs, outlier)])
        s1 = Text("a few outlier channels dominate the range", color=AMBER).scale(0.4).to_edge(DOWN, buff=0.8)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.06), FadeIn(s1))
        self.wait(0.5)
        sb = VGroup(*[vbar(x, h, MINT) for x, h in zip(xs, spread)])
        s2 = Text("rotation spreads the energy evenly", color=MINT).scale(0.4).to_edge(DOWN, buff=0.8)
        self.play(ReplacementTransform(bars, sb), ReplacementTransform(s1, s2), run_time=1.3)
        self.wait(0.4)
        levels = VGroup(*[DashedLine([-4.4, BASE + h, 0], [4.4, BASE + h, 0],
                                     color="#3a4658", stroke_width=1.5)
                          for h in (0.8, 1.6, 2.4)])
        s3 = Text("now 3 bits per value is nearly lossless", color=INK).scale(0.4).to_edge(DOWN, buff=0.8)
        self.play(Create(levels), ReplacementTransform(s2, s3))
        snapped = [round(h / 0.8) * 0.8 for h in spread]
        qb = VGroup(*[vbar(x, h, MINT) for x, h in zip(xs, snapped)])
        self.play(ReplacementTransform(sb, qb), run_time=1.0)
        self.wait(0.7)
