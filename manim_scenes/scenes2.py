"""
More Manim scenes: #05 autograd, #09 generation, #13 DPO.

Render (then copy mp4 to docs/assets/anim/<NN>.mp4):
  .venv/bin/manim -qm --media_dir /tmp/manim_media -o 05 manim_scenes/scenes2.py Autograd
  .venv/bin/manim -qm --media_dir /tmp/manim_media -o 09 manim_scenes/scenes2.py Generate
  .venv/bin/manim -qm --media_dir /tmp/manim_media -o 13 manim_scenes/scenes2.py DPO
"""

from manim import *

BG = "#0a0c10"
MINT = "#5ef0c0"
AMBER = "#f2b65e"
ROSE = "#f08a8a"
INK = "#ece8de"
MUTED = "#9aa1ad"
PANEL = "#12151c"


class Autograd(Scene):
    """#05 — forward values flow right, gradients flow back left."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("Autograd: build the graph going forward, send gradients back",
                     color=INK).scale(0.5).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        labels = ["x", "× w", "h", "+ b", "z", "tanh", "out"]
        xs = [-6, -4, -2, 0, 2, 4, 6]
        nodes = []
        for lab, x in zip(labels, xs):
            box = RoundedRectangle(width=1.4, height=0.85, corner_radius=0.1,
                                   stroke_color=MINT, stroke_width=2.5,
                                   fill_color=PANEL, fill_opacity=1).move_to([x, 0, 0])
            t = Text(lab, color=INK).scale(0.4).move_to(box)
            nodes.append(VGroup(box, t))
        arrows = [Arrow(nodes[i].get_right(), nodes[i + 1].get_left(), buff=0.1,
                        color="#5a6472", stroke_width=3,
                        max_tip_length_to_length_ratio=0.4) for i in range(len(nodes) - 1)]

        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.15),
                  *[GrowArrow(a) for a in arrows], run_time=1.6)

        fwd = Text("forward — compute each value", color=MINT).scale(0.4).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(fwd))
        for a in arrows:
            self.play(Indicate(a, color=MINT, scale_factor=1.15), run_time=0.28)
        self.play(FadeOut(fwd))

        bwd = Text("backward — chain rule sends ∂loss/∂each back", color=AMBER).scale(0.4).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(bwd))
        for a in reversed(arrows):
            self.play(Indicate(a, color=AMBER, scale_factor=1.15), run_time=0.28)
        self.wait(0.6)


class Generate(Scene):
    """#09 — predict the next token, append, repeat."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("Autoregressive generation: predict, append, repeat",
                     color=INK).scale(0.52).to_edge(UP, buff=0.5)
        model = RoundedRectangle(width=2.6, height=1.0, corner_radius=0.14,
                                 stroke_color=MINT, stroke_width=3,
                                 fill_color=PANEL, fill_opacity=1).shift(UP * 1.4)
        mlbl = Text("tiny GPT", color=MINT).scale(0.42).move_to(model)
        self.play(FadeIn(title), FadeIn(model), FadeIn(mlbl))

        chars = list("ROMEO:")
        n = len(chars)
        start_x = -(n - 1) * 0.75 / 1 * 0.5 - 1.2
        tiles = VGroup()
        for i, ch in enumerate(chars):
            x = -2.0 + i * 0.8
            tile = Square(side_length=0.7, stroke_color="#3a4658", stroke_width=2,
                          fill_color=PANEL, fill_opacity=1).move_to([x, -1.6, 0])
            ct = Text(ch, color=INK).scale(0.45).move_to(tile)
            t = VGroup(tile, ct)
            arrow = Arrow(model.get_bottom(), tile.get_top(), buff=0.15,
                          color=AMBER, stroke_width=3, max_tip_length_to_length_ratio=0.3)
            self.play(Indicate(model, color=MINT, scale_factor=1.05), run_time=0.3)
            self.play(GrowArrow(arrow), run_time=0.3)
            self.play(FadeIn(t, shift=UP * 0.2), FadeOut(arrow), run_time=0.4)
            tiles.add(t)
        self.wait(0.7)


class DPO(Scene):
    """#13 — preference tuning widens the chosen-vs-rejected gap."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("DPO: push the chosen answer up, the rejected one down",
                     color=INK).scale(0.5).to_edge(UP, buff=0.6)
        self.play(FadeIn(title))

        base = -2.2
        ch = ValueTracker(2.0)
        rj = ValueTracker(2.0)

        def bar(tr, x, color):
            return always_redraw(lambda: Rectangle(
                width=1.6, height=max(tr.get_value(), 0.01),
                fill_color=color, fill_opacity=0.9, stroke_width=0
            ).move_to([x, base + tr.get_value() / 2, 0]))

        cbar = bar(ch, -2.2, MINT)
        rbar = bar(rj, 2.2, ROSE)
        clab = Text("chosen", color=MINT).scale(0.42).move_to([-2.2, base - 0.4, 0])
        rlab = Text("rejected", color=ROSE).scale(0.42).move_to([2.2, base - 0.4, 0])
        floor = Line([-4.2, base, 0], [4.2, base, 0], color="#3a4658", stroke_width=2)

        self.play(FadeIn(floor), FadeIn(clab), FadeIn(rlab))
        self.play(FadeIn(cbar), FadeIn(rbar))
        self.wait(0.4)
        note = Text("preference gap widens", color=AMBER).scale(0.4).to_edge(DOWN, buff=0.7)
        self.play(ch.animate.set_value(3.6), rj.animate.set_value(0.6),
                  FadeIn(note), run_time=2.2, rate_func=smooth)
        self.wait(0.7)
