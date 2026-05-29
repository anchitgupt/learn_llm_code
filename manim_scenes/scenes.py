"""
Extra Manim scenes for the ladder.

Render each (then copy the mp4 into docs/assets/anim/<NN>.mp4):
  .venv/bin/manim -qm --media_dir /tmp/manim_media -o 00 manim_scenes/scenes.py GradDescent
  .venv/bin/manim -qm --media_dir /tmp/manim_media -o 08 manim_scenes/scenes.py TransformerStack
"""

from manim import *

BG = "#0a0c10"
MINT = "#5ef0c0"
AMBER = "#f2b65e"
INK = "#ece8de"
MUTED = "#9aa1ad"
PANEL = "#12151c"


class GradDescent(Scene):
    """#00 — a point rolling down the loss curve via gradient steps."""

    def construct(self):
        self.camera.background_color = BG
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 9, 3], x_length=9.5, y_length=4.8,
                  axis_config={"stroke_color": "#3a4658", "stroke_width": 2,
                               "include_ticks": False})
        curve = ax.plot(lambda x: x ** 2, x_range=[-3, 3], color=AMBER, stroke_width=5)
        title = Text("Gradient descent: step downhill to minimize the loss",
                     color=INK).scale(0.55).to_edge(UP, buff=0.6)
        loss_lbl = Text("loss", color=MUTED).scale(0.4).next_to(ax.y_axis, UP, buff=0.1)

        xs, lr = [-2.7], 0.16
        for _ in range(10):
            xs.append(xs[-1] - lr * 2 * xs[-1])     # x <- x - lr * dL/dx,  L = x^2
        dot = Dot(ax.c2p(xs[0], xs[0] ** 2), color=MINT, radius=0.11)
        dot.set_z_index(3)

        self.play(Create(ax), FadeIn(loss_lbl), run_time=1.0)
        self.play(Create(curve), FadeIn(title))
        self.play(FadeIn(dot, scale=0.5))
        self.wait(0.3)
        for x in xs[1:]:
            self.play(dot.animate.move_to(ax.c2p(x, x * x)),
                      rate_func=rate_functions.ease_in_out_sine, run_time=0.55)
        self.play(Flash(dot, color=MINT, line_length=0.25, num_lines=12))
        self.wait(0.6)


class TransformerStack(Scene):
    """#08 — blocks stack because shape in == shape out (B, T, C)."""

    def construct(self):
        self.camera.background_color = BG
        title = Text("A transformer block: shape in = shape out, so they stack",
                     color=INK).scale(0.5).to_edge(UP, buff=0.5)

        def block():
            box = RoundedRectangle(width=4.2, height=0.95, corner_radius=0.12,
                                   stroke_color=MINT, stroke_width=3,
                                   fill_color=PANEL, fill_opacity=1)
            lbl = Text("attention  +  feed-forward", color=INK).scale(0.32).move_to(box)
            return VGroup(box, lbl)

        blocks = VGroup(*[block() for _ in range(3)]).arrange(UP, buff=0.95)
        blocks.shift(DOWN * 0.3)

        inp = Text("tokens + positions", color=MUTED).scale(0.34)
        inp.next_to(blocks, DOWN, buff=0.7)
        out = Text("logits  →  next token", color=AMBER).scale(0.34)
        out.next_to(blocks, UP, buff=0.7)

        def shape_arrow(a, b):
            ar = Arrow(a, b, buff=0.12, color="#5a6472", stroke_width=4,
                       max_tip_length_to_length_ratio=0.25)
            tag = Text("(B, T, C)", color=MUTED).scale(0.26).next_to(ar, RIGHT, buff=0.18)
            return VGroup(ar, tag)

        self.play(FadeIn(title))
        self.play(FadeIn(inp, shift=UP * 0.2))
        prev = inp
        for blk in blocks:
            arrow = shape_arrow(prev.get_top(), blk.get_bottom())
            self.play(GrowArrow(arrow[0]), FadeIn(arrow[1]), run_time=0.5)
            self.play(FadeIn(blk, shift=UP * 0.2), run_time=0.6)
            prev = blk
        arrow = shape_arrow(prev.get_top(), out.get_bottom())
        self.play(GrowArrow(arrow[0]), FadeIn(arrow[1]), run_time=0.5)
        self.play(FadeIn(out, shift=UP * 0.2))
        self.wait(0.7)
