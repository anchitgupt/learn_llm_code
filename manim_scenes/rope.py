"""
Manim scene for step #14: Rotary Position Embeddings (RoPE).

Shows a query and key vector being rotated together by a position angle — the
angle between them (and therefore their dot product, i.e. the attention score)
stays constant. That is the whole idea behind RoPE.

Render:
  .venv/bin/manim -qm --media_dir /tmp/manim_media -o 14 manim_scenes/rope.py RoPE
then copy the mp4 to docs/assets/anim/14.mp4
"""

from manim import *

BG = "#0a0c10"
MINT = "#5ef0c0"
AMBER = "#f2b65e"
INK = "#ece8de"
MUTED = "#9aa1ad"


class RoPE(Scene):
    def construct(self):
        self.camera.background_color = BG

        plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#223040",
                                   "stroke_width": 1, "stroke_opacity": 0.5},
            axis_config={"stroke_color": "#3a4658", "stroke_width": 1},
        )

        title = Text("RoPE: rotate q and k by position", color=INK).scale(0.6).to_edge(UP, buff=0.55)
        sub = Text("the angle between them stays fixed — so the dot product (the score) is preserved",
                   color=MUTED).scale(0.32).next_to(title, DOWN, buff=0.18)

        q = Arrow(ORIGIN, [2.4, 1.2, 0], buff=0, color=MINT, stroke_width=7,
                  max_tip_length_to_length_ratio=0.18)
        k = Arrow(ORIGIN, [2.8, -0.5, 0], buff=0, color=AMBER, stroke_width=7,
                  max_tip_length_to_length_ratio=0.18)
        ql = Text("q", color=MINT).scale(0.55)
        kl = Text("k", color=AMBER).scale(0.55)
        ql.add_updater(lambda m: m.move_to(q.get_end() + 0.3 * normalize(q.get_vector())))
        kl.add_updater(lambda m: m.move_to(k.get_end() + 0.3 * normalize(k.get_vector())))
        arc = always_redraw(lambda: Angle(k, q, radius=0.7, color=INK, stroke_width=3))

        self.play(Create(plane), run_time=1.1)
        self.play(FadeIn(title), FadeIn(sub))
        self.play(GrowArrow(q), GrowArrow(k), FadeIn(ql), FadeIn(kl))
        self.play(Create(arc))
        self.wait(0.4)

        vecs = VGroup(q, k)
        for ang in (PI / 3, PI / 4, -PI / 2, PI / 5):
            self.play(Rotate(vecs, angle=ang, about_point=ORIGIN),
                      rate_func=smooth, run_time=1.5)
            self.wait(0.25)
        self.wait(0.6)
