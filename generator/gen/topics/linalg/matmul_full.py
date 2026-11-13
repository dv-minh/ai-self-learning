"""1.5 matmul-full: tính toàn bộ tích AB."""
from __future__ import annotations

import random

from ...core import latex as L
from ...core.nice import nontrivial_matrix
from ...core.steps import matmul_steps
from ...core.verify import verify_matmul
from ...schema import Answer, Exercise, Prompt
from ._base import TopicSpec, build

TYPE, LESSON, LEVELS = "matmul-full", "1.5", (2, 3)


def generate(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        shapes = [(2, 2, 2), (2, 3, 2), (3, 2, 2), (2, 2, 3), (3, 3, 2), (2, 3, 3), (3, 3, 3)]
        lo_v, hi_v, fp = -5, 5, 0.0
    else:
        shapes = [(4, 4, 4), (2, 4, 4), (4, 3, 3), (4, 4, 2), (3, 4, 4), (4, 3, 4), (2, 4, 3), (3, 3, 4)]
        lo_v, hi_v, fp = -9, 9, 0.15
    m, n, p = rng.choice(shapes)
    a = nontrivial_matrix(rng, m, n, lo_v, hi_v, fp)
    b = nontrivial_matrix(rng, n, p, lo_v, hi_v, fp)
    c = a * b
    verified = verify_matmul(a, b, c)
    return build(
        type_=TYPE, lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(
            text="Tính tích $\\mathbf{A}\\mathbf{B}$.",
            display=[L.mat(a, L.bold("A")), L.mat(b, L.bold("B"))],
        ),
        answer=Answer(kind="matrix", rows=c.rows, cols=c.cols, entries=L.entries(c)),
        hints=[f"Kích thước của $\\mathbf{{AB}}$ là ${m}\\times {p}$.",
               "Phần tử $c_{ij}$ = hàng $i$ của $\\mathbf{A}$ nhân cột $j$ của $\\mathbf{B}$.",
               "Tính lần lượt từng phần tử, hàng trên trước."],
        steps=matmul_steps(a, b, L.bold("A"), L.bold("B")),
        verified=verified, tags=[f"{m}x{n}", f"{n}x{p}"],
    )


SPEC = TopicSpec(TYPE, LESSON, LEVELS, generate)
