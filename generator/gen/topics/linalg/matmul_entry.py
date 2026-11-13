"""1.5 matmul-entry: tính một phần tử c_ij của tích AB."""
from __future__ import annotations

import random

from ...core import latex as L
from ...core.nice import nontrivial_matrix
from ...core.verify import verify_matmul
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, build
from ...core.steps import matmul_entry_step

TYPE, LESSON, LEVELS = "matmul-entry", "1.5", (2, 3)


def generate(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        hi, lo_v, hi_v, fp = 3, -5, 5, 0.0
    else:
        hi, lo_v, hi_v, fp = 4, -9, 9, 0.2
    m, n, p = rng.randint(2, hi), rng.randint(2, hi), rng.randint(2, hi)
    a = nontrivial_matrix(rng, m, n, lo_v, hi_v, fp)
    b = nontrivial_matrix(rng, n, p, lo_v, hi_v, fp)
    c = a * b
    verified = verify_matmul(a, b, c)
    i, j = rng.randrange(m), rng.randrange(p)
    sub = f"c_{{{i + 1}{j + 1}}}"
    return build(
        type_=TYPE, lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(
            text=f"Cho $\\mathbf{{C}} = \\mathbf{{A}}\\mathbf{{B}}$. Tính phần tử ${sub}$.",
            display=[L.mat(a, L.bold("A")), L.mat(b, L.bold("B"))],
        ),
        answer=Answer(kind="number", value=L.rat_str(c[i, j])),
        hints=[f"${sub}$ nằm ở hàng {i + 1}, cột {j + 1} của $\\mathbf{{C}}$.",
               f"Lấy hàng {i + 1} của $\\mathbf{{A}}$ nhân từng phần tử với cột {j + 1} của $\\mathbf{{B}}$ rồi cộng lại."],
        steps=[matmul_entry_step(a, b, i, j, L.bold("A"), L.bold("B"))],
        verified=verified, tags=[f"{m}x{n}", f"{n}x{p}"],
    )


SPEC = TopicSpec(TYPE, LESSON, LEVELS, generate)
