"""1.5 matvec: ma trận nhân vector cột."""
from __future__ import annotations

import random

from ...core import latex as L
from ...core.nice import nontrivial_matrix
from ...core.steps import matmul_steps
from ...core.verify import verify_matmul
from ...schema import Answer, Exercise, Prompt
from ._base import TopicSpec, build

TYPE, LESSON, LEVELS = "matvec", "1.5", (2, 3)


def generate(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        hi, lo_v, hi_v, fp = 3, -5, 5, 0.0
    else:
        hi, lo_v, hi_v, fp = 4, -9, 9, 0.15
    m, n = rng.randint(2, hi), rng.randint(2, hi)
    a = nontrivial_matrix(rng, m, n, lo_v, hi_v, fp)
    x = nontrivial_matrix(rng, n, 1, lo_v, hi_v, fp)
    y = a * x
    verified = verify_matmul(a, x, y)
    return build(
        type_=TYPE, lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(
            text="Tính $\\mathbf{y} = \\mathbf{A}\\mathbf{x}$.",
            display=[L.mat(a, L.bold("A")), L.mat(x, L.bold("x"))],
        ),
        answer=Answer(kind="vector", rows=m, cols=1, entries=L.entries(y)),
        hints=[f"$\\mathbf{{y}}$ có {m} phần tử (bằng số hàng của $\\mathbf{{A}}$).",
               "Phần tử thứ $i$ của $\\mathbf{y}$ là hàng $i$ của $\\mathbf{A}$ nhân vô hướng với $\\mathbf{x}$."],
        steps=matmul_steps(a, x, L.bold("A"), L.bold("x")),
        verified=verified, tags=[f"{m}x{n}"],
    )


SPEC = TopicSpec(TYPE, LESSON, LEVELS, generate)
