"""1.5 matmul-noncommute: AB có bằng BA không?"""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix

from ...core import latex as L
from ...core.nice import is_trivial, nontrivial_matrix
from ...core.verify import check, to_np, verify_matmul
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, build
from ...core.steps import matmul_steps

TYPE, LESSON, LEVELS = "matmul-noncommute", "1.5", (4,)


def generate(rng: random.Random, level: int, seed: int) -> Exercise:
    n = rng.choice([2, 2, 3])
    a = nontrivial_matrix(rng, n, n, -3, 3)
    want_equal = rng.random() < 0.5
    b = None
    if want_equal:
        # B là đa thức của A nên chắc chắn giao hoán với A
        p, q = rng.randint(-3, 3), rng.choice([-2, -1, 1, 2])
        cand = p * Matrix.eye(n) + q * a
        if not is_trivial(cand) and cand != a:
            b = cand
    if b is None:
        for _ in range(100):
            cand = nontrivial_matrix(rng, n, n, -3, 3)
            if cand * a != a * cand:
                b = cand
                break
    if b is None:
        raise RuntimeError("không dựng được cặp")
    ab, ba = a * b, b * a
    verified = verify_matmul(a, b, ab) + verify_matmul(b, a, ba)
    equal = ab == ba
    steps = [Step(text="Tính $\\mathbf{AB}$:")] + matmul_steps(a, b, L.bold("A"), L.bold("B"))[1:]
    steps += [Step(text="Tính $\\mathbf{BA}$:")] + matmul_steps(b, a, L.bold("B"), L.bold("A"))[1:]
    steps.append(Step(text=("Hai ma trận bằng nhau từng phần tử, nên khẳng định đúng." if equal
                            else "Hai ma trận khác nhau ở ít nhất một phần tử, nên khẳng định sai (nhân ma trận không giao hoán).")))
    check((ab == ba) == (np.allclose(to_np(ab), to_np(ba))), "noncommute: SymPy và NumPy bất đồng")
    return build(
        type_=TYPE, lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Khẳng định $\\mathbf{AB} = \\mathbf{BA}$ đúng hay sai?",
                      display=[L.mat(a, L.bold("A")), L.mat(b, L.bold("B"))]),
        answer=Answer(kind="boolean", value="true" if equal else "false"),
        hints=["Chỉ cần tìm một phần tử mà $\\mathbf{AB}$ và $\\mathbf{BA}$ khác nhau là kết luận được sai.",
               "Nếu các phần tử đầu giống nhau thì phải kiểm tra toàn bộ."],
        steps=steps, verified=verified, tags=["equal" if equal else "differ", f"{n}x{n}"],
    )


SPEC = TopicSpec(TYPE, LESSON, LEVELS, generate)
