"""1.5 matmul-shape: kích thước của tích hai ma trận, hoặc "không xác định"."""
from __future__ import annotations

import random

from ...core import latex as L
from ...core.verify import matmul_shape
from ...schema import Exercise, Prompt, Step
from ._base import TopicSpec, build, shape_answer, shape_tex

TYPE, LESSON, LEVELS = "matmul-shape", "1.5", (1, 2)


def generate(rng: random.Random, level: int, seed: int) -> Exercise:
    hi = 3 if level == 1 else 4
    m, n = rng.randint(1, hi), rng.randint(1, hi)
    q = rng.randint(1, hi)
    p = n if rng.random() < 0.6 else rng.randint(1, hi)
    a_shape, b_shape = (m, n), (p, q)
    order = rng.choice(["AB", "BA"])
    if order == "AB":
        first, second = a_shape, b_shape
    else:
        first, second = b_shape, a_shape
    result = matmul_shape(first, second)
    f, s = order[0], order[1]
    fs, ss = first, second
    text = (
        f"Cho $\\mathbf{{A}}$ có kích thước ${shape_tex(a_shape)}$ và $\\mathbf{{B}}$ có kích thước "
        f"${shape_tex(b_shape)}$. Tích $\\mathbf{{{order}}}$ có kích thước bao nhiêu? "
        f"Nếu phép nhân không xác định, hãy chọn \"không xác định\"."
    )
    ok = fs[1] == ss[0]
    steps = [
        Step(
            text=f"Số cột của ${f}$ là {fs[1]}, số hàng của ${s}$ là {ss[0]}.",
            latex=f"({shape_tex(fs)})({shape_tex(ss)})",
        )
    ]
    if ok:
        steps.append(
            Step(text="Hai số này bằng nhau nên phép nhân xác định; kích thước là (số hàng của thừa số đầu) × (số cột của thừa số sau).",
                 latex=f"{shape_tex(result)}")
        )
    else:
        steps.append(Step(text="Hai số này khác nhau nên phép nhân không xác định."))
    return build(
        type_=TYPE, lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=text),
        answer=shape_answer(result),
        hints=["Điều kiện để nhân $(m\\times n)(p\\times q)$ là gì: so sánh $n$ với $p$.",
               "Nếu xác định, kích thước kết quả là $m\\times q$."],
        steps=steps, verified=["sympy", "numpy", "shape"],
        tags=["defined" if ok else "undefined"],
    )


SPEC = TopicSpec(TYPE, LESSON, LEVELS, generate)
