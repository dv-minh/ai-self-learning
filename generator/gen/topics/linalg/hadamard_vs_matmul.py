"""1.5 hadamard-vs-matmul: phân biệt nhân ma trận và tích từng phần tử (Hadamard)."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix

from ...core import latex as L
from ...core.nice import nontrivial_matrix
from ...core.verify import check, verify_hadamard, verify_matmul
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, build

TYPE, LESSON, LEVELS = "hadamard-vs-matmul", "1.5", (1, 4)


def _hadamard(a: Matrix, b: Matrix) -> Matrix:
    return a.multiply_elementwise(b)


def _identify(rng: random.Random, level: int, seed: int) -> Exercise:
    """C là kết quả của phép nào?"""
    n = 2 if level == 1 else 3
    lo, hi = (-5, 5) if level == 1 else (-4, 4)
    for _ in range(200):
        a = nontrivial_matrix(rng, n, n, lo, hi)
        b = nontrivial_matrix(rng, n, n, lo, hi)
        ops = {"mm": a * b, "had": _hadamard(a, b), "ba": b * a, "add": a + b}
        if len({tuple(v) for v in ops.values()}) == 4:
            break
    else:
        raise RuntimeError("không dựng được cặp phân biệt")
    verified = verify_matmul(a, b, ops["mm"]) + verify_hadamard(a, b, ops["had"])
    target = rng.choice(["mm", "had", "ba"] if level == 1 else ["mm", "had", "ba", "add"])
    labels = {
        "mm": "$\\mathbf{AB}$ (nhân ma trận)",
        "had": "$\\mathbf{A}\\odot\\mathbf{B}$ (nhân từng phần tử)",
        "ba": "$\\mathbf{BA}$ (nhân ma trận)",
        "add": "$\\mathbf{A}+\\mathbf{B}$",
    }
    keys = list(labels)
    rng.shuffle(keys)
    choices = [labels[k] for k in keys]
    correct = keys.index(target)
    c = ops[target]
    names = {"mm": "AB", "had": "A \\odot B", "ba": "BA", "add": "A+B"}
    steps = [Step(text="Tính thử từng phép rồi so với $\\mathbf{C}$. Phần tử đầu tiên giúp loại nhanh các phép sai:")]
    for k in ["mm", "had", "ba", "add"]:
        hit = ops[k][0, 0] == c[0, 0]
        steps.append(Step(text=f"{labels[k]}, phần tử $(1,1)$ " + ("khớp với $\\mathbf{C}$:" if hit else "khác $\\mathbf{C}$:"),
                          latex=f"{L.num(ops[k][0, 0])}"))
    steps.append(Step(text="Kiểm tra các phần tử còn lại của phép được chọn để chắc chắn.", latex=L.mat(c, L.bold("C"))))
    return build(
        type_=TYPE, lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Ma trận $\\mathbf{C}$ dưới đây là kết quả của phép nào?",
                      display=[L.mat(a, L.bold("A")), L.mat(b, L.bold("B")), L.mat(c, L.bold("C"))]),
        answer=Answer(kind="choice", correct=[correct]),
        choices=choices,
        hints=["Tích từng phần tử: $c_{ij} = a_{ij}b_{ij}$.", "Nhân ma trận: $c_{ij}$ là hàng $i$ nhân cột $j$."],
        steps=steps, verified=verified, tags=["identify", target],
    )


def _defined(rng: random.Random, seed: int) -> Exercise:
    """Chọn mọi phép xác định với kích thước cho trước."""
    shp = lambda: (rng.randint(1, 4), rng.randint(1, 4))
    a_s = shp()
    b_s = a_s if rng.random() < 0.4 else (a_s[1] if rng.random() < 0.5 else rng.randint(1, 4), rng.randint(1, 4))
    ops = {
        "$\\mathbf{AB}$": (lambda a, b: a @ b),
        # Hadamard theo toán yêu cầu cùng kích thước; `*` của NumPy sẽ broadcast nên phải chặn thủ công.
        "$\\mathbf{A}\\odot\\mathbf{B}$": (lambda a, b: a * b if a.shape == b.shape else (_ for _ in ()).throw(ValueError("shape"))),
        "$\\mathbf{A}\\T\\mathbf{B}$": (lambda a, b: a.T @ b),
        "$\\mathbf{A}\\mathbf{B}\\T$": (lambda a, b: a @ b.T),
    }
    why = {
        "$\\mathbf{AB}$": ("số cột $\\mathbf{A}$ = số hàng $\\mathbf{B}$", a_s[1] == b_s[0]),
        "$\\mathbf{A}\\odot\\mathbf{B}$": ("hai ma trận cùng kích thước", a_s == b_s),
        "$\\mathbf{A}\\T\\mathbf{B}$": ("số hàng $\\mathbf{A}$ = số hàng $\\mathbf{B}$", a_s[0] == b_s[0]),
        "$\\mathbf{A}\\mathbf{B}\\T$": ("số cột $\\mathbf{A}$ = số cột $\\mathbf{B}$", a_s[1] == b_s[1]),
    }
    choices = list(ops)
    correct = []
    for i, k in enumerate(choices):
        try:
            ops[k](np.zeros(a_s), np.zeros(b_s))
            ok = True
        except ValueError:
            ok = False
        check(ok == why[k][1], f"điều kiện {k} không khớp NumPy")
        if ok:
            correct.append(i)
    if not correct:  # tránh đáp án rỗng
        return _defined(rng, seed)
    steps = [Step(text=f"Điều kiện của {k}: {why[k][0]}. " + ("Thỏa." if why[k][1] else "Không thỏa."))
             for k in choices]
    return build(
        type_=TYPE, lesson=LESSON, level=4, seed=seed,
        prompt=Prompt(text=f"$\\mathbf{{A}}$ có kích thước ${a_s[0]}\\times {a_s[1]}$ và $\\mathbf{{B}}$ có kích thước "
                           f"${b_s[0]}\\times {b_s[1]}$. Chọn tất cả các phép xác định."),
        answer=Answer(kind="multi", correct=correct),
        choices=choices,
        hints=["Mỗi phép có một điều kiện về kích thước riêng; hãy viết ra từng điều kiện."],
        steps=steps, verified=["sympy", "numpy", "shape"], tags=["defined-set"],
    )


def generate(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 4 and rng.random() < 0.5:
        return _defined(rng, seed)
    return _identify(rng, level, seed)


SPEC = TopicSpec(TYPE, LESSON, LEVELS, generate)
