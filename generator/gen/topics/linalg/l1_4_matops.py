"""Bài 1.4 Cộng ma trận, nhân vô hướng: mat-add, mat-scale, mat-lincomb."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational

from ...core import latex as L
from ...core.nice import SIMPLE_FRACTIONS, nontrivial_matrix
from ...core.verify import check, np_close, to_np
from ...schema import Exercise, Prompt, Step
from ._base import TopicSpec, bool_ans, build, mat_ans, shape_tex, size_tag

LESSON = "1.4"
B = L.bold
N = L.num_paren


def _mat(rng, r, c, lo, hi, fp=0.0):
    return nontrivial_matrix(rng, r, c, lo, hi, fp)


def _elementwise(parts_fn, m: Matrix) -> str:
    return L.expr_mat([[parts_fn(i, j) for j in range(m.cols)] for i in range(m.rows)])


# ------------------------------------------------------------------ mat-add
def gen_add(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 1:  # có cộng được không?
        a_shape = (rng.randint(1, 3), rng.randint(1, 3))
        b_shape = a_shape if rng.random() < 0.5 else (rng.randint(1, 3), rng.randint(1, 3))
        a, b = _mat(rng, *a_shape, -9, 9), _mat(rng, *b_shape, -9, 9)
        ok = a_shape == b_shape
        try:
            np.zeros(a_shape) + np.zeros(b_shape)
            numpy_ok = True
        except ValueError:
            numpy_ok = False
        # NumPy broadcast: chỉ dùng làm tham chiếu khi hai shape bằng nhau hoặc không broadcast được
        if ok != numpy_ok:
            check(not ok and numpy_ok, "mat-add: lệch NumPy ngoài trường hợp broadcast")
        return build(
            type_="mat-add", lesson=LESSON, level=1, seed=seed,
            prompt=Prompt(text="Khẳng định \"phép cộng $\\mathbf{A}+\\mathbf{B}$ xác định\" đúng hay sai?",
                          display=[L.mat(a, B("A")), L.mat(b, B("B"))]),
            answer=bool_ans(ok),
            hints=["Hai ma trận chỉ cộng được khi cùng kích thước (cùng số hàng và cùng số cột)."],
            steps=[Step(text=f"$\\mathbf{{A}}$ là ${shape_tex(a_shape)}$ và $\\mathbf{{B}}$ là ${shape_tex(b_shape)}$."),
                   Step(text="Cùng kích thước nên cộng được." if ok else "Khác kích thước nên không cộng được.")],
            verified=["sympy", "shape"], tags=["defined" if ok else "undefined", size_tag(a)],
        )
    if level == 2:
        r, c = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
        lo, hi, fp = -5, 5, 0.0
    else:
        r, c = rng.choice([(4, 4), (3, 4), (4, 3), (4, 2), (2, 4)])
        lo, hi, fp = -9, 9, 0.2
    a, b = _mat(rng, r, c, lo, hi, fp), _mat(rng, r, c, lo, hi, fp)
    sub = rng.random() < 0.4
    res = a - b if sub else a + b
    np_close(res, to_np(a) - to_np(b) if sub else to_np(a) + to_np(b))
    op = "-" if sub else "+"
    return build(
        type_="mat-add", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính $\\mathbf{{A}} {op} \\mathbf{{B}}$.", display=[L.mat(a, B("A")), L.mat(b, B("B"))]),
        answer=mat_ans(res),
        hints=["Cộng (hoặc trừ) các phần tử ở cùng vị trí."],
        steps=[Step(text="Tính từng phần tử:",
                    latex=f"\\mathbf{{A}} {op} \\mathbf{{B}} = " + _elementwise(lambda i, j: f"{N(a[i, j])} {op} {N(b[i, j])}", a) + f" = {L.mat(res)}")],
        verified=["sympy", "numpy", "shape"], tags=[size_tag(a), "sub" if sub else "add"],
    )


# ----------------------------------------------------------------- mat-scale
def gen_scale(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        r, c = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
        lo, hi = -5, 5
        k = Rational(rng.choice([x for x in range(-4, 5) if x not in (0, 1)]))
    else:
        r, c = rng.choice([(4, 4), (3, 4), (4, 3)])
        lo, hi = -9, 9
        k = rng.choice(SIMPLE_FRACTIONS + [Rational(-3), Rational(5)])
    a = _mat(rng, r, c, lo, hi)
    res = k * a
    np_close(res, float(k) * to_np(a))
    return build(
        type_="mat-scale", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính ${L.num(k)}\\mathbf{{A}}$.", display=[L.mat(a, B("A"))]),
        answer=mat_ans(res),
        hints=["Nhân số vô hướng với mọi phần tử."],
        steps=[Step(text="Nhân từng phần tử với số vô hướng:",
                    latex=f"{L.num(k)}\\mathbf{{A}} = " + _elementwise(lambda i, j: f"{N(k)} \\cdot {N(a[i, j])}", a) + f" = {L.mat(res)}")],
        verified=["sympy", "numpy", "shape"], tags=[size_tag(a)],
    )


# --------------------------------------------------------------- mat-lincomb
def gen_lincomb(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 5:  # mixup: lam*A + (1-lam)*B
        r, c = rng.choice([(2, 2), (2, 3), (3, 2)])
        a, b = _mat(rng, r, c, 0, 9), _mat(rng, r, c, 0, 9)
        lam = rng.choice([Rational(1, 4), Rational(1, 2), Rational(3, 4), Rational(1, 3), Rational(2, 3)])
        res = lam * a + (1 - lam) * b
        np_close(res, float(lam) * to_np(a) + (1 - float(lam)) * to_np(b))
        return build(
            type_="mat-lincomb", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text=f"Kỹ thuật \"mixup\" trộn hai ảnh (ma trận điểm ảnh) thành ảnh mới "
                               f"$\\lambda\\mathbf{{A}} + (1-\\lambda)\\mathbf{{B}}$. Với $\\lambda = {L.num(lam)}$, tính ảnh trộn.",
                          display=[L.mat(a, B("A")), L.mat(b, B("B"))]),
            answer=mat_ans(res),
            hints=[f"Tính $1-\\lambda = {L.num(1 - lam)}$.", "Nhân từng ma trận với hệ số rồi cộng."],
            steps=[Step(text="Hệ số của $\\mathbf{B}$:", latex=f"1-\\lambda = {L.num(1 - lam)}"),
                   Step(text="Tính từng phần tử:",
                        latex=_elementwise(lambda i, j: f"{N(lam)}\\cdot {N(a[i, j])} + {N(1 - lam)}\\cdot {N(b[i, j])}", a) + f" = {L.mat(res)}")],
            verified=["sympy", "numpy", "shape"], tags=["ml", size_tag(a)],
        )
    if level == 4:  # giải 2X + A = B  hoặc  A - kX = B
        r, c = rng.choice([(2, 2), (2, 3), (3, 3)])
        a, x0 = _mat(rng, r, c, -5, 5), _mat(rng, r, c, -5, 5)
        k = Rational(rng.choice([-3, -2, 2, 3, 4]))
        b = a + k * x0
        x = (b - a) / k
        check(x == x0 and a + k * x == b, "lincomb4")
        np_close(x, (to_np(b) - to_np(a)) / float(k))
        return build(
            type_="mat-lincomb", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"Tìm ma trận $\\mathbf{{X}}$ thỏa $\\mathbf{{A}} + {L.num(k)}\\mathbf{{X}} = \\mathbf{{B}}$.",
                          display=[L.mat(a, B("A")), L.mat(b, B("B"))]),
            answer=mat_ans(x),
            hints=["Chuyển $\\mathbf{A}$ sang vế phải: ${k}\\mathbf{X} = \\mathbf{B}-\\mathbf{A}$.".replace("{k}", L.num(k)),
                   "Rồi chia mọi phần tử cho hệ số của $\\mathbf{X}$."],
            steps=[Step(text="Tính $\\mathbf{B}-\\mathbf{A}$:", latex=L.mat(b - a, "\\mathbf{B}-\\mathbf{A}")),
                   Step(text=f"Chia cho ${L.num(k)}$:", latex=L.mat(x, "\\mathbf{X}"))],
            verified=["sympy", "numpy", "substitution"], tags=["solve", size_tag(a)],
        )
    if level == 2:
        r, c = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
        lo, hi, terms = -4, 4, 2
    else:
        r, c = rng.choice([(3, 3), (4, 3), (3, 4), (4, 4)])
        lo, hi, terms = -6, 6, 3
    names = ["A", "B", "C"][:terms]
    ms = [_mat(rng, r, c, lo, hi) for _ in range(terms)]
    ks = [Rational(rng.choice([x for x in range(-3, 4) if x not in (0, 1)])) for _ in range(terms)]
    res = sum((k * m for k, m in zip(ks, ms)), Matrix.zeros(r, c))
    np_close(res, sum(float(k) * to_np(m) for k, m in zip(ks, ms)))
    expr = " + ".join(f"{N(k)}\\mathbf{{{nm}}}" for k, nm in zip(ks, names))
    cell = lambda i, j: " + ".join(f"{N(k)}\\cdot {N(m[i, j])}" for k, m in zip(ks, ms))
    return build(
        type_="mat-lincomb", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính ${expr}$.", display=[L.mat(m, B(nm)) for m, nm in zip(ms, names)]),
        answer=mat_ans(res),
        hints=["Nhân từng ma trận với hệ số, rồi cộng các ma trận cùng kích thước."],
        steps=[Step(text="Tính từng phần tử:", latex=_elementwise(cell, ms[0]) + f" = {L.mat(res)}")],
        verified=["sympy", "numpy", "shape"], tags=[size_tag(ms[0]), f"{terms}terms"],
    )


SPECS = [
    TopicSpec("mat-add", LESSON, (1, 2, 3), gen_add),
    TopicSpec("mat-scale", LESSON, (2, 3), gen_scale),
    TopicSpec("mat-lincomb", LESSON, (2, 3, 4, 5), gen_lincomb),
]
