"""Bài 1.1 Vector: vec-add, vec-scale, vec-lincomb, vec-norm."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, sqrt

from ...core import latex as L
from ...core.nice import SIMPLE_FRACTIONS, int_norm_vector, level_range, rand_vector
from ...core.steps import componentwise
from ...core.verify import check, np_close, to_np
from ...schema import Exercise, Prompt, Step
from ._base import TopicSpec, build, num_ans, size_tag, vec_ans

LESSON = "1.1"
B = L.bold
N = L.num_paren


def _vecs_disp(**vs: Matrix) -> list[str]:
    return [L.mat(v, B(k)) for k, v in vs.items()]


# ------------------------------------------------------------------ vec-add
def gen_add(rng: random.Random, level: int, seed: int) -> Exercise:
    n = {1: 2, 2: rng.choice([2, 3]), 3: rng.choice([3, 4])}[level]
    lo, hi = level_range(level)
    fp = 0.2 if level == 3 else 0.0
    u, v = rand_vector(rng, n, lo, hi, fp), rand_vector(rng, n, lo, hi, fp)
    sub = level >= 2 and rng.random() < 0.5
    r = u - v if sub else u + v
    np_close(r, to_np(u) - to_np(v) if sub else to_np(u) + to_np(v))
    op = "-" if sub else "+"
    parts = [f"{N(a)} {op} {N(b)}" for a, b in zip(u, v)]
    return build(
        type_="vec-add", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính $\\mathbf{{u}} {op} \\mathbf{{v}}$.", display=_vecs_disp(u=u, v=v)),
        answer=vec_ans(r),
        hints=["Hai vector phải cùng số thành phần.", f"Tính từng thành phần: $u_i {op} v_i$."],
        steps=[componentwise(f"\\mathbf{{u}} {op} \\mathbf{{v}}", parts, r)],
        verified=["sympy", "numpy", "shape"], tags=["sub" if sub else "add", f"R{n}"],
    )


# ---------------------------------------------------------------- vec-scale
def gen_scale(rng: random.Random, level: int, seed: int) -> Exercise:
    n = {1: 2, 2: rng.choice([2, 3]), 3: rng.choice([3, 4])}[level]
    lo, hi = level_range(level)
    u = rand_vector(rng, n, lo, hi)
    if level == 3:
        k = rng.choice(SIMPLE_FRACTIONS + [Rational(-3), Rational(4)])
    else:
        k = Rational(rng.choice([x for x in range(-5 if level == 2 else 2, 6) if x not in (0, 1)]))
    r = k * u
    np_close(r, float(k) * to_np(u))
    parts = [f"{N(k)} \\cdot {N(a)}" for a in u]
    return build(
        type_="vec-scale", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính ${L.num(k)}\\mathbf{{u}}$.", display=_vecs_disp(u=u)),
        answer=vec_ans(r),
        hints=["Nhân số vô hướng với từng thành phần của vector."],
        steps=[componentwise(f"{L.num(k)}\\mathbf{{u}}", parts, r)],
        verified=["sympy", "numpy", "shape"], tags=[f"R{n}"],
    )


# -------------------------------------------------------------- vec-lincomb
def _solve_coeffs(u: Matrix, v: Matrix, w: Matrix) -> tuple[Rational, Rational, tuple[int, int]]:
    """Giải w = a u + b v từ hai thành phần có định thức khác 0 (Cramer)."""
    for i in range(u.rows):
        for j in range(i + 1, u.rows):
            d = u[i] * v[j] - u[j] * v[i]
            if d != 0:
                a = (w[i] * v[j] - w[j] * v[i]) / d
                b = (u[i] * w[j] - u[j] * w[i]) / d
                return a, b, (i, j)
    raise ValueError("u, v phụ thuộc tuyến tính")


def _lincomb_find(rng: random.Random, level: int, seed: int) -> Exercise:
    n = rng.choice([2, 3])
    while True:
        u, v = rand_vector(rng, n, -4, 4), rand_vector(rng, n, -4, 4)
        if Matrix.hstack(u, v).rank() == 2:
            break
    a0, b0 = Rational(rng.randint(-4, 4)), Rational(rng.randint(-4, 4))
    if a0 == 0 and b0 == 0:
        a0 = Rational(2)
    w = a0 * u + b0 * v
    a, b, (i, j) = _solve_coeffs(u, v, w)
    check((a, b) == (a0, b0), "lincomb: Cramer sai")
    check(w == a * u + b * v, "lincomb: không thỏa")
    np_close(w, a * to_np(u) + b * to_np(v))
    steps = [
        Step(text="Viết $\\mathbf{w} = a\\mathbf{u} + b\\mathbf{v}$ theo từng thành phần:",
             latex=" \\quad ".join(f"{L.num(u[k])}a + {L.num(v[k])}b = {L.num(w[k])}" for k in range(n))),
        Step(text=f"Dùng thành phần {i + 1} và {j + 1} để giải $a, b$ (định thức hệ khác 0):",
             latex=f"a = {L.num(a)},\\quad b = {L.num(b)}"),
    ]
    if n == 3:
        steps.append(Step(text="Thành phần còn lại kiểm tra được nghiệm vừa tìm."))
    return build(
        type_="vec-lincomb", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tìm các hệ số $a, b$ sao cho $\\mathbf{w} = a\\mathbf{u} + b\\mathbf{v}$. Nhập đáp án là vector $(a, b)$.",
                      display=_vecs_disp(u=u, v=v, w=w)),
        answer=vec_ans(Matrix([a, b])),
        hints=["Mỗi thành phần cho một phương trình bậc nhất theo $a$ và $b$.",
               "Chọn hai phương trình rồi giải hệ."],
        steps=steps, verified=["sympy", "numpy", "substitution"], tags=["find", f"R{n}"],
    )


def _lincomb_analogy(rng: random.Random, level: int, seed: int) -> Exercise:
    base = [rand_vector(rng, 3, -5, 5) for _ in range(3)]
    e_a, e_b, e_c = base
    r = e_a - e_b + e_c
    parts = [f"{N(x)} - {N(y)} + {N(z)}" for x, y, z in zip(e_a, e_b, e_c)]
    return build(
        type_="vec-lincomb", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(
            text="Trong mô hình embedding (mỗi từ là một vector), phép loại suy "
                 "\"A so với B giống C so với ?\" được tính bằng $\\mathbf{q} = \\mathbf{e}_A - \\mathbf{e}_B + \\mathbf{e}_C$. "
                 "Tính $\\mathbf{q}$.",
            display=[L.mat(e_a, "\\mathbf{e}_A"), L.mat(e_b, "\\mathbf{e}_B"), L.mat(e_c, "\\mathbf{e}_C")]),
        answer=vec_ans(r),
        hints=["Cộng và trừ từng thành phần tương ứng."],
        steps=[componentwise("\\mathbf{q}", parts, r)],
        verified=["sympy", "numpy", "shape"], tags=["ml", "R3"],
    )


def gen_lincomb(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 4:
        return _lincomb_find(rng, level, seed)
    if level == 5:
        return _lincomb_analogy(rng, level, seed)
    lo, hi = level_range(level)
    n = rng.choice([2, 3]) if level == 2 else rng.choice([3, 4])
    terms = 2 if level == 2 else 3
    names = ["u", "v", "w"][:terms]
    vs = [rand_vector(rng, n, lo, hi) for _ in range(terms)]
    ks = [Rational(rng.choice([x for x in range(-4, 5) if x not in (0, 1)])) for _ in range(terms)]
    r = sum((k * v for k, v in zip(ks, vs)), Matrix.zeros(n, 1))
    np_close(r, sum(float(k) * to_np(v) for k, v in zip(ks, vs)))
    expr = " + ".join(f"{L.num_paren(k)}\\mathbf{{{nm}}}" for k, nm in zip(ks, names))
    parts = [" + ".join(f"{N(k)}\\cdot {N(v[i])}" for k, v in zip(ks, vs)) for i in range(n)]
    return build(
        type_="vec-lincomb", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính ${expr}$.", display=_vecs_disp(**dict(zip(names, vs)))),
        answer=vec_ans(r),
        hints=["Nhân từng vector với hệ số trước, rồi cộng các vector.", "Hoặc tính từng thành phần một."],
        steps=[componentwise(expr, parts, r)],
        verified=["sympy", "numpy", "shape"], tags=[f"R{n}", f"{terms}terms"],
    )


# ----------------------------------------------------------------- vec-norm
def gen_norm(rng: random.Random, level: int, seed: int) -> Exercise:
    nm = lambda p: f"\\norm{{\\mathbf{{u}}}}_{p}"
    if level == 1:  # chuẩn L1
        n = rng.choice([2, 3, 4])
        u = rand_vector(rng, n, -9, 9)
        r = sum(abs(x) for x in u)
        check(float(r) == float(np.linalg.norm(to_np(u), 1)), "L1 numpy")
        body = " + ".join(f"|{L.num(x)}|" for x in u)
        return build(
            type_="vec-norm", lesson=LESSON, level=1, seed=seed,
            prompt=Prompt(text=f"Tính chuẩn L1 ${nm(1)}$ (tổng các giá trị tuyệt đối).", display=_vecs_disp(u=u)),
            answer=num_ans(r),
            hints=["Chuẩn L1 là tổng giá trị tuyệt đối của các thành phần."],
            steps=[Step(text="Cộng giá trị tuyệt đối từng thành phần:", latex=f"{nm(1)} = {body} = {L.num(r)}")],
            verified=["sympy", "numpy"], tags=["l1", f"R{n}"],
        )
    n = rng.choice([2, 3]) if level in (2, 3, 4) else rng.choice([3, 4])
    u = int_norm_vector(rng, n, hi=12 if n < 4 else 8)
    s = sum(x * x for x in u)
    r = sqrt(s)
    check(r.is_Integer and float(r) == float(np.linalg.norm(to_np(u))), "L2 numpy")
    sq = " + ".join(f"{N(x)}^2" for x in u)
    if level == 2:
        return build(
            type_="vec-norm", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text=f"Tính chuẩn L2 ${nm(2)}$.", display=_vecs_disp(u=u)),
            answer=num_ans(r),
            hints=["$\\norm{\\mathbf{u}}_2 = \\sqrt{u_1^2 + u_2^2 + \\cdots}$.", "Tổng bình phương là số chính phương."],
            steps=[Step(text="Tính tổng bình phương rồi lấy căn:", latex=f"{nm(2)} = \\sqrt{{{sq}}} = \\sqrt{{{s}}} = {r}")],
            verified=["sympy", "numpy"], tags=["l2", f"R{n}"],
        )
    if level == 3:  # vector đơn vị
        unit = u / r
        check(sum(x * x for x in unit) == 1, "unit")
        np_close(unit, to_np(u) / np.linalg.norm(to_np(u)))
        return build(
            type_="vec-norm", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text="Tìm vector đơn vị cùng hướng với $\\mathbf{u}$, tức $\\mathbf{u}/\\norm{\\mathbf{u}}_2$.",
                          display=_vecs_disp(u=u)),
            answer=vec_ans(unit),
            hints=["Tính $\\norm{\\mathbf{u}}_2$ trước.", "Chia từng thành phần cho chuẩn."],
            steps=[Step(text="Chuẩn L2:", latex=f"{nm(2)} = \\sqrt{{{sq}}} = {r}"),
                   componentwise("\\frac{\\mathbf{u}}{\\norm{\\mathbf{u}}_2}", [f"\\frac{{{x}}}{{{r}}}" for x in u], unit,
                                 "Chia từng thành phần:")],
            verified=["sympy", "numpy"], tags=["unit", f"R{n}"],
        )
    if level == 4:  # ‖k u‖
        k = rng.choice([Rational(-3), Rational(-2), Rational(2), Rational(3), Rational(1, 2), Rational(-1, 2), Rational(3, 2)])
        res = abs(k) * r
        check(float(res) == float(np.linalg.norm(float(k) * to_np(u))), "knorm numpy")
        return build(
            type_="vec-norm", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"Tính $\\norm{{{L.num(k)}\\mathbf{{u}}}}_2$.", display=_vecs_disp(u=u)),
            answer=num_ans(res),
            hints=["Có thể dùng $\\norm{k\\mathbf{u}} = |k|\\,\\norm{\\mathbf{u}}$, hoặc nhân vector rồi tính chuẩn."],
            steps=[Step(text="Tính chuẩn của $\\mathbf{u}$:", latex=f"{nm(2)} = \\sqrt{{{sq}}} = {r}"),
                   Step(text="Dùng $\\norm{k\\mathbf{u}}_2 = |k|\\,\\norm{\\mathbf{u}}_2$:",
                        latex=f"|{L.num(k)}| \\cdot {r} = {L.num(res)}")],
            verified=["sympy", "numpy"], tags=["scaled", f"R{n}"],
        )
    # level 5: ML, hình phạt L2 ‖w‖² (không cần căn)
    w = rand_vector(rng, n, -4, 4, 0.25)
    res = sum(x * x for x in w)
    check(np.isclose(float(res), float(np.linalg.norm(to_np(w)) ** 2)), "l2 sq numpy")
    sq = " + ".join(f"{N(x)}^2" for x in w)
    return build(
        type_="vec-norm", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text="Vector trọng số $\\mathbf{w}$ của một mô hình có hình phạt L2 (regularization) bằng "
                           "$\\norm{\\mathbf{w}}_2^2$. Tính giá trị đó.", display=_vecs_disp(w=w)),
        answer=num_ans(res),
        hints=["$\\norm{\\mathbf{w}}_2^2$ là tổng bình phương các thành phần, không cần lấy căn."],
        steps=[Step(text="Cộng bình phương:", latex=f"\\norm{{\\mathbf{{w}}}}_2^2 = {sq} = {L.num(res)}")],
        verified=["sympy", "numpy"], tags=["ml", f"R{n}"],
    )


SPECS = [
    TopicSpec("vec-add", LESSON, (1, 2, 3), gen_add),
    TopicSpec("vec-scale", LESSON, (1, 2, 3), gen_scale),
    TopicSpec("vec-lincomb", LESSON, (2, 3, 4, 5), gen_lincomb),
    TopicSpec("vec-norm", LESSON, (1, 2, 3, 4, 5), gen_norm),
]
