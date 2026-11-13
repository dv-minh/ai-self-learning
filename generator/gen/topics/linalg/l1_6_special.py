"""Bài 1.6 Ma trận đặc biệt: mat-identify, mat-power-diag, transpose-product."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, diag, eye

from ...core import latex as L
from ...core.nice import is_trivial, nontrivial_matrix
from ...core.steps import matmul_steps
from ...core.verify import check, np_close, to_np, verify_matmul
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, build, mat_ans, shape_answer, shape_tex, size_tag

LESSON = "1.6"
B = L.bold

PROPS = [
    "ma trận đơn vị",
    "ma trận đường chéo",
    "ma trận tam giác trên",
    "ma trận tam giác dưới",
    "ma trận đối xứng",
    "ma trận trực giao",
]
NONE_LABEL = "không loại nào ở trên"


def props_of(m: Matrix) -> list[bool]:
    n = m.rows
    if m.rows != m.cols:
        return [False] * 6
    up = all(m[i, j] == 0 for i in range(n) for j in range(i))
    lo = all(m[i, j] == 0 for i in range(n) for j in range(i + 1, n))
    # trực giao: kiểm bằng SymPy chính xác và bằng NumPy
    orth = m.T * m == eye(n)
    check(orth == bool(np.allclose(to_np(m).T @ to_np(m), np.eye(n))), "orthogonal sympy/numpy")
    return [m == eye(n), up and lo, up, lo, m == m.T, orth]


def _build_special(rng: random.Random, kind: str, n: int) -> Matrix:
    nz = [x for x in range(-5, 6) if x]
    if kind == "identity":
        return eye(n)
    if kind == "diagonal":
        return diag(*[rng.choice(nz + [0]) for _ in range(n)])
    if kind == "upper":
        return Matrix(n, n, lambda i, j: rng.randint(-5, 5) if j >= i else 0)
    if kind == "lower":
        return Matrix(n, n, lambda i, j: rng.randint(-5, 5) if j <= i else 0)
    if kind == "symmetric":
        m = Matrix(n, n, lambda i, j: 0)
        for i in range(n):
            for j in range(i, n):
                m[i, j] = m[j, i] = rng.randint(-5, 5)
        return m
    if kind == "orthogonal":
        if n == 2 and rng.random() < 0.5:  # quay với cos, sin hữu tỉ
            c, s = rng.choice([(3, 4), (4, 3), (5, 12), (12, 5)]), None
            a, b = c
            h = Rational(a * a + b * b) ** Rational(1, 2)
            q = Matrix([[a, -b], [b, a]]) / h
            return q if rng.random() < 0.5 else q.T
        perm = list(range(n))
        rng.shuffle(perm)
        return Matrix(n, n, lambda i, j: rng.choice([-1, 1]) if perm[i] == j else 0)
    return nontrivial_matrix(rng, n, n, -5, 5)  # generic


def gen_identify(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 5:
        r, c = rng.choice([(3, 2), (4, 2), (4, 3), (3, 3), (2, 2)])
        x = nontrivial_matrix(rng, r, c, -3, 3)
        g = x.T * x
        verify_matmul(x.T, x, g)
        a = g
        stmt = f"Cho ma trận dữ liệu $\\mathbf{{X}}$ (mỗi hàng là một mẫu). Ma trận Gram $\\mathbf{{X}}\\T\\mathbf{{X}}$ thuộc những loại nào dưới đây?"
        display = [L.mat(x, B("X"))]
        pre = [Step(text="Tính $\\mathbf{X}\\T\\mathbf{X}$ (kích thước " + f"${c}\\times {c}$):", latex=L.mat(g, "\\mathbf{X}\\T\\mathbf{X}"))]
    else:
        n = rng.choice([2, 3]) if level == 1 else rng.choice([3, 4])
        kind = rng.choice(["identity", "diagonal", "upper", "lower", "symmetric", "orthogonal", "generic", "generic"])
        a = _build_special(rng, kind, n)
        if is_trivial(a) and kind != "identity":
            a = _build_special(rng, "generic", n)
        stmt = "Ma trận vuông $\\mathbf{A}$ thuộc những loại nào dưới đây? Chọn tất cả loại đúng."
        display = [L.mat(a, B("A"))]
        pre = []
    flags = props_of(a)
    choices = PROPS + [NONE_LABEL]
    correct = [i for i, f in enumerate(flags) if f] or [len(PROPS)]
    steps = pre
    defs = [
        "đơn vị: đường chéo toàn $1$, còn lại toàn $0$",
        "đường chéo: mọi phần tử ngoài đường chéo bằng $0$",
        "tam giác trên: mọi phần tử dưới đường chéo bằng $0$",
        "tam giác dưới: mọi phần tử trên đường chéo bằng $0$",
        "đối xứng: $\\mathbf{A} = \\mathbf{A}\\T$",
        "trực giao: $\\mathbf{A}\\T\\mathbf{A} = \\mathbf{I}$",
    ]
    for d, f in zip(defs, flags):
        steps.append(Step(text=f"Kiểm tra {d}: " + ("thỏa." if f else "không thỏa.")))
    return build(
        type_="mat-identify", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=stmt, display=display),
        answer=Answer(kind="multi", correct=correct), choices=choices,
        hints=["Mỗi loại có một điều kiện riêng; kiểm từng điều kiện.",
               "Ma trận đường chéo vừa là tam giác trên vừa là tam giác dưới.",
               "Trực giao: các cột có chuẩn $1$ và vuông góc đôi một, tức $\\mathbf{A}\\T\\mathbf{A}=\\mathbf{I}$."],
        steps=steps, verified=["sympy", "numpy"], tags=[size_tag(a), "ml" if level == 5 else "classify"],
    )


# ------------------------------------------------------------ mat-power-diag
def gen_power_diag(rng: random.Random, level: int, seed: int) -> Exercise:
    n = {2: 2, 3: 3, 4: 4}[level]
    pool = [-3, -2, -1, 1, 2, 3] if level < 4 else [Rational(-2), Rational(-1), Rational(1, 2), Rational(1), Rational(2), Rational(3), Rational(-1, 2)]
    entries = [rng.choice(pool + ([0] if level > 2 else [])) for _ in range(n)]
    if all(e == 0 for e in entries):
        entries[0] = Rational(2)
    k = rng.randint(2, 3 if level == 2 else (4 if level == 3 else 5))
    d = diag(*entries)
    res = d ** k
    check(res == diag(*[e ** k for e in entries]), "diag power formula")
    np_close(res, np.linalg.matrix_power(to_np(d), k))
    return build(
        type_="mat-power-diag", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính $\\mathbf{{D}}^{{{k}}}$.", display=[L.mat(d, B("D"))]),
        answer=mat_ans(res),
        hints=["Với ma trận đường chéo, lũy thừa chỉ cần lũy thừa từng phần tử trên đường chéo.",
               "Các phần tử ngoài đường chéo vẫn bằng $0$."],
        steps=[Step(text=f"Lũy thừa ${k}$ từng phần tử trên đường chéo:",
                    latex="\\diag(" + ", ".join(f"{L.num_paren(e)}^{{{k}}}" for e in entries) + ") = " + L.mat(res))],
        verified=["sympy", "numpy"], tags=[size_tag(d), f"k={k}"],
    )


# ---------------------------------------------------------- transpose-product
def gen_transpose_product(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 4:
        k = rng.choice([3, 3, 4])
        names = rng.sample(list("ABCDMNPQRSTUVWXYZ"), k)
        mb = lambda x: "\\mathbf{" + x + "}"
        tr = lambda x: mb(x) + "\\T"
        fwd = names
        rev = names[::-1]
        swapped = rev[:]
        i = rng.randrange(k - 1)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        right = "".join(tr(x) for x in rev)
        pool = {
            right: True,
            "".join(tr(x) for x in fwd): False,
            "".join(mb(x) for x in fwd): False,
            "".join(mb(x) for x in rev): False,
            "".join(tr(x) for x in swapped): False,
        }
        keys = list(pool)
        rng.shuffle(keys)
        choices = [f"${key}$" for key in keys]
        correct = [i for i, key in enumerate(keys) if pool[key]]
        # kiểm chứng số trên ma trận vuông ngẫu nhiên
        ms = [nontrivial_matrix(rng, 2, 2, -3, 3) for _ in range(k)]
        prod = ms[0]
        for m in ms[1:]:
            prod = prod * m
        back = ms[-1].T
        for m in ms[-2::-1]:
            back = back * m.T
        check(prod.T == back, "(tích)^T = tích đảo của chuyển vị")
        np_close(prod.T, np.linalg.multi_dot([to_np(m) for m in ms]).T)
        prod_expr = "".join(mb(x) for x in fwd)
        return build(
            type_="transpose-product", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"Biểu thức nào bằng $({prod_expr})\\T$ (các kích thước đều phù hợp)?"),
            answer=Answer(kind="choice", correct=correct), choices=choices,
            hints=["Chuyển vị một tích thì đảo thứ tự các thừa số và chuyển vị từng thừa số.",
                   "Áp dụng $(\\mathbf{X}\\mathbf{Y})\\T = \\mathbf{Y}\\T\\mathbf{X}\\T$ liên tiếp."],
            steps=[Step(text="Đảo thứ tự các thừa số và chuyển vị từng thừa số:",
                        latex=f"({prod_expr})\\T = {right}")],
            verified=["sympy", "numpy"], tags=["choice", f"{k}factors"],
        )
    if level == 5:
        n, d, h = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)
        check((np.zeros((n, d)) @ np.zeros((d, h))).T.shape == (h, n), "(XW)^T shape")
        return build(
            type_="transpose-product", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text=f"Đầu ra của một lớp tuyến tính là $\\mathbf{{Y}} = \\mathbf{{X}}\\mathbf{{W}}$ với $\\mathbf{{X}}$ cỡ ${n}\\times {d}$ "
                               f"và $\\mathbf{{W}}$ cỡ ${d}\\times {h}$. Biểu thức $\\mathbf{{W}}\\T\\mathbf{{X}}\\T$ có kích thước bao nhiêu?"),
            answer=shape_answer((h, n)),
            hints=["Theo tính chất chuyển vị của tích, $\\mathbf{W}\\T\\mathbf{X}\\T = (\\mathbf{X}\\mathbf{W})\\T$."],
            steps=[Step(text="$\\mathbf{W}\\T$ cỡ " + f"${h}\\times {d}$, $\\mathbf{{X}}\\T$ cỡ ${d}\\times {n}$:", latex=f"({h}\\times {d})({d}\\times {n})"),
                   Step(text="Kích thước:", latex=shape_tex((h, n)))],
            verified=["numpy", "shape"], tags=["ml"],
        )
    if level == 2:
        m, n, p = rng.choice([(2, 2, 2), (2, 3, 2), (3, 2, 2), (2, 2, 3)])
        lo, hi = -4, 4
    else:
        m, n, p = rng.choice([(3, 3, 3), (4, 3, 2), (2, 4, 3), (3, 4, 4)])
        lo, hi = -6, 6
    a, b = nontrivial_matrix(rng, m, n, lo, hi), nontrivial_matrix(rng, n, p, lo, hi)
    c = a * b
    verify_matmul(a, b, c)
    check((c.T == b.T * a.T), "(AB)^T = B^T A^T")
    np_close(c.T, (to_np(a) @ to_np(b)).T)
    target = "(\\mathbf{AB})\\T" if level == 2 else "\\mathbf{B}\\T\\mathbf{A}\\T"
    steps = [Step(text="Nhân $\\mathbf{A}\\mathbf{B}$:", latex=L.mat(c, "\\mathbf{AB}")),
             Step(text="Chuyển vị (hàng thành cột):", latex=L.mat(c.T, "(\\mathbf{AB})\\T"))]
    if level == 3:
        steps = [Step(text="Chuyển vị từng ma trận:", latex=f"{L.mat(b.T, B('B') + chr(92) + 'T')},\\quad {L.mat(a.T, B('A') + chr(92) + 'T')}")] + \
                matmul_steps(b.T, a.T, "\\mathbf{B}\\T", "\\mathbf{A}\\T")[1:]
    return build(
        type_="transpose-product", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính ${target}$.", display=[L.mat(a, B("A")), L.mat(b, B("B"))]),
        answer=mat_ans(c.T),
        hints=["$(\\mathbf{AB})\\T = \\mathbf{B}\\T\\mathbf{A}\\T$, thứ tự bị đảo.", f"Kết quả có kích thước ${p}\\times {m}$."],
        steps=steps, verified=["sympy", "numpy", "shape"], tags=[size_tag(a), size_tag(b)],
    )


SPECS = [
    TopicSpec("mat-identify", LESSON, (1, 2, 5), gen_identify),
    TopicSpec("mat-power-diag", LESSON, (2, 3, 4), gen_power_diag),
    TopicSpec("transpose-product", LESSON, (2, 3, 4, 5), gen_transpose_product),
]
