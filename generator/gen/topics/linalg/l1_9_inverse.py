"""Bài 1.9 Ma trận nghịch đảo: inv-check, inv-2x2, inv-gauss-jordan, inv-solve."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, eye

from ...core import latex as L
from ...core.gauss import eliminate
from ...core.nice import nontrivial_matrix
from ...core.steps import matmul_steps
from ...core.verify import check, np_close, to_np, verify_det, verify_inverse, verify_matmul, verify_solution
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, bool_ans, build, mat_ans, size_tag, vec_ans
from .l1_7_gauss import _elementary_P

LESSON = "1.9"
B = L.bold
N = L.num_paren


def _singular(rng: random.Random, n: int, lo: int, hi: int) -> Matrix:
    """Ma trận suy biến: một hàng là tổ hợp của các hàng khác (và không có hàng 0)."""
    for _ in range(500):
        a = nontrivial_matrix(rng, n, n, lo, hi)
        i = rng.randrange(n)
        others = [r for r in range(n) if r != i]
        k = [rng.choice([-2, -1, 1, 2]) for _ in others]
        a[i, :] = sum((k_r * a[r, :] for k_r, r in zip(k, others)), Matrix.zeros(1, n))
        if a.det() == 0 and not any(all(x == 0 for x in a[r, :]) for r in range(n)):
            return a
    raise RuntimeError("singular")


def _unimodular(rng: random.Random, n: int, bound: int, ops: int | None = None) -> Matrix:
    for _ in range(500):
        a = _elementary_P(rng, n, ops or rng.randint(3, 5)) * Matrix(n, n, lambda i, j: 1 if i == j else 0)
        if max(abs(x) for x in a) <= bound and a != eye(n):
            return a
    raise RuntimeError("unimodular")


def _invertible(rng: random.Random, n: int, lo: int, hi: int) -> Matrix:
    for _ in range(500):
        a = nontrivial_matrix(rng, n, n, lo, hi)
        if a.det() != 0:
            return a
    raise RuntimeError("invertible")


# ---------------------------------------------------------------- inv-check
def gen_check(rng: random.Random, level: int, seed: int) -> Exercise:
    if level in (1, 2):
        n = 2 if level == 1 else 3
        want_sing = rng.random() < 0.5
        a = _singular(rng, n, -6, 6) if want_sing else _invertible(rng, n, -6, 6)
        d = a.det()
        verified = verify_det(a, d)
        rk = a.rank()
        return build(
            type_="inv-check", lesson=LESSON, level=level, seed=seed,
            prompt=Prompt(text="Ma trận $\\mathbf{A}$ có khả nghịch (invertible) không?", display=[L.mat(a, B("A"))]),
            answer=bool_ans(d != 0),
            hints=["Ma trận vuông khả nghịch khi và chỉ khi định thức khác $0$."],
            steps=[Step(text="Tính định thức:", latex=f"\\det(\\mathbf{{A}}) = {L.num(d)}")] +
                  det_steps_short(a, d) +
                  [Step(text="Định thức khác $0$ nên khả nghịch." if d != 0 else "Định thức bằng $0$ nên không khả nghịch (suy biến).")],
            verified=verified, tags=[size_tag(a), "singular" if d == 0 else "invertible"],
        )
    if level == 4:  # chọn các ma trận khả nghịch
        mats = []
        for _ in range(4):
            mats.append(_singular(rng, 3, -4, 4) if rng.random() < 0.45 else _invertible(rng, 3, -4, 4))
        dets = [m.det() for m in mats]
        for m, d in zip(mats, dets):
            check(np.isclose(np.linalg.det(to_np(m)), float(d), atol=1e-8), "numpy det")
        correct = [i for i, d in enumerate(dets) if d != 0]
        if not correct:  # tránh đáp án rỗng
            return gen_check(rng, level, seed)
        names = ["A", "B", "C", "D"]
        return build(
            type_="inv-check", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Chọn tất cả các ma trận khả nghịch.", display=[L.mat(m, B(nm)) for m, nm in zip(mats, names)]),
            answer=Answer(kind="multi", correct=correct), choices=[f"${B(nm)}$" for nm in names],
            hints=["Tính định thức từng ma trận (Sarrus). Khác $0$ thì khả nghịch."],
            steps=[Step(text=f"${B(nm)}$:", latex=f"\\det({B(nm)}) = {L.num(d)}") for nm, d in zip(names, dets)],
            verified=["sympy", "numpy", "laplace"], tags=["3x3", "multi"],
        )
    # level 5: X^T X có khả nghịch không?
    r, c = rng.choice([(3, 2), (4, 2), (4, 3), (3, 3)])
    for _ in range(300):
        x = nontrivial_matrix(rng, r, c, -3, 3)
        if rng.random() < 0.5 and c >= 2:
            k = rng.choice([-2, -1, 1, 2])
            x[:, c - 1] = k * x[:, 0]
            if any(all(v == 0 for v in x[:, j]) for j in range(c)):
                continue
        break
    g = x.T * x
    d = g.det()
    verify_det(g, d)
    inv_ok = d != 0
    check(inv_ok == (x.rank() == c), "Gram khả nghịch ⇔ cột độc lập")
    return build(
        type_="inv-check", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text="Trong hồi quy tuyến tính bằng phương trình chuẩn (normal equation), ta cần nghịch đảo $\\mathbf{X}\\T\\mathbf{X}$ "
                           "với $\\mathbf{X}$ là ma trận dữ liệu (mỗi hàng là một mẫu). Khẳng định \"$\\mathbf{X}\\T\\mathbf{X}$ khả nghịch\" đúng hay sai?",
                      display=[L.mat(x, B("X"))]),
        answer=bool_ans(inv_ok),
        hints=["$\\mathbf{X}\\T\\mathbf{X}$ khả nghịch khi và chỉ khi các cột của $\\mathbf{X}$ độc lập tuyến tính.",
               "Hoặc tính trực tiếp $\\det(\\mathbf{X}\\T\\mathbf{X})$."],
        steps=[Step(text="Tính ma trận Gram:", latex=L.mat(g, "\\mathbf{X}\\T\\mathbf{X}")),
               Step(text="Định thức:", latex=f"\\det(\\mathbf{{X}}\\T\\mathbf{{X}}) = {L.num(d)}"),
               Step(text="Khác $0$: khả nghịch." if inv_ok else "Bằng $0$: không khả nghịch (các cột của $\\mathbf{X}$ phụ thuộc tuyến tính).")],
        verified=["sympy", "numpy", "rank"], tags=["ml", "invertible" if inv_ok else "singular"],
    )


def det_steps_short(a: Matrix, d) -> list[Step]:
    from .l1_8_det import det_steps

    return det_steps(a, d)[:1] if a.rows <= 3 else []


# ----------------------------------------------------------------- inv-2x2
def gen_inv2(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 4:  # (AB)^-1 = B^-1 A^-1
        for _ in range(300):
            a, b = _unimodular(rng, 2, 6, 3), _unimodular(rng, 2, 6, 3)
            if a != eye(2) and b != eye(2):
                break
        c = a * b
        inv = c.inv()
        verify_inverse(c, inv)
        check(inv == b.inv() * a.inv(), "(AB)^-1 = B^-1 A^-1")
        return build(
            type_="inv-2x2", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Tính $(\\mathbf{A}\\mathbf{B})^{-1}$.", display=[L.mat(a, B("A")), L.mat(b, B("B"))]),
            answer=mat_ans(inv),
            hints=["$(\\mathbf{AB})^{-1} = \\mathbf{B}^{-1}\\mathbf{A}^{-1}$ (thứ tự đảo), hoặc tính $\\mathbf{AB}$ rồi nghịch đảo.",
                   "Nghịch đảo $2\\times 2$: $\\frac{1}{ad-bc}\\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}$."],
            steps=[Step(text="Nhân $\\mathbf{AB}$:", latex=L.mat(c, "\\mathbf{AB}")),
                   Step(text="Định thức:", latex=f"\\det(\\mathbf{{AB}}) = {L.num(c.det())}"),
                   Step(text="Áp dụng công thức nghịch đảo $2\\times 2$:", latex=f"(\\mathbf{{AB}})^{{-1}} = \\frac{{1}}{{{N(c.det())}}}{L.expr_mat([[L.num(c[1, 1]), L.num(-c[0, 1])], [L.num(-c[1, 0]), L.num(c[0, 0])]])} = {L.mat(inv)}")],
            verified=["sympy", "numpy", "identity"], tags=["2x2", "product"],
        )
    if level == 2:
        a = _unimodular(rng, 2, 8, 3)
    else:
        a = _invertible(rng, 2, -6, 6)
    d = a.det()
    inv = a.inv()
    verified = verify_inverse(a, inv)
    return build(
        type_="inv-2x2", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tính $\\mathbf{A}^{-1}$ bằng công thức nghịch đảo của ma trận $2\\times 2$.", display=[L.mat(a, B("A"))]),
        answer=mat_ans(inv),
        hints=["Công thức: $\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}^{-1} = \\frac{1}{ad-bc}\\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}$.",
               "Tính $ad - bc$ trước; nó phải khác $0$."],
        steps=[Step(text="Định thức:", latex=f"\\det(\\mathbf{{A}}) = {det2(a)} = {L.num(d)}"),
               Step(text="Hoán vị đường chéo chính, đổi dấu đường chéo phụ, rồi chia cho định thức:",
                    latex=f"\\mathbf{{A}}^{{-1}} = \\frac{{1}}{{{N(d)}}}{L.expr_mat([[L.num(a[1, 1]), L.num(-a[0, 1])], [L.num(-a[1, 0]), L.num(a[0, 0])]])} = {L.mat(inv)}"),
               Step(text="Kiểm tra $\\mathbf{A}\\mathbf{A}^{-1} = \\mathbf{I}$ (tùy chọn):", latex=L.mat(a * inv, "\\mathbf{A}\\mathbf{A}^{-1}"))],
        verified=verified, tags=["2x2", "int-inverse" if all(x.is_integer for x in inv) else "frac-inverse"],
    )


def det2(m: Matrix) -> str:
    return f"{N(m[0, 0])}\\cdot {N(m[1, 1])} - {N(m[0, 1])}\\cdot {N(m[1, 0])}"


# --------------------------------------------------------- inv-gauss-jordan
def gen_inv_gj(rng: random.Random, level: int, seed: int) -> Exercise:
    n = {2: 2, 3: 3, 4: 4}[level]
    a = _unimodular(rng, n, 8 if n < 4 else 10, 3 if n < 4 else 5)
    inv = a.inv()
    check(all(x.is_integer for x in inv), "nghịch đảo nguyên")
    verified = verify_inverse(a, inv)
    el = eliminate(a.row_join(eye(n)), aug=n, mode="rref")
    check(el.matrix[:, n:] == inv and el.matrix[:, :n] == eye(n), "Gauss–Jordan [A|I] → [I|A^-1]")
    return build(
        type_="inv-gauss-jordan", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tìm $\\mathbf{A}^{-1}$ bằng khử Gauss–Jordan trên $[\\mathbf{A}\\,|\\,\\mathbf{I}]$. "
                           "(Nghịch đảo của ma trận này có toàn số nguyên.)", display=[L.mat(a, B("A"))]),
        answer=mat_ans(inv),
        hints=["Lập ma trận $[\\mathbf{A}\\,|\\,\\mathbf{I}]$ và đưa vế trái về $\\mathbf{I}$ bằng phép biến đổi hàng.",
               "Khi vế trái thành $\\mathbf{I}$, vế phải chính là $\\mathbf{A}^{-1}$."],
        steps=el.steps + [Step(text="Vế phải là nghịch đảo:", latex=L.mat(inv, "\\mathbf{A}^{-1}"))],
        verified=verified, tags=[size_tag(a)],
    )


# ----------------------------------------------------------------- inv-solve
def gen_inv_solve(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:  # cho sẵn A^-1
        n = 2
        a = _unimodular(rng, n, 7, 3)
        inv = a.inv()
        x = Matrix([rng.randint(-4, 4) for _ in range(n)])
        b = a * x
        verify_solution(a, b, x)
        steps = [Step(text="Nghiệm là $\\mathbf{x} = \\mathbf{A}^{-1}\\mathbf{b}$:")] + matmul_steps(inv, b, "\\mathbf{A}^{-1}", "\\mathbf{b}")[1:]
        check(inv * b == x, "x = A^-1 b")
        return build(
            type_="inv-solve", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text="Cho hệ $\\mathbf{A}\\mathbf{x} = \\mathbf{b}$ và biết $\\mathbf{A}^{-1}$. Tính $\\mathbf{x}$.",
                          display=[L.mat(inv, "\\mathbf{A}^{-1}"), L.mat(b, B("b"))]),
            answer=vec_ans(x),
            hints=["Nhân hai vế bên trái với $\\mathbf{A}^{-1}$: $\\mathbf{x} = \\mathbf{A}^{-1}\\mathbf{b}$."],
            steps=steps, verified=["sympy", "numpy", "substitution"], tags=["2x2", "given-inverse"],
        )
    n = 2 if level == 3 else 3
    a = _invertible(rng, 2, -5, 5) if n == 2 else _unimodular(rng, 3, 8, 4)
    inv = a.inv()
    x = Matrix([rng.randint(-4, 4) for _ in range(n)])
    b = a * x
    verify_solution(a, b, x)
    verified = verify_inverse(a, inv)
    check(inv * b == x, "x = A^-1 b")
    np_close(x, np.linalg.solve(to_np(a), to_np(b)))
    return build(
        type_="inv-solve", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Giải hệ $\\mathbf{A}\\mathbf{x} = \\mathbf{b}$ bằng cách tính $\\mathbf{A}^{-1}$ rồi $\\mathbf{x} = \\mathbf{A}^{-1}\\mathbf{b}$.",
                      display=[L.mat(a, B("A")), L.mat(b, B("b"))]),
        answer=vec_ans(x),
        hints=["Tính $\\mathbf{A}^{-1}$ trước (công thức $2\\times 2$ hoặc Gauss–Jordan).", "Rồi nhân $\\mathbf{A}^{-1}$ với $\\mathbf{b}$."],
        steps=[Step(text="Nghịch đảo:", latex=L.mat(inv, "\\mathbf{A}^{-1}"))] +
              [Step(text="Nhân với $\\mathbf{b}$:")] + matmul_steps(inv, b, "\\mathbf{A}^{-1}", "\\mathbf{b}")[1:],
        verified=verified, tags=[size_tag(a)],
    )


SPECS = [
    TopicSpec("inv-check", LESSON, (1, 2, 4, 5), gen_check),
    TopicSpec("inv-2x2", LESSON, (2, 3, 4), gen_inv2),
    TopicSpec("inv-gauss-jordan", LESSON, (2, 3, 4), gen_inv_gj),
    TopicSpec("inv-solve", LESSON, (2, 3, 4), gen_inv_solve),
]
