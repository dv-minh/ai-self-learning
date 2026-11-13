"""Bài 1.13 Ma trận đối xứng và xác định dương: sym-eig, quad-form, posdef-check."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, eye, zeros

from ...core import latex as L
from ...core.gauss import eliminate
from ...core.nice import nontrivial_matrix, rand_vector
from ...core.verify import check, np_close, to_np
from ...schema import Answer, Check, Exercise, Prompt, Step
from ._base import TopicSpec, build, mat_ans, num_ans, size_tag, vec_ans
from .l1_12_eigen import _charpoly_steps, _verify_eigs
from .nice_sym import pick_sym

LESSON = "1.13"
B, N = L.bold, L.num_paren


# ------------------------------------------------------------------- sym-eig
def gen_sym_eig(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        a, eig = pick_sym(rng, 2, 8)
    else:
        a, eig = pick_sym(rng, 3, 3)
    n = a.rows
    roots = sorted(Rational(x) for x in eig)
    _verify_eigs(a, roots)
    check(a == a.T, "đối xứng")
    if level == 4:  # tìm một vector riêng
        j = rng.randrange(n)
        lam = roots[j]
        m = a - lam * eye(n)
        ns = m.nullspace()
        check(len(ns) >= 1, "có vector riêng")
        v = ns[0]
        # quy về nguyên
        den = 1
        for x in v:
            den = np.lcm(den, int(Rational(x).q))
        v = v * int(den)
        check(a * v == lam * v, "Av = λv")
        el = eliminate(m, mode="rref")
        return build(
            type_="sym-eig", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"$\\mathbf{{A}}$ là ma trận đối xứng và $\\lambda = {L.num(lam)}$ là một trị riêng. Tìm một vector riêng ứng với $\\lambda$ "
                               f"(mọi bội vô hướng khác $0$ đều được chấp nhận).", display=[L.mat(a, B("A"))]),
            answer=vec_ans(v), check=Check(mode="eigvec", a=L.entries(a), lam=L.rat_str(lam)),
            hints=["Giải $(\\mathbf{A} - \\lambda\\mathbf{I})\\mathbf{v} = \\mathbf{0}$.",
                   "Vì $\\mathbf{A}$ đối xứng, vector riêng của hai trị riêng khác nhau vuông góc: dùng để kiểm tra."],
            steps=[Step(text="Lập $\\mathbf{A} - \\lambda\\mathbf{I}$:", latex=L.mat(m))] + el.steps[1:] +
                  [Step(text="Một vector riêng:", latex=L.mat(v, "\\mathbf{v}"))],
            verified=["sympy", "numpy", "eigen-equation"], tags=[size_tag(a)],
        )
    return build(
        type_="sym-eig", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="$\\mathbf{A}$ là ma trận đối xứng. Liệt kê các trị riêng theo thứ tự tăng dần (lặp lại theo bội).",
                      display=[L.mat(a, B("A"))]),
        answer=vec_ans(Matrix(roots)),
        hints=["Giải $\\det(\\lambda\\mathbf{I} - \\mathbf{A}) = 0$.", "Ma trận đối xứng thực luôn có các trị riêng là số thực.",
               "Kiểm tra: tổng trị riêng bằng vết, tích bằng định thức."],
        steps=_charpoly_steps(a, roots), verified=["sympy", "numpy", "trace-det"], tags=[size_tag(a)] + (["repeated"] if len(set(roots)) < n else []),
    )


# ------------------------------------------------------------------ quad-form
def gen_quad_form(rng: random.Random, level: int, seed: int) -> Exercise:
    def quad(a: Matrix, x: Matrix):
        v = (x.T * a * x)[0, 0]
        check(np.isclose(float(v), float((to_np(x).T @ to_np(a) @ to_np(x))[0, 0])), "xᵀAx numpy")
        return v

    if level in (2, 3):
        n = 2 if level == 2 else 3
        a = nontrivial_matrix(rng, n, n, -4, 4)
        a = (a + a.T)  # đối xứng, phần tử nguyên chẵn ở đường chéo
        a = a / 2 if all(x % 2 == 0 for x in a) else a
        x = rand_vector(rng, n, -3, 3)
        v = quad(a, x)
        ax = a * x
        return build(
            type_="quad-form", lesson=LESSON, level=level, seed=seed,
            prompt=Prompt(text="Tính dạng toàn phương $q(\\mathbf{x}) = \\mathbf{x}\\T\\mathbf{A}\\mathbf{x}$.", display=[L.mat(a, B("A")), L.mat(x, B("x"))]),
            answer=num_ans(v),
            hints=["Tính $\\mathbf{A}\\mathbf{x}$ trước (một vector), rồi $\\mathbf{x}\\T(\\mathbf{A}\\mathbf{x})$ (tích vô hướng)."],
            steps=[Step(text="$\\mathbf{A}\\mathbf{x}$:", latex=L.mat(ax, "\\mathbf{A}\\mathbf{x}")),
                   Step(text="$\\mathbf{x}\\T(\\mathbf{A}\\mathbf{x})$:", latex="\\mathbf{x}\\T\\mathbf{A}\\mathbf{x} = " + " + ".join(f"{N(p)}\\cdot {N(q)}" for p, q in zip(x, ax)) + f" = {L.num(v)}")],
            verified=["sympy", "numpy"], tags=[f"{n}x{n}"],
        )
    if level == 4:  # ma trận đối xứng của cùng dạng toàn phương
        a = nontrivial_matrix(rng, 2, 2, -5, 5)
        while a[0, 1] == a[1, 0]:
            a = nontrivial_matrix(rng, 2, 2, -5, 5)
        s = (a + a.T) / 2
        for _ in range(5):
            x = rand_vector(rng, 2, -4, 4)
            check(quad(a, x) == quad(s, x), "xᵀAx = xᵀSx")
        return build(
            type_="quad-form", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Dạng toàn phương $q(\\mathbf{x}) = \\mathbf{x}\\T\\mathbf{A}\\mathbf{x}$ không đổi nếu thay $\\mathbf{A}$ bằng một ma trận **đối xứng** "
                               "$\\mathbf{S}$. Tìm $\\mathbf{S}$ (gợi ý: $\\mathbf{S} = \\frac{1}{2}(\\mathbf{A} + \\mathbf{A}\\T)$).", display=[L.mat(a, B("A"))]),
            answer=mat_ans(s),
            hints=["Phần đối xứng của $\\mathbf{A}$ là $\\frac{1}{2}(\\mathbf{A}+\\mathbf{A}\\T)$.", "Phần phản đối xứng không đóng góp vào dạng toàn phương."],
            steps=[Step(text="$\\mathbf{A} + \\mathbf{A}\\T$:", latex=L.mat(a + a.T)),
                   Step(text="Chia đôi:", latex=L.mat(s, "\\mathbf{S}"))],
            verified=["sympy", "numpy", "substitution"], tags=["2x2", "symmetrize"],
        )
    # level 5: phương sai theo hướng w^T Σ w
    pairs = [(3, 4), (4, 3), (5, 12), (12, 5)]
    for _ in range(100):
        a, _eig = pick_sym(rng, 2, 6)
        if all(x >= 0 for x in a.diagonal()):
            break
    p, q = rng.choice(pairs)
    h = Rational(p * p + q * q) ** Rational(1, 2)
    w = Matrix([Rational(p) / h, Rational(q) / h]) * rng.choice([1, -1])
    check((w.T * w)[0, 0] == 1, "w đơn vị")
    v = quad(a, w)
    return build(
        type_="quad-form", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text="Ma trận hiệp phương sai $\\boldsymbol{\\Sigma}$ của dữ liệu hai chiều và một hướng đơn vị $\\mathbf{w}$. "
                           "Phương sai của dữ liệu khi chiếu lên hướng $\\mathbf{w}$ là $\\mathbf{w}\\T\\boldsymbol{\\Sigma}\\mathbf{w}$. Tính giá trị đó.",
                      display=[L.mat(a, "\\boldsymbol{\\Sigma}"), L.mat(w, B("w"))]),
        answer=num_ans(v),
        hints=["Tính $\\boldsymbol{\\Sigma}\\mathbf{w}$ rồi lấy tích vô hướng với $\\mathbf{w}$."],
        steps=[Step(text="$\\boldsymbol{\\Sigma}\\mathbf{w}$:", latex=L.mat(a * w)),
               Step(text="Tích vô hướng với $\\mathbf{w}$:", latex=f"\\mathbf{{w}}\\T\\boldsymbol{{\\Sigma}}\\mathbf{{w}} = {L.num(v)}")],
        verified=["sympy", "numpy"], tags=["ml", "2x2"],
    )


# --------------------------------------------------------------- posdef-check
CLASSES = ["xác định dương", "bán xác định dương nhưng không xác định dương", "xác định âm", "không xác định (có cả hướng dương và hướng âm)"]
ML_CLASSES = ["cực tiểu địa phương", "điểm yên ngựa", "cực đại địa phương", "chưa kết luận được (Hessian bán xác định)"]


def _class_of(eig: list[int]) -> int:
    if all(e > 0 for e in eig):
        return 0
    if all(e >= 0 for e in eig):
        return 1
    if all(e < 0 for e in eig):
        return 2
    if any(e > 0 for e in eig) and any(e < 0 for e in eig):
        return 3
    return -1  # bán xác định âm: không dùng


def gen_posdef(rng: random.Random, level: int, seed: int) -> Exercise:
    n = 2 if level in (2, 5) else 3
    target = rng.randrange(4)
    r = 8 if n == 2 else 3
    a = eig = None
    for _ in range(200):
        a, eig = pick_sym(rng, n, r)
        if _class_of(eig) == target:
            break
    else:
        raise RuntimeError("posdef: không tìm được lớp mong muốn")
    cls = _class_of(eig)
    np_close(Matrix(eig), np.linalg.eigvalsh(to_np(a)))
    roots = sorted(Rational(x) for x in eig)
    minors = [a[:k, :k].det() for k in range(1, n + 1)]
    steps = _charpoly_steps(a, roots)
    if cls == 0:
        steps.append(Step(text="Kiểm tra bằng các định thức con chính (tiêu chuẩn Sylvester):", latex=", ".join(f"\\Delta_{{{k + 1}}} = {L.num(m)}" for k, m in enumerate(minors))))
        steps.append(Step(text="Mọi trị riêng dương (và mọi $\\Delta_k > 0$): xác định dương."))
    else:
        steps.append(Step(text="Dấu các trị riêng quyết định loại: " + {1: "không âm và có trị riêng bằng $0$", 2: "tất cả âm", 3: "có trị riêng dương lẫn âm"}[cls] + "."))
    if level == 5:
        labels = ML_CLASSES
        mapping = {0: 0, 1: 3, 2: 2, 3: 1}
        correct = mapping[cls]
        return build(
            type_="posdef-check", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text="Hàm mất mát $f$ có gradient bằng $\\mathbf{0}$ tại điểm $\\mathbf{x}_0$ và Hessian (ma trận đạo hàm bậc hai, đối xứng) tại đó là $\\mathbf{H}$. "
                               "Điểm $\\mathbf{x}_0$ là gì?", display=[L.mat(a, B("H"))]),
            answer=Answer(kind="choice", correct=[correct]), choices=labels,
            hints=["$\\mathbf{H}$ xác định dương: cực tiểu. Xác định âm: cực đại. Không xác định: yên ngựa.", "Xét dấu các trị riêng của $\\mathbf{H}$."],
            steps=steps, verified=["sympy", "numpy"], tags=["ml", size_tag(a)],
        )
    return build(
        type_="posdef-check", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Ma trận đối xứng $\\mathbf{A}$ thuộc loại nào?", display=[L.mat(a, B("A"))]),
        answer=Answer(kind="choice", correct=[cls]), choices=CLASSES,
        hints=["Xét dấu các trị riêng: tất cả dương thì xác định dương.", "Với $\\mathbf{A}$ đối xứng, dạng toàn phương $\\mathbf{x}\\T\\mathbf{A}\\mathbf{x}$ cùng dấu với các trị riêng theo các hướng riêng."],
        steps=steps, verified=["sympy", "numpy", "sylvester"], tags=[size_tag(a), ["pd", "psd", "nd", "indef"][cls]],
    )


SPECS = [
    TopicSpec("sym-eig", LESSON, (2, 3, 4), gen_sym_eig),
    TopicSpec("quad-form", LESSON, (2, 3, 4, 5), gen_quad_form),
    TopicSpec("posdef-check", LESSON, (2, 3, 4, 5), gen_posdef),
]
