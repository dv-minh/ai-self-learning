"""Bài 1.12 Trị riêng, vector riêng: eig-values, eig-vectors, diagonalize, power-via-diag."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, Symbol, diag, eye, factor, latex, zeros

from ...core import latex as L
from ...core.gauss import eliminate
from ...core.nice import nontrivial_matrix
from ...core.verify import check, np_close, to_np, verify_matmul
from ...schema import Answer, Check, Exercise, Prompt, Step
from ._base import TopicSpec, bool_ans, build, mat_ans, size_tag, vec_ans
from .l1_9_inverse import _unimodular
from .nice_orth import cayley

LESSON = "1.12"
B, N = L.bold, L.num_paren
LAM = Symbol("\\lambda")


def _diag_vals(rng: random.Random, n: int, distinct: bool = True, pool=(-3, -2, -1, 1, 2, 3, 4), allow_zero: bool = False) -> list[int]:
    vals = list(pool) + ([0] if allow_zero else [])
    if distinct:
        return rng.sample(vals, n)
    return [rng.choice(vals) for _ in range(n)]


def _make(rng: random.Random, n: int, vals: list[int], pbound: int = 4) -> tuple[Matrix, Matrix, Matrix]:
    """A = P D P^-1 nguyên, với P unimodular. Trả về (A, P, D)."""
    for _ in range(2000):
        p = _unimodular(rng, n, pbound, rng.randint(2, 4))
        # tránh cột tỉ lệ hoặc quá phức tạp
        d = diag(*vals)
        a = p * d * p.inv()
        if max(abs(x) for x in a) <= 20 and a != d and not any(all(x == 0 for x in p[:, j]) for j in range(n)):
            return a, p, d
    raise RuntimeError("không dựng được A = PDP^-1")


def _charpoly_steps(a: Matrix, roots: list) -> list[Step]:
    n = a.rows
    poly = (LAM * eye(n) - a).det().expand()
    steps = [Step(text="Phương trình đặc trưng $\\det(\\lambda\\mathbf{I} - \\mathbf{A}) = 0$:",
                  latex=f"\\det\\left(\\lambda\\mathbf{{I}} - {L.mat(a)}\\right) = {latex(poly)} = 0")]
    steps.append(Step(text="Phân tích thành nhân tử:", latex=f"{latex(factor(poly))} = 0"))
    steps.append(Step(text="Các trị riêng (theo thứ tự tăng dần, lặp theo bội):", latex=", ".join(L.num(r) for r in roots)))
    return steps


def _verify_eigs(a: Matrix, roots: list) -> None:
    n = a.rows
    check(sum(roots) == a.trace() and np_prod(roots) == a.det(), "tổng = vết, tích = định thức")
    for r in set(roots):
        check((a - r * eye(n)).det() == 0, "det(A - λI) = 0")
    np_close(Matrix(sorted(roots)), np.sort(np.real(np.linalg.eigvals(to_np(a)))))


def np_prod(xs):
    out = Rational(1)
    for x in xs:
        out *= x
    return out


# ---------------------------------------------------------------- eig-values
def gen_eig_values(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 1:  # tam giác: trị riêng là đường chéo
        n = rng.choice([2, 3, 4])
        lower = rng.random() < 0.5
        d = [rng.choice([x for x in range(-4, 5)]) for _ in range(n)]
        a = Matrix(n, n, lambda i, j: d[i] if i == j else ((rng.randint(-5, 5)) if (j > i) != lower else 0))
        roots = sorted(Rational(x) for x in d)
        _verify_eigs(a, roots)
        steps = [Step(text="Ma trận là tam giác nên $\\det(\\lambda\\mathbf{I} - \\mathbf{A})$ bằng tích các phần tử đường chéo của $\\lambda\\mathbf{I} - \\mathbf{A}$:",
                      latex="\\det(\\lambda\\mathbf{I} - \\mathbf{A}) = " + "".join(f"(\\lambda - {N(x)})" for x in d)),
                 Step(text="Trị riêng là các phần tử trên đường chéo, xếp tăng dần:", latex=", ".join(L.num(r) for r in roots))]
        text = "Ma trận sau là ma trận tam giác. Liệt kê các trị riêng theo thứ tự tăng dần (lặp lại theo bội)."
        tags = [f"{n}x{n}", "triangular"]
    elif level == 5:  # hiệp phương sai 2x2
        for _ in range(100):
            q = cayley(rng, 2, 50)
            d = diag(*sorted(rng.sample([1, 2, 3, 4, 5, 6, 8, 9, 10, 12], 2)))
            a = q * d * q.T
            if a != d and a.is_symmetric() and not any(x.q > 25 for x in a):
                break
        roots = sorted(d[i, i] for i in range(2))
        _verify_eigs(a, roots)
        steps = _charpoly_steps(a, roots)
        text = ("Ma trận hiệp phương sai $\\boldsymbol{\\Sigma}$ của dữ liệu hai chiều dưới đây có hai trị riêng là phương sai của dữ liệu dọc theo hai hướng chính. "
                "Liệt kê hai trị riêng theo thứ tự tăng dần.")
        tags = ["2x2", "ml"]
    else:
        n = 2 if level == 2 else 3
        if level == 4:
            x, y = rng.sample([-2, -1, 0, 1, 2, 3], 2)  # bội $2$ (chéo hóa được), nhưng không phải ma trận vô hướng
            vals = [x, x, y]
            rng.shuffle(vals)
        else:
            vals = _diag_vals(rng, n)
        a, p, d = _make(rng, n, vals)
        roots = sorted(Rational(x) for x in vals)
        _verify_eigs(a, roots)
        steps = _charpoly_steps(a, roots)
        text = "Tìm các trị riêng của $\\mathbf{A}$. Liệt kê theo thứ tự tăng dần (lặp lại theo bội)."
        tags = [f"{n}x{n}"] + (["repeated"] if len(set(vals)) < len(vals) else [])
    return build(
        type_="eig-values", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=text, display=[L.mat(a, B("A") if level != 5 else "\\boldsymbol{\\Sigma}")]),
        answer=vec_ans(Matrix(roots)),
        hints=["Trị riêng $\\lambda$ là nghiệm của $\\det(\\lambda\\mathbf{I} - \\mathbf{A}) = 0$.",
               "Tổng các trị riêng bằng vết (tổng đường chéo) và tích bằng định thức: dùng để kiểm tra."],
        steps=steps, verified=["sympy", "numpy", "trace-det"], tags=tags,
    )


# --------------------------------------------------------------- eig-vectors
def gen_eig_vectors(rng: random.Random, level: int, seed: int) -> Exercise:
    n = 2 if level == 2 else 3
    if level == 4:  # một trị riêng bội $2$ (không gian riêng $2$ chiều), trị còn lại khác
        x, y = rng.sample([-3, -2, -1, 1, 2, 3, 4], 2)
        vals = [x, x, y]
        rng.shuffle(vals)
    else:
        vals = _diag_vals(rng, n)
    a, p, d = _make(rng, n, vals)
    j = rng.randrange(n)
    lam = Rational(vals[j])
    v = p[:, j]
    check(a * v == lam * v, "Av = λv")
    np_close(a * v, float(lam) * to_np(v))
    m = a - lam * eye(n)
    el = eliminate(m, mode="rref")
    ns = m.nullspace()
    check(len(ns) >= 1 and all(m * x == zeros(n, 1) for x in ns), "null space (A-λI)")
    return build(
        type_="eig-vectors", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"$\\lambda = {L.num(lam)}$ là một trị riêng của $\\mathbf{{A}}$. Tìm một vector riêng (khác $\\mathbf{{0}}$) ứng với $\\lambda$. "
                           f"Mọi bội vô hướng khác $0$ của đáp án đều được chấp nhận.",
                      display=[L.mat(a, B("A"))]),
        answer=vec_ans(v),
        check=Check(mode="eigvec", a=L.entries(a), lam=L.rat_str(lam)),
        hints=["Vector riêng thỏa $(\\mathbf{A} - \\lambda\\mathbf{I})\\mathbf{v} = \\mathbf{0}$: giải hệ thuần nhất này.",
               "Hệ có vô số nghiệm; chọn một nghiệm khác $\\mathbf{0}$ (đặt ẩn tự do bằng $1$ cho dễ)."],
        steps=[Step(text="Lập $\\mathbf{A} - \\lambda\\mathbf{I}$:", latex=L.mat(m, f"\\mathbf{{A}} - {N(lam)}\\mathbf{{I}}"))] + el.steps[1:] +
              [Step(text="Một nghiệm khác $\\mathbf{0}$ của hệ là một vector riêng:", latex=L.mat(v, "\\mathbf{v}")),
               Step(text="Kiểm tra $\\mathbf{A}\\mathbf{v} = \\lambda\\mathbf{v}$:", latex=f"{L.mat(a * v)} = {N(lam)}{L.mat(v)}")],
        verified=["sympy", "numpy", "eigen-equation"], tags=[f"{n}x{n}"] + (["repeated"] if len(set(vals)) < n else []),
    )


# ---------------------------------------------------------------- diagonalize
def gen_diagonalize(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 4:  # có chéo hóa được không?
        n = rng.choice([2, 3])
        diagonalizable = rng.random() < 0.5
        for _ in range(300):
            p = _unimodular(rng, n, 4, rng.randint(2, 4))
            if diagonalizable:
                vals = [rng.choice([-2, -1, 1, 2, 3]) for _ in range(n)]
                vals[0] = vals[1]  # bội 2 nhưng vẫn chéo hóa được
                j = diag(*vals)
            else:
                lam = rng.choice([-2, -1, 1, 2, 3])
                j = Matrix(n, n, lambda i, k: lam if i == k else (1 if k == i + 1 and i == 0 else 0))  # khối Jordan
                if n == 3:
                    j[2, 2] = rng.choice([x for x in [-2, -1, 1, 2, 3] if x != lam])
            a = p * j * p.inv()
            if max(abs(x) for x in a) <= 14 and a != j:
                break
        check(a.is_diagonalizable() == diagonalizable, "diagonalizable sympy")
        roots = {}
        for r, mult in a.eigenvals().items():
            roots[r] = mult
        steps = []
        for r, mult in sorted(roots.items()):
            rk = (a - r * eye(n)).rank()
            geo = n - rk
            steps.append(Step(text=f"Trị riêng $\\lambda = {L.num(r)}$ có bội đại số {mult}; $\\rank(\\mathbf{{A}} - \\lambda\\mathbf{{I}}) = {rk}$ nên số vector riêng độc lập (bội hình học) là ${n} - {rk} = {geo}$."))
        steps.append(Step(text="Chéo hóa được khi và chỉ khi với mọi trị riêng, bội hình học bằng bội đại số (tổng bội hình học bằng $n$). "
                               + ("Điều kiện này thỏa." if diagonalizable else "Có trị riêng mà bội hình học nhỏ hơn bội đại số: không chéo hóa được.")))
        return build(
            type_="diagonalize", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Ma trận $\\mathbf{A}$ có chéo hóa được không (tồn tại $\\mathbf{P}$ khả nghịch để $\\mathbf{P}^{-1}\\mathbf{A}\\mathbf{P}$ là ma trận đường chéo)?",
                          display=[L.mat(a, B("A"))]),
            answer=bool_ans(diagonalizable),
            hints=["Tìm các trị riêng và số vector riêng độc lập của mỗi trị riêng.", "Cần đủ $n$ vector riêng độc lập."],
            steps=steps, verified=["sympy", "rank"], tags=[f"{n}x{n}", "diagonalizable" if diagonalizable else "defective"],
        )
    n = 2 if level == 2 else 3
    vals = _diag_vals(rng, n)
    a, p, d = _make(rng, n, vals)
    check(p.inv() * a * p == d, "P^-1 A P = D")
    verify_matmul(a, p, p * d)
    inv = p.inv()
    return build(
        type_="diagonalize", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Các cột của $\\mathbf{P}$ là các vector riêng của $\\mathbf{A}$ (theo thứ tự đó). Tính $\\mathbf{D} = \\mathbf{P}^{-1}\\mathbf{A}\\mathbf{P}$.",
                      display=[L.mat(a, B("A")), L.mat(p, B("P")), L.mat(inv, "\\mathbf{P}^{-1}")]),
        answer=mat_ans(d),
        hints=["$\\mathbf{D}$ là ma trận đường chéo chứa các trị riêng theo đúng thứ tự các cột của $\\mathbf{P}$.",
               "Hoặc tính trực tiếp: $\\mathbf{A}\\mathbf{P}$ trước, rồi nhân bên trái với $\\mathbf{P}^{-1}$."],
        steps=[Step(text="Tính $\\mathbf{A}\\mathbf{P}$ (mỗi cột là $\\lambda_j\\mathbf{p}_j$):", latex=L.mat(a * p, "\\mathbf{A}\\mathbf{P}")),
               Step(text="Nhân bên trái với $\\mathbf{P}^{-1}$:", latex=L.mat(d, "\\mathbf{D} = \\mathbf{P}^{-1}\\mathbf{A}\\mathbf{P}"))],
        verified=["sympy", "numpy", "identity"], tags=[size_tag(a)],
    )


# ------------------------------------------------------------ power-via-diag
def gen_power(rng: random.Random, level: int, seed: int) -> Exercise:
    n = 2 if level == 2 else 3
    vals = _diag_vals(rng, n, pool=(-2, -1, 1, 2, 3))
    a, p, d = _make(rng, n, vals, pbound=3)
    inv = p.inv()
    k = rng.randint(2, 5 if n == 2 else 4)
    if level == 4:
        c = Matrix([rng.choice([-2, -1, 1, 2]) for _ in range(n)])
        v = p * c
        res = (a ** k) * v
        dk = diag(*[x ** k for x in vals])
        check(res == p * dk * c, "A^k v = P D^k c")
        np_close(res, np.linalg.matrix_power(to_np(a), k) @ to_np(v))
        return build(
            type_="power-via-diag", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"$\\mathbf{{A}} = \\mathbf{{P}}\\mathbf{{D}}\\mathbf{{P}}^{{-1}}$ với các ma trận dưới đây, và $\\mathbf{{v}} = \\mathbf{{P}}\\mathbf{{c}}$. "
                               f"Tính $\\mathbf{{A}}^{{{k}}}\\mathbf{{v}}$ (gợi ý: $\\mathbf{{A}}^{{{k}}}\\mathbf{{v}} = \\mathbf{{P}}\\mathbf{{D}}^{{{k}}}\\mathbf{{c}}$).",
                          display=[L.mat(a, B("A")), L.mat(p, B("P")), L.mat(d, B("D")), L.mat(c, B("c"))]),
            answer=vec_ans(res),
            hints=[f"Tính $\\mathbf{{D}}^{{{k}}}$ bằng cách lũy thừa từng phần tử đường chéo.", "Rồi $\\mathbf{P}(\\mathbf{D}^{k}\\mathbf{c})$: nhân từ phải sang trái."],
            steps=[Step(text=f"$\\mathbf{{D}}^{{{k}}}$:", latex=L.mat(dk, f"\\mathbf{{D}}^{{{k}}}")),
                   Step(text="$\\mathbf{D}^{k}\\mathbf{c}$:", latex=L.mat(dk * c)),
                   Step(text="Nhân với $\\mathbf{P}$:", latex=L.mat(res, f"\\mathbf{{A}}^{{{k}}}\\mathbf{{v}}"))],
            verified=["sympy", "numpy"], tags=[f"{n}x{n}", f"k={k}"],
        )
    dk = diag(*[x ** k for x in vals])
    res = p * dk * inv
    check(res == a ** k, "A^k = P D^k P^-1")
    np_close(res, np.linalg.matrix_power(to_np(a), k))
    return build(
        type_="power-via-diag", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"$\\mathbf{{A}} = \\mathbf{{P}}\\mathbf{{D}}\\mathbf{{P}}^{{-1}}$ với các ma trận dưới đây. Tính $\\mathbf{{A}}^{{{k}}}$ bằng cách dùng $\\mathbf{{A}}^{{{k}}} = \\mathbf{{P}}\\mathbf{{D}}^{{{k}}}\\mathbf{{P}}^{{-1}}$.",
                      display=[L.mat(a, B("A")), L.mat(p, B("P")), L.mat(d, B("D")), L.mat(inv, "\\mathbf{P}^{-1}")]),
        answer=mat_ans(res),
        hints=[f"Lũy thừa ${k}$ từng phần tử trên đường chéo của $\\mathbf{{D}}$.", "Rồi nhân $\\mathbf{P}\\mathbf{D}^{k}\\mathbf{P}^{-1}$."],
        steps=[Step(text=f"$\\mathbf{{D}}^{{{k}}}$:", latex=L.mat(dk, f"\\mathbf{{D}}^{{{k}}}")),
               Step(text="$\\mathbf{P}\\mathbf{D}^{k}$:", latex=L.mat(p * dk)),
               Step(text="Nhân với $\\mathbf{P}^{-1}$:", latex=L.mat(res, f"\\mathbf{{A}}^{{{k}}}"))],
        verified=["sympy", "numpy"], tags=[size_tag(a), f"k={k}"],
    )


SPECS = [
    TopicSpec("eig-values", LESSON, (1, 2, 3, 4, 5), gen_eig_values),
    TopicSpec("eig-vectors", LESSON, (2, 3, 4), gen_eig_vectors),
    TopicSpec("diagonalize", LESSON, (2, 3, 4), gen_diagonalize),
    TopicSpec("power-via-diag", LESSON, (2, 3, 4), gen_power),
]
