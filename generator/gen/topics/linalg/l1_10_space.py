"""Bài 1.10 Không gian vector và hạng: lin-indep, rank, basis-col, null-space."""
from __future__ import annotations

import random
from math import lcm

from sympy import Matrix, Rational

from ...core import latex as L
from ...core.gauss import eliminate
from ...core.nice import nontrivial_matrix, rand_vector
from ...core.verify import check, verify_rank, verify_rref
from ...schema import Answer, Check, Exercise, Prompt, Step
from ._base import TopicSpec, bool_ans, build, mat_ans, num_ans, size_tag
from .l1_7_gauss import _nice_from_rref

LESSON = "1.10"
B, N = L.bold, L.num_paren


def _cols(vs: list[Matrix]) -> Matrix:
    return Matrix.hstack(*vs)


def _relation(vs: list[Matrix]) -> str | None:
    """Một quan hệ phụ thuộc c1 v1 + ... + ck vk = 0 (từ không gian null), hoặc None nếu độc lập."""
    ns = _cols(vs).nullspace()
    if not ns:
        return None
    c = ns[0]
    scale = 1
    for x in c:
        scale = lcm(scale, Rational(x).q)
    c = c * scale
    terms = " + ".join(f"{N(x)}\\mathbf{{v}}_{{{i + 1}}}" for i, x in enumerate(c) if x != 0)
    check(sum((c[i] * vs[i] for i in range(len(vs))), Matrix.zeros(vs[0].rows, 1)) == Matrix.zeros(vs[0].rows, 1), "quan hệ phụ thuộc")
    return f"{terms} = \\mathbf{{0}}"


# ----------------------------------------------------------------- lin-indep
def gen_indep(rng: random.Random, level: int, seed: int) -> Exercise:
    k, n = {1: (2, 2), 2: rng.choice([(2, 3), (3, 2)]), 3: (3, 3), 4: rng.choice([(4, 3), (3, 4)]), 5: (3, 4)}[level]
    want_dep = rng.random() < 0.5 and not (k > n)
    for _ in range(500):
        vs = [rand_vector(rng, n, -4, 4) for _ in range(k)]
        if want_dep and k >= 2 and k <= n:
            # một vector là tổ hợp của các vector còn lại
            j = rng.randrange(k)
            others = [i for i in range(k) if i != j]
            coef = [rng.choice([-2, -1, 1, 2]) for _ in others]
            vs[j] = sum((c * vs[i] for c, i in zip(coef, others)), Matrix.zeros(n, 1))
            if all(x == 0 for x in vs[j]):
                continue
        a = _cols(vs)
        rk = a.rank()
        if (rk == k) == (not want_dep) or k > n:
            break
    a = _cols(vs)
    rk = a.rank()
    verify_rank(a, rk)
    indep = rk == k
    rel = _relation(vs)
    el = eliminate(a, mode="ref")
    steps = [Step(text="Xếp các vector thành các cột của một ma trận:", latex=L.mat(a))] + el.steps[1:]
    steps.append(Step(text=f"Số pivot (hạng) là {rk}, số vector là {k}."))
    if indep:
        steps.append(Step(text="Hạng bằng số vector nên chúng **độc lập tuyến tính**."))
    else:
        steps.append(Step(text="Hạng nhỏ hơn số vector nên chúng **phụ thuộc tuyến tính**. Một quan hệ phụ thuộc:", latex=rel))
    disp = [L.mat(v, f"\\mathbf{{v}}_{{{i + 1}}}") for i, v in enumerate(vs)]
    text = "Các vector sau có độc lập tuyến tính không?"
    tags = [f"{k}vec", f"R{n}", "independent" if indep else "dependent"]
    if level == 5:
        x = a
        text = (f"Ma trận dữ liệu $\\mathbf{{X}}$ có ${x.rows}$ mẫu và ${x.cols}$ đặc trưng (mỗi cột là một đặc trưng). "
                f"Các đặc trưng (các cột) có độc lập tuyến tính không? (Nếu không, ta gặp hiện tượng đa cộng tuyến.)")
        disp = [L.mat(x, B("X"))]
        tags.append("ml")
    return build(
        type_="lin-indep", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=text, display=disp),
        answer=bool_ans(indep),
        hints=["Các vector độc lập khi cách duy nhất để tổ hợp chúng ra $\\mathbf{0}$ là mọi hệ số bằng $0$.",
               "Xếp chúng thành cột của một ma trận và so sánh hạng với số vector."],
        steps=steps, verified=["sympy", "numpy-rank", "gauss"], tags=tags,
    )


# ---------------------------------------------------------------------- rank
def gen_rank(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 1:
        m, n = rng.choice([(2, 2), (3, 3)])
        r = rng.randint(1, min(m, n))
        a, rr, piv = _nice_from_rref(rng, m, n, r, bound=9, free_max=3)
    elif level == 2:
        m, n = 3, 3
        r = rng.randint(1, 3)
        a, rr, piv = _nice_from_rref(rng, m, n, r, bound=10, free_max=3)
    elif level == 3:
        m, n = rng.choice([(3, 4), (4, 3)])
        r = rng.randint(2, 3)
        a, rr, piv = _nice_from_rref(rng, m, n, r, bound=12, free_max=3)
    elif level == 4:
        m, n = 4, 4
        r = rng.choice([2, 3])
        a, rr, piv = _nice_from_rref(rng, m, n, r, bound=18, free_max=3)
    else:
        m, n = rng.choice([(4, 3), (3, 3), (4, 4)])
        r = n - rng.choice([0, 1, 1])
        a, rr, piv = _nice_from_rref(rng, m, n, max(r, 1), bound=14, free_max=3)
    verify_rank(a, r)
    el = eliminate(a, mode="ref")
    check(len(el.pivots) == r, "rank = số pivot")
    text = "Tính hạng $\\rank(\\mathbf{A})$."
    if level == 5:
        text = ("Ma trận dữ liệu $\\mathbf{X}$ có các hàng là mẫu, các cột là đặc trưng. Tính $\\rank(\\mathbf{X})$, "
                "tức số đặc trưng độc lập tuyến tính thực sự mà dữ liệu mang lại.")
    return build(
        type_="rank", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=text, display=[L.mat(a, B("X") if level == 5 else B("A"))]),
        answer=num_ans(r),
        hints=["Hạng là số pivot sau khi khử Gauss (số hàng khác $0$ của dạng bậc thang).", "Hạng không vượt quá số hàng hay số cột."],
        steps=el.steps + [Step(text="Các cột pivot: " + ", ".join(str(c + 1) for c in el.pivots) + f". Hạng bằng {r}.")],
        verified=["sympy", "numpy-rank", "gauss"], tags=[size_tag(a), f"rank{r}"] + (["ml"] if level == 5 else []),
    )


# ------------------------------------------------------------------ basis-col
def gen_basis_col(rng: random.Random, level: int, seed: int) -> Exercise:
    m, n, r = {2: (3, 3, 2), 3: rng.choice([(3, 4, 2), (3, 4, 3), (4, 3, 2), (4, 3, 3)]), 4: rng.choice([(4, 4, 2), (4, 4, 3)])}[level]
    a, rr, piv = _nice_from_rref(rng, m, n, r, bound=12, free_max=3)
    el = eliminate(a, mode="rref")
    verified = verify_rref(a, el.matrix, el.pivots)
    basis = _cols([a[:, j] for j in el.pivots])
    check(basis.rank() == r and Matrix.hstack(a, basis).rank() == r, "cơ sở không gian cột")
    return build(
        type_="basis-col", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tìm một cơ sở (basis) của không gian cột (column space) của $\\mathbf{A}$. "
                           "Nhập các vector cơ sở thành các **cột** của ma trận kết quả (mọi cơ sở đúng đều được chấp nhận).",
                      display=[L.mat(a, B("A"))]),
        answer=mat_ans(basis), check=Check(mode="span"),
        hints=["Khử Gauss để tìm các cột pivot.", "Lấy các cột **tương ứng của ma trận gốc $\\mathbf{A}$** (không phải của dạng bậc thang)."],
        steps=el.steps + [Step(text="Cột pivot: " + ", ".join(str(c + 1) for c in el.pivots) + ". Lấy các cột này của $\\mathbf{A}$ gốc:", latex=L.mat(basis)),
                          Step(text=f"Số chiều của không gian cột bằng hạng, là {r}.")],
        verified=verified + ["span"], tags=[size_tag(a), f"rank{r}"],
    )


# ------------------------------------------------------------------ null-space
def gen_null_space(rng: random.Random, level: int, seed: int) -> Exercise:
    m, n, r = {2: rng.choice([(2, 3, 1), (2, 3, 2)]), 3: rng.choice([(3, 4, 2), (3, 4, 3), (3, 3, 2)]), 4: rng.choice([(4, 4, 2), (4, 4, 3)])}[level]
    a, rr, piv = _nice_from_rref(rng, m, n, r, bound=12, free_max=3)
    el = eliminate(a, mode="rref")
    verified = verify_rref(a, el.matrix, el.pivots)
    free = [j for j in range(n) if j not in el.pivots]
    basis = []
    for f in free:
        v = Matrix.zeros(n, 1)
        v[f] = 1
        for i, pc in enumerate(el.pivots):
            v[pc] = -el.matrix[i, f]
        basis.append(v)
    nb = _cols(basis)
    check(a * nb == Matrix.zeros(m, len(basis)) and nb.rank() == len(basis) == n - r, "null space")
    ns = a.nullspace()
    check(len(ns) == len(basis) and Matrix.hstack(*ns, *basis).rank() == len(basis), "null space khác sympy")
    return build(
        type_="null-space", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tìm một cơ sở của không gian null (null space) của $\\mathbf{A}$, tức tập nghiệm của $\\mathbf{A}\\mathbf{x} = \\mathbf{0}$. "
                           "Nhập các vector cơ sở thành các **cột** của ma trận kết quả.",
                      display=[L.mat(a, B("A"))]),
        answer=mat_ans(nb), check=Check(mode="span"),
        hints=["Khử Gauss–Jordan $\\mathbf{A}$ để tìm cột pivot và ẩn tự do.", "Mỗi ẩn tự do cho một vector cơ sở: đặt nó bằng $1$, các ẩn tự do khác bằng $0$."],
        steps=el.steps + [Step(text="Ẩn tự do: " + ", ".join(f"$x_{{{j + 1}}}$" for j in free) + ". Mỗi ẩn tự do cho một vector cơ sở:", latex=L.mat(nb)),
                          Step(text=f"Số chiều không gian null bằng $n - r = {n} - {r} = {n - r}$.")],
        verified=verified + ["span", "nullspace"], tags=[size_tag(a), f"nullity{n - r}"],
    )


SPECS = [
    TopicSpec("lin-indep", LESSON, (1, 2, 3, 4, 5), gen_indep),
    TopicSpec("rank", LESSON, (1, 2, 3, 4, 5), gen_rank),
    TopicSpec("basis-col", LESSON, (2, 3, 4), gen_basis_col),
    TopicSpec("null-space", LESSON, (2, 3, 4), gen_null_space),
]
