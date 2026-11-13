"""Bài 1.7 Hệ phương trình và khử Gauss: rowop-apply, ref, rref, solve-unique, solve-param, classify-system."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, eye, zeros

from ...core import latex as L
from ...core.gauss import apply_op, eliminate, op_tex, show
from ...core.nice import nontrivial_matrix
from ...core.verify import (
    check, np_close, to_np, verify_rank, verify_rref, verify_solution, verify_solution_set,
)
from ...schema import Answer, Check, Exercise, Prompt, Step
from ._base import TopicSpec, build, mat_ans, shuffled_choices, size_tag, vec_ans

LESSON = "1.7"
B = L.bold


# ----------------------------------------------------------- dựng ma trận đẹp
def _lu(rng: random.Random, m: int, n: int, dmax: int = 3, off: int = 4, lmax: int = 3) -> tuple[Matrix, Matrix]:
    """A = L·U với L unit lower (m×m), U bậc thang m×n có pivot khác 0 trên đường chéo.
    Khi khử xuôi (không đổi hàng), bội số chính là các phần tử của L: toàn số nguyên."""
    lo = Matrix(m, m, lambda i, j: 1 if i == j else (rng.randint(-lmax, lmax) if i > j else 0))
    nzd = [x for x in range(-dmax, dmax + 1) if x]
    u = Matrix(m, n, lambda i, j: (rng.choice(nzd) if i == j else rng.randint(-off, off)) if j >= i else 0)
    for i in range(min(m, n), m):  # m > n: các hàng cuối của U bằng 0
        u[i, :] = zeros(1, n)
    return lo * u, u


def _elementary_P(rng: random.Random, m: int, ops: int = 4, kmax: int = 2) -> Matrix:
    p = eye(m)
    for _ in range(ops):
        if rng.random() < 0.2:
            i, j = rng.sample(range(m), 2)
            p = apply_op(p, ("swap", i, j))
        else:
            i, j = rng.sample(range(m), 2)
            p = apply_op(p, ("add", i, j, rng.choice([x for x in range(-kmax, kmax + 1) if x])))
    return p


def _rref_target(rng: random.Random, m: int, n: int, r: int, free_max: int = 4) -> tuple[Matrix, list[int]]:
    """Ma trận RREF m×n hạng r với các cột pivot ngẫu nhiên."""
    pivots = sorted(rng.sample(range(n), r))
    rr = zeros(m, n)
    for i, pc in enumerate(pivots):
        rr[i, pc] = 1
        for j in range(pc + 1, n):
            if j not in pivots:
                rr[i, j] = rng.randint(-free_max, free_max)
    return rr, pivots


def _nice_from_rref(rng: random.Random, m: int, n: int, r: int, bound: int, free_max: int = 3) -> tuple[Matrix, Matrix, list[int]]:
    for _ in range(300):
        rr, piv = _rref_target(rng, m, n, r, free_max)
        a = _elementary_P(rng, m, rng.randint(2, 4)) * rr
        if max(abs(x) for x in a) <= bound and a != rr and not any(all(x == 0 for x in a[i, :]) for i in range(m) if i < r):
            return a, rr, piv
    raise RuntimeError("không dựng được ma trận từ RREF")


# ------------------------------------------------------------- rowop-apply
def gen_rowop(rng: random.Random, level: int, seed: int) -> Exercise:
    m = rng.choice([2, 3]) if level == 2 else rng.choice([3, 4])
    n = rng.choice([3, 4]) if level >= 3 else rng.choice([2, 3])
    aug = 1 if level >= 3 and n >= 3 else 0
    mat = nontrivial_matrix(rng, m, n, -6, 6)
    i, j = rng.sample(range(m), 2)
    kinds = ["swap", "add"] if level == 2 else ["swap", "scale", "add", "add"]
    kind = rng.choice(kinds)
    if kind == "swap":
        op = ("swap", i, j)
    elif kind == "scale":
        op = ("scale", i, rng.choice([Rational(-1), Rational(2), Rational(-2), Rational(1, 2), Rational(-3), Rational(3)]))
    else:
        op = ("add", i, j, Rational(rng.choice([-3, -2, -1, 1, 2, 3])))
    res = apply_op(mat, op)
    # đối chiếu độc lập bằng NumPy
    arr = to_np(mat).copy()
    if op[0] == "swap":
        arr[[op[1], op[2]]] = arr[[op[2], op[1]]]
    elif op[0] == "scale":
        arr[op[1]] *= float(op[2])
    else:
        arr[op[1]] += float(op[3]) * arr[op[2]]
    np_close(res, arr)
    if level == 4:  # đoán phép biến đổi
        cands = {op_tex(op)}
        while len(cands) < 4:
            a2, b2 = rng.sample(range(m), 2)
            k2 = rng.choice(["swap", "add", "scale"])
            if k2 == "swap":
                c = ("swap", a2, b2)
            elif k2 == "scale":
                c = ("scale", a2, Rational(rng.choice([-2, 2, 3, -1])))
            else:
                c = ("add", a2, b2, Rational(rng.choice([-3, -2, -1, 1, 2, 3])))
            if apply_op(mat, c) != res:
                cands.add(op_tex(c))
        texs = list(cands)
        choices, correct = shuffled_choices(rng, [f"${op_tex(op)}$"], [f"${t}$" for t in texs if t != op_tex(op)])
        return build(
            type_="rowop-apply", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Phép biến đổi sơ cấp nào biến $\\mathbf{M}$ thành $\\mathbf{M}'$?",
                          display=[L.mat(mat, B("M")), L.mat(res, "\\mathbf{M}'")]),
            answer=Answer(kind="choice", correct=correct), choices=choices,
            hints=["So sánh từng hàng của hai ma trận: hàng nào đã đổi?", "Thử áp dụng từng phép lên $\\mathbf{M}$ rồi so với $\\mathbf{M}'$."],
            steps=[Step(text=f"Áp dụng ${op_tex(op)}$ lên $\\mathbf{{M}}$ cho đúng $\\mathbf{{M}}'$:", latex=show(res))],
            verified=["sympy", "numpy"], tags=["identify", size_tag(mat)],
        )
    return build(
        type_="rowop-apply", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Áp dụng phép biến đổi sơ cấp ${op_tex(op)}$ lên ma trận sau. Ghi ma trận kết quả.",
                      display=[show(mat, aug)]),
        answer=mat_ans(res),
        hints=["Chỉ hàng được nêu ở vế trái của mũi tên thay đổi; các hàng khác giữ nguyên."],
        steps=[Step(text=f"Sau ${op_tex(op)}$:", latex=show(res, aug))],
        verified=["sympy", "numpy"], tags=[size_tag(mat), kind],
    )


# --------------------------------------------------------------------- ref
def gen_ref(rng: random.Random, level: int, seed: int) -> Exercise:
    m, n = {2: rng.choice([(2, 2), (2, 3), (3, 3)]),
            3: rng.choice([(3, 4), (4, 3), (3, 3)]),
            4: rng.choice([(4, 4), (3, 4)])}[level]
    for _ in range(300):
        a, u = _lu(rng, m, n, dmax=3, off=3, lmax=2 if level < 4 else 3)
        if max(abs(x) for x in a) <= (14 if level < 4 else 22) and not any(all(x == 0 for x in a[i, :]) for i in range(m)):
            break
    el = eliminate(a, mode="ref")
    check(el.swaps == 0 and el.matrix == u, "ref: khử Gauss khác U dựng sẵn")
    verify_rank(a, len([i for i in range(min(m, n)) if u[i, i] != 0]))
    return build(
        type_="ref", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Đưa ma trận về dạng bậc thang (row echelon form) bằng khử Gauss thuần túy: chỉ dùng phép "
                           "$R_i \\leftarrow R_i + kR_j$ (cộng bội của hàng pivot vào hàng dưới), không đổi hàng, "
                           "không chia hàng pivot. Ghi ma trận kết quả.",
                      display=[L.mat(a, B("A"))]),
        answer=mat_ans(el.matrix),
        hints=["Khử từng cột từ trái sang: dùng phần tử pivot để làm các phần tử bên dưới bằng $0$.",
               "Với phần tử $a$ cần khử và pivot $p$: dùng $R_i \\leftarrow R_i - (a/p)R_j$."],
        steps=el.steps, verified=["sympy", "gauss", "numpy-rank"], tags=[size_tag(a)],
    )


# -------------------------------------------------------------------- rref
def gen_rref(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        m, n, r = rng.choice([(2, 2, 2), (2, 3, 2), (2, 3, 1)])
    elif level == 3:
        m, n, r = rng.choice([(3, 3, 3), (3, 4, 3), (3, 4, 2), (3, 3, 2), (4, 3, 3)])
    else:
        m, n, r = rng.choice([(4, 4, 4), (4, 4, 3), (3, 4, 3), (4, 4, 2)])
    r = min(r, m, n)
    a, rr, piv = _nice_from_rref(rng, m, n, r, bound=12 if level < 4 else 20)
    el = eliminate(a, mode="rref")
    verified = verify_rref(a, el.matrix, el.pivots)
    check(el.matrix == rr, "rref: khác RREF dựng sẵn")
    return build(
        type_="rref", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Đưa ma trận về dạng bậc thang rút gọn (RREF) bằng khử Gauss–Jordan.", display=[L.mat(a, B("A"))]),
        answer=mat_ans(rr),
        hints=["Mỗi pivot phải bằng $1$ và là phần tử khác $0$ duy nhất trong cột của nó.",
               "Khử xuống dưới trước, rồi khử lên trên, hoặc kết hợp từng cột."],
        steps=el.steps + [Step(text=f"Các cột pivot: " + ", ".join(str(c + 1) for c in el.pivots) + f". Hạng bằng {len(el.pivots)}.")],
        verified=verified, tags=[size_tag(a), f"rank{r}"],
    )


# ------------------------------------------------------------- solve-unique
def gen_solve_unique(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 5:  # đường thẳng qua hai điểm
        for _ in range(100):
            x1, x2 = rng.sample(range(-3, 5), 2)
            a0, b0 = rng.randint(-4, 4), rng.choice([x for x in range(-3, 4) if x])
            break
        y1, y2 = a0 + b0 * x1, a0 + b0 * x2
        a = Matrix([[1, x1], [1, x2]])
        b = Matrix([y1, y2])
        x = Matrix([a0, b0])
        verify_solution(a, b, x)
        np_close(x, np.linalg.solve(to_np(a), to_np(b)))
        el = eliminate(a.row_join(b), aug=1, mode="rref")
        return build(
            type_="solve-unique", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text=f"Đường thẳng $y = a + bx$ đi qua hai điểm $({x1}, {y1})$ và $({x2}, {y2})$. "
                               f"Tìm $(a, b)$ (đây là bài toán hồi quy tuyến tính đơn giản nhất với hai điểm).",
                          display=[L.system(a, b, "w").replace("w_{1}", "a").replace("w_{2}", "b")]),
            answer=vec_ans(x),
            hints=["Thế từng điểm vào $y = a + bx$ để được hệ hai phương trình hai ẩn $a, b$."],
            steps=el.steps, verified=["sympy", "numpy", "substitution"], tags=["ml", "2x2"],
        )
    n = {2: 2, 3: 3, 4: 4}[level]
    for _ in range(300):
        if level < 4:
            a, _u = _lu(rng, n, n, dmax=3, off=3, lmax=2)
        else:
            a = nontrivial_matrix(rng, n, n, -4, 4)
        if a.det() != 0 and max(abs(x) for x in a) <= 14:
            break
    x = Matrix([rng.randint(-5, 5) for _ in range(n)])
    b = a * x
    verify_solution(a, b, x)
    np_close(x, np.linalg.solve(to_np(a), to_np(b)))
    el = eliminate(a.row_join(b), aug=1, mode="rref")
    check(el.matrix[:, n] == x, "RREF cho nghiệm khác")
    return build(
        type_="solve-unique", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Giải hệ phương trình. Nhập nghiệm là vector $(x_1, \\dots, x_n)$.", display=[L.system(a, b)]),
        answer=vec_ans(x),
        hints=["Viết ma trận mở rộng $[\\mathbf{A}\\,|\\,\\mathbf{b}]$ rồi khử Gauss–Jordan.", "Đọc nghiệm ở cột cuối của dạng RREF."],
        steps=[Step(text="Ma trận mở rộng và khử Gauss–Jordan:")] + el.steps[0:] +
              [Step(text="Đọc nghiệm từ cột cuối:", latex=L.mat(x, "\\mathbf{x}"))],
        verified=["sympy", "numpy", "substitution"], tags=[f"{n}x{n}"],
    )


# -------------------------------------------------------------- solve-param
def _solution_set_from_rref(rr: Matrix, n: int, piv: list[int]) -> tuple[Matrix, list[Matrix]]:
    """Từ RREF của [A|b]: nghiệm riêng (ẩn tự do = 0) và cơ sở không gian null."""
    free = [j for j in range(n) if j not in piv]
    xp = zeros(n, 1)
    for i, pc in enumerate(piv):
        xp[pc] = rr[i, n]
    basis = []
    for f in free:
        v = zeros(n, 1)
        v[f] = 1
        for i, pc in enumerate(piv):
            v[pc] = -rr[i, f]
        basis.append(v)
    return xp, basis


def gen_solve_param(rng: random.Random, level: int, seed: int) -> Exercise:
    n, m, r = rng.choice([(3, 3, 2), (3, 2, 2)]) if level == 3 else rng.choice([(4, 3, 2), (4, 3, 3), (4, 4, 3), (4, 4, 2)])
    # RREF của A (m×n, hạng r) rồi A = P·R; b = A·x0
    for _ in range(300):
        a, rr, piv = _nice_from_rref(rng, m, n, r, bound=10, free_max=3)
        x0 = Matrix([rng.randint(-3, 3) for _ in range(n)])
        b = a * x0
        if max(abs(v) for v in b) <= 30:
            break
    aug = a.row_join(b)
    el = eliminate(aug, aug=1, mode="rref")
    check(el.pivots == piv, "pivot khác dự kiến")
    xp, basis = _solution_set_from_rref(el.matrix, n, el.pivots)
    verified = verify_solution_set(a, b, xp, basis)
    # đối chiếu không gian null với SymPy
    ns = a.nullspace()
    check(len(ns) == len(basis) and Matrix.hstack(*ns, *basis).rank() == len(basis), "null space khác sympy")
    free = [j for j in range(n) if j not in el.pivots]
    gen_txt = L.mat(xp, "\\mathbf{x}") + "".join(f" + t_{{{k + 1}}}{L.mat(v)}" for k, v in enumerate(basis))
    steps = el.steps + [
        Step(text=f"Ẩn tự do: " + ", ".join(f"$x_{{{j + 1}}}$" for j in free) + ". Đặt chúng bằng tham số, các ẩn pivot tính theo tham số."),
        Step(text="Nghiệm tổng quát (nghiệm riêng cộng tổ hợp các vector cơ sở của không gian null):", latex=gen_txt),
    ]
    return build(
        type_="solve-param", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Hệ sau có vô số nghiệm. Viết nghiệm dưới dạng $\\mathbf{x} = \\mathbf{x}_p + t_1\\mathbf{v}_1 + \\cdots$: "
                           "nhập một nghiệm riêng $\\mathbf{x}_p$ và các vector hướng $\\mathbf{v}_i$ (mỗi ẩn tự do ứng với một vector).",
                      display=[L.system(a, b)]),
        answer=Answer(kind="solution-set", particular=L.entries(xp)[0:0] or [L.rat_str(v) for v in xp],
                      basis=[[L.rat_str(v) for v in vec] for vec in basis], rows=n, cols=len(basis)),
        check=Check(mode="solution-set", a=L.entries(a), b=[L.rat_str(v) for v in b]),
        hints=["Khử Gauss–Jordan ma trận mở rộng. Cột nào không có pivot ứng với ẩn tự do.",
               "Đặt ẩn tự do bằng $0$ để có nghiệm riêng; đặt một ẩn tự do bằng $1$ (còn lại $0$, vế phải $0$) để có vector hướng."],
        steps=steps, verified=verified, tags=[f"{m}x{n}", f"free{len(basis)}"],
    )


# ----------------------------------------------------------- classify-system
UNIQUE, INFINITE, NONE = "Có nghiệm duy nhất", "Có vô số nghiệm", "Vô nghiệm"


def _classify(a: Matrix, b: Matrix) -> str:
    ra, rb = a.rank(), a.row_join(b).rank()
    verify_rank(a, ra)
    verify_rank(a.row_join(b), rb)
    if ra < rb:
        return NONE
    return UNIQUE if ra == a.cols else INFINITE


def gen_classify(rng: random.Random, level: int, seed: int) -> Exercise:
    target = rng.choice([UNIQUE, INFINITE, NONE])
    if level == 5:  # hồi quy: 3 điểm, mô hình y = a + b x
        xs = rng.sample(range(-2, 4), 3)
        a0, b0 = rng.randint(-3, 3), rng.choice([-2, -1, 1, 2])
        ys = [a0 + b0 * x for x in xs]
        if target != UNIQUE:
            k = rng.randrange(3)
            ys[k] += rng.choice([-2, -1, 1, 2])
            target = NONE
        a = Matrix([[1, x] for x in xs])
        b = Matrix(ys)
        label = _classify(a, b)
        pts = ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))
        steps = eliminate(a.row_join(b), aug=1, mode="ref").steps
        steps.append(Step(text="Hạng của $\\mathbf{A}$ và của ma trận mở rộng: " + f"{a.rank()} và {a.row_join(b).rank()}."))
        choices = [UNIQUE, INFINITE, NONE]
        return build(
            type_="classify-system", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text=f"Ta muốn một đường thẳng $y = a + bx$ đi qua đúng cả ba điểm {pts} (hệ có $3$ phương trình, $2$ ẩn $a, b$). "
                               f"Hệ phương trình này có bao nhiêu nghiệm?", display=[L.system(a, b, "w").replace("w_{1}", "a").replace("w_{2}", "b")]),
            answer=Answer(kind="choice", correct=[choices.index(label)]), choices=choices,
            hints=["Khử Gauss và so sánh hạng của $\\mathbf{A}$ với hạng của $[\\mathbf{A}|\\mathbf{b}]$."],
            steps=steps, verified=["sympy", "numpy-rank"], tags=["ml", "overdetermined"],
        )
    n = rng.choice([2, 3]) if level == 2 else rng.choice([3, 4])
    for _ in range(300):
        if target == UNIQUE:
            m = n if level < 4 else rng.choice([n, n + 1 if n < 4 else n])
            a = nontrivial_matrix(rng, m, n, -4, 4)
            if a.rank() != n:
                continue
            x0 = Matrix([rng.randint(-3, 3) for _ in range(n)])
            b = a * x0
        elif target == INFINITE:
            m = n if level < 4 else rng.choice([n - 1, n])
            r = n - 1
            a, _, _ = _nice_from_rref(rng, m, n, min(r, m), bound=8, free_max=2)
            x0 = Matrix([rng.randint(-3, 3) for _ in range(n)])
            b = a * x0
        else:
            m = n if level < 4 else n
            r = n - 1
            a, _, _ = _nice_from_rref(rng, m, n, r, bound=8, free_max=2)
            b = a * Matrix([rng.randint(-3, 3) for _ in range(n)])
            b = b + Matrix([0] * (m - 1) + [rng.choice([-2, -1, 1, 2, 3])])
        if _classify(a, b) == target:
            break
    else:
        raise RuntimeError("classify")
    el = eliminate(a.row_join(b), aug=1, mode="ref")
    ra, rb = a.rank(), a.row_join(b).rank()
    steps = el.steps + [Step(text=f"Hạng của $\\mathbf{{A}}$ là {ra}, hạng của $[\\mathbf{{A}}|\\mathbf{{b}}]$ là {rb}, số ẩn là {n}.")]
    if target == NONE:
        steps.append(Step(text="Hạng của ma trận mở rộng lớn hơn hạng của $\\mathbf{A}$ (xuất hiện hàng $0 = c$ với $c \\neq 0$): vô nghiệm."))
    elif target == UNIQUE:
        steps.append(Step(text="Hai hạng bằng nhau và bằng số ẩn: nghiệm duy nhất."))
    else:
        steps.append(Step(text="Hai hạng bằng nhau nhưng nhỏ hơn số ẩn: còn ẩn tự do nên vô số nghiệm."))
    choices = [UNIQUE, INFINITE, NONE]
    return build(
        type_="classify-system", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Hệ phương trình sau có bao nhiêu nghiệm?", display=[L.system(a, b)]),
        answer=Answer(kind="choice", correct=[choices.index(target)]), choices=choices,
        hints=["So sánh $\\operatorname{rank}(\\mathbf{A})$, $\\operatorname{rank}([\\mathbf{A}|\\mathbf{b}])$ và số ẩn.",
               "Khác hạng: vô nghiệm. Cùng hạng bằng số ẩn: duy nhất. Cùng hạng nhỏ hơn số ẩn: vô số."],
        steps=steps, verified=["sympy", "numpy-rank", "rank"], tags=[f"{a.rows}x{n}", {UNIQUE: "unique", INFINITE: "infinite", NONE: "none"}[target]],
    )


SPECS = [
    TopicSpec("rowop-apply", LESSON, (2, 3, 4), gen_rowop),
    TopicSpec("ref", LESSON, (2, 3, 4), gen_ref),
    TopicSpec("rref", LESSON, (2, 3, 4), gen_rref),
    TopicSpec("solve-unique", LESSON, (2, 3, 4, 5), gen_solve_unique),
    TopicSpec("solve-param", LESSON, (3, 4), gen_solve_param),
    TopicSpec("classify-system", LESSON, (2, 3, 4, 5), gen_classify),
]
