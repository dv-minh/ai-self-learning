"""Bài 1.8 Định thức: det, det-laplace, det-rowops, det-properties, det-param."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, Symbol, solve, zeros

from ...core import latex as L
from ...core.gauss import eliminate
from ...core.nice import nontrivial_matrix
from ...core.verify import check, np_close, to_np, verify_det
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, bool_ans, build, num_ans, shuffled_choices, size_tag

LESSON = "1.8"
B = L.bold
N = L.num_paren


# ----------------------------------------------------------- lời giải det
def det2_expr(m: Matrix) -> str:
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    return f"{N(a)}\\cdot {N(d)} - {N(b)}\\cdot {N(c)}"


def sarrus_expr(m: Matrix) -> str:
    g = lambda i, j: N(m[i, j])
    pos = [(0, 0, 1, 1, 2, 2), (0, 1, 1, 2, 2, 0), (0, 2, 1, 0, 2, 1)]
    neg = [(0, 2, 1, 1, 2, 0), (0, 0, 1, 2, 2, 1), (0, 1, 1, 0, 2, 2)]
    t = lambda p: f"{g(p[0], p[1])}{g(p[2], p[3])}{g(p[4], p[5])}"
    return " + ".join(t(p) for p in pos) + " - " + " - ".join(f"[{t(p)}]" for p in neg)


def det_steps(m: Matrix, d, row: int | None = None) -> list[Step]:
    n = m.rows
    if n == 2:
        return [Step(text="Công thức $ad-bc$:", latex=f"{det2_expr(m)} = {L.num(d)}")]
    if n == 3 and row is None:
        return [Step(text="Quy tắc Sarrus (ba tích theo đường chéo chính trừ ba tích theo đường chéo phụ):",
                     latex=f"{sarrus_expr(m)} = {L.num(d)}")]
    return laplace_steps(m, d, row)


def laplace_steps(m: Matrix, d, row: int | None = None, col: int | None = None) -> list[Step]:
    n = m.rows
    along_col = col is not None
    k = col if along_col else (row if row is not None else min(range(n), key=lambda i: sum(1 for j in range(n) if m[i, j] != 0)))
    cells = [(i, k) if along_col else (k, j) for i in range(n) for j in range(n) if (j == k if False else True)]
    cells = [(i, k) for i in range(n)] if along_col else [(k, j) for j in range(n)]
    what = f"cột {k + 1}" if along_col else f"hàng {k + 1}"
    steps = [Step(text=f"Khai triển Laplace theo {what}: $\\det(\\mathbf{{A}}) = \\sum a_{{ij}} C_{{ij}}$, với $C_{{ij}} = (-1)^{{i+j}} M_{{ij}}$ "
                       f"và $M_{{ij}}$ là định thức của ma trận con bỏ hàng $i$ và cột $j$.")]
    terms = []
    total = 0
    for (i, j) in cells:
        a = m[i, j]
        if a == 0:
            steps.append(Step(text=f"$a_{{{i + 1}{j + 1}}} = 0$ nên số hạng này bằng $0$ (bỏ qua)."))
            continue
        minor = m.minor_submatrix(i, j)
        mij = minor.det()
        sign = (-1) ** (i + j)
        total += sign * a * mij
        steps.append(Step(
            text=f"Số hạng $a_{{{i + 1}{j + 1}}}C_{{{i + 1}{j + 1}}}$, ma trận con $M_{{{i + 1}{j + 1}}} = {L.num(mij)}$ (dấu {'+' if sign > 0 else '−'}):",
            latex=f"{N(a)}\\cdot ({'+' if sign > 0 else '-'}1)\\cdot {N(mij)} = {L.num(sign * a * mij)}",
        ))
        terms.append(L.num(sign * a * mij))
    check(total == d, "Laplace: tổng các số hạng khác định thức")
    steps.append(Step(text="Cộng các số hạng:", latex=" + ".join(terms) + f" = {L.num(d)}" if terms else L.num(d)))
    return steps


# ------------------------------------------------------------------ det
def _sparse4(rng: random.Random, lo: int, hi: int) -> Matrix:
    for _ in range(500):
        m = nontrivial_matrix(rng, 4, 4, lo, hi)
        zeros_in_row = max(sum(1 for j in range(4) if m[i, j] == 0) for i in range(4))
        if zeros_in_row >= 2:
            return m
        # ép hai số 0 ở một hàng ngẫu nhiên
        i = rng.randrange(4)
        js = rng.sample(range(4), 2)
        for j in js:
            m[i, j] = 0
        if not any(all(m[r, c] == 0 for c in range(4)) for r in range(4)) and not any(all(m[r, c] == 0 for r in range(4)) for c in range(4)):
            return m
    raise RuntimeError("sparse4")


def gen_det(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 1:
        m = nontrivial_matrix(rng, 2, 2, -9, 9)
    elif level == 2:
        m = nontrivial_matrix(rng, 3, 3, -5, 5)
    elif level == 3:
        m = nontrivial_matrix(rng, 3, 3, -9, 9) if rng.random() < 0.5 else _sparse4(rng, -4, 4)
    else:
        m = _sparse4(rng, -6, 6)
    d = m.det()
    verified = verify_det(m, d)
    return build(
        type_="det", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tính định thức $\\det(\\mathbf{A})$.", display=[L.mat(m, B("A"))]),
        answer=num_ans(d),
        hints={2: ["Công thức: $ad - bc$."], 3: ["Dùng quy tắc Sarrus hoặc khai triển Laplace."]}.get(m.rows, ["Khai triển Laplace theo hàng hoặc cột có nhiều số $0$ nhất."]),
        steps=det_steps(m, d), verified=verified, tags=[size_tag(m)],
    )


# ------------------------------------------------------------- det-laplace
def gen_laplace(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:  # một phần phụ đại số
        m = nontrivial_matrix(rng, 3, 3, -5, 5)
        i, j = rng.randrange(3), rng.randrange(3)
        minor = m.minor_submatrix(i, j)
        cof = (-1) ** (i + j) * minor.det()
        check(cof == m.cofactor(i, j), "cofactor sympy")
        return build(
            type_="det-laplace", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text=f"Tính phần phụ đại số $C_{{{i + 1}{j + 1}}} = (-1)^{{{i + 1}+{j + 1}}} M_{{{i + 1}{j + 1}}}$, "
                               f"trong đó $M_{{{i + 1}{j + 1}}}$ là định thức sau khi bỏ hàng {i + 1} và cột {j + 1}.",
                          display=[L.mat(m, B("A"))]),
            answer=num_ans(cof),
            hints=[f"Bỏ hàng {i + 1}, cột {j + 1} để được ma trận con $2\\times 2$.", "Nhân với dấu $(-1)^{i+j}$."],
            steps=[Step(text="Ma trận con:", latex=L.mat(minor)),
                   Step(text="Định thức của ma trận con:", latex=f"M_{{{i + 1}{j + 1}}} = {det2_expr(minor)} = {L.num(minor.det())}"),
                   Step(text="Nhân dấu:", latex=f"C_{{{i + 1}{j + 1}}} = (-1)^{{{i + 1}+{j + 1}}}\\cdot {N(minor.det())} = {L.num(cof)}")],
            verified=["sympy", "cofactor"], tags=["3x3", "cofactor"],
        )
    n = 3 if level == 3 else 4
    m = nontrivial_matrix(rng, 3, 3, -6, 6) if n == 3 else _sparse4(rng, -5, 5)
    along_col = rng.random() < 0.4
    # chọn hàng/cột có nhiều số 0 nhất cho n = 4
    counts = [sum(1 for j in range(n) if (m[k, j] if not along_col else m[j, k]) == 0) for k in range(n)]
    k = counts.index(max(counts)) if n == 4 else rng.randrange(n)
    d = m.det()
    verified = verify_det(m, d)
    where = f"cột {k + 1}" if along_col else f"hàng {k + 1}"
    return build(
        type_="det-laplace", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tính $\\det(\\mathbf{{A}})$ bằng khai triển Laplace theo {where}.", display=[L.mat(m, B("A"))]),
        answer=num_ans(d),
        hints=["Mỗi số hạng là $a_{ij}$ nhân phần phụ đại số $C_{ij} = (-1)^{i+j}M_{ij}$.",
               "Số hạng có $a_{ij}=0$ bỏ qua được.", "Dấu theo bàn cờ: $+\\,-\\,+\\,-\\cdots$ bắt đầu bằng $+$ ở góc trái trên."],
        steps=laplace_steps(m, d, row=None if along_col else k, col=k if along_col else None),
        verified=verified, tags=[size_tag(m), "col" if along_col else "row"],
    )


# ------------------------------------------------------------- det-rowops
def gen_rowops(rng: random.Random, level: int, seed: int) -> Exercise:
    from .l1_7_gauss import _lu

    n = 3 if level == 2 else 4
    for _ in range(300):
        a, _u = _lu(rng, n, n, dmax=3, off=3, lmax=2)
        if level == 4:
            i, j = rng.sample(range(n), 2)
            a.row_swap(i, j)
        if max(abs(x) for x in a) <= (12 if level < 4 else 16):
            break
    d = a.det()
    verified = verify_det(a, d)
    el = eliminate(a, mode="ref")
    diag_prod = 1
    for i in range(n):
        diag_prod *= el.matrix[i, i]
    sign = (-1) ** el.swaps
    check(sign * diag_prod == d, "det-rowops")
    steps = el.steps + [
        Step(text="Phép $R_i \\leftarrow R_i + kR_j$ không đổi định thức; mỗi phép đổi hàng đổi dấu định thức. "
                  f"Đã đổi hàng {el.swaps} lần. Ma trận bậc thang là tam giác trên nên định thức bằng tích đường chéo (nhân $(-1)^{{{el.swaps}}}$):",
             latex=f"\\det(\\mathbf{{A}}) = (-1)^{{{el.swaps}}} \\cdot " + " \\cdot ".join(N(el.matrix[i, i]) for i in range(n)) + f" = {L.num(d)}"),
    ]
    return build(
        type_="det-rowops", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tính $\\det(\\mathbf{A})$ bằng cách đưa về dạng tam giác trên bằng các phép biến đổi hàng.",
                      display=[L.mat(a, B("A"))]),
        answer=num_ans(d),
        hints=["Cộng bội của một hàng vào hàng khác không làm đổi định thức.", "Đổi chỗ hai hàng làm đổi dấu định thức.",
               "Định thức của ma trận tam giác bằng tích các phần tử trên đường chéo."],
        steps=steps, verified=verified, tags=[size_tag(a), f"swaps{el.swaps}"],
    )


# ---------------------------------------------------------- det-properties
def _claims(rng: random.Random) -> tuple[str, bool, list[Step], Matrix, Matrix | None, str]:
    """Trả về (phát biểu, đúng?, lời giải, A, B) cho một khẳng định kiểm chứng bằng số."""
    n = 2
    a, b = nontrivial_matrix(rng, n, n, -4, 4), nontrivial_matrix(rng, n, n, -4, 4)
    pre = ""
    kind = rng.choice(["T", "mult", "add", "scalar-wrong", "scalar-ok", "swap", "inv"])
    k = Rational(rng.choice([2, 3, -2]))
    da, db = a.det(), b.det()
    if kind == "T":
        stmt, truth = "\\det(\\mathbf{A}\\T) = \\det(\\mathbf{A})", a.T.det() == da
        lhs, rhs = a.T.det(), da
        sol = [Step(text="Tính hai vế:", latex=f"\\det(\\mathbf{{A}}\\T) = {L.num(lhs)},\\quad \\det(\\mathbf{{A}}) = {L.num(rhs)}")]
    elif kind == "mult":
        stmt, truth = "\\det(\\mathbf{AB}) = \\det(\\mathbf{A})\\det(\\mathbf{B})", (a * b).det() == da * db
        sol = [Step(text="Tính hai vế:", latex=f"\\det(\\mathbf{{AB}}) = {L.num((a * b).det())},\\quad \\det(\\mathbf{{A}})\\det(\\mathbf{{B}}) = {L.num(da)}\\cdot {N(db)} = {L.num(da * db)}")]
    elif kind == "add":
        stmt, truth = "\\det(\\mathbf{A}+\\mathbf{B}) = \\det(\\mathbf{A}) + \\det(\\mathbf{B})", (a + b).det() == da + db
        sol = [Step(text="Tính hai vế:", latex=f"\\det(\\mathbf{{A}}+\\mathbf{{B}}) = {L.num((a + b).det())},\\quad \\det(\\mathbf{{A}})+\\det(\\mathbf{{B}}) = {L.num(da + db)}")]
    elif kind == "scalar-wrong":
        stmt, truth = f"\\det({L.num(k)}\\mathbf{{A}}) = {L.num(k)}\\det(\\mathbf{{A}})", (k * a).det() == k * da
        sol = [Step(text=f"Với ma trận ${n}\\times {n}$, mỗi hàng nhân với ${L.num(k)}$ nên định thức nhân với ${L.num(k)}^{n}$:",
                    latex=f"\\det({L.num(k)}\\mathbf{{A}}) = {L.num((k * a).det())}, \\quad {L.num(k)}\\det(\\mathbf{{A}}) = {L.num(k * da)}")]
    elif kind == "scalar-ok":
        stmt, truth = f"\\det({L.num(k)}\\mathbf{{A}}) = {L.num(k ** n)}\\det(\\mathbf{{A}})", (k * a).det() == k ** n * da
        sol = [Step(text="Tính hai vế:", latex=f"\\det({L.num(k)}\\mathbf{{A}}) = {L.num((k * a).det())},\\quad {L.num(k ** n)}\\det(\\mathbf{{A}}) = {L.num(k ** n * da)}")]
    elif kind == "swap":
        s = a.copy()
        s.row_swap(0, 1)
        stmt, truth = "\\det(\\mathbf{A}') = \\det(\\mathbf{A})", s.det() == da
        pre = "Gọi $\\mathbf{A}'$ là ma trận có được từ $\\mathbf{A}$ bằng cách đổi chỗ hai hàng đầu. "
        sol = [Step(text="Đổi chỗ hai hàng:", latex=f"\\det(\\mathbf{{A}}') = {L.num(s.det())},\\quad \\det(\\mathbf{{A}}) = {L.num(da)}")]
    else:
        if da == 0:
            return _claims(rng)
        stmt, truth = "\\det(\\mathbf{A}^{-1}) = \\dfrac{1}{\\det(\\mathbf{A})}", a.inv().det() == 1 / da
        sol = [Step(text="Tính hai vế:", latex=f"\\det(\\mathbf{{A}}^{{-1}}) = {L.num(a.inv().det())},\\quad 1/\\det(\\mathbf{{A}}) = {L.num(1 / da)}")]
    # đối chiếu NumPy cho vế kiểm tra
    check(np.isclose(np.linalg.det(to_np(a)), float(da)), "numpy det A")
    return stmt, truth, sol, a, b, pre


def gen_properties(rng: random.Random, level: int, seed: int) -> Exercise:
    if level in (1, 2):
        stmt, truth, sol, a, b, pre = _claims(rng)
        disp = [L.mat(a, B("A")), L.mat(b, B("B"))] if level == 2 else [L.mat(a, B("A")), L.mat(b, B("B"))]
        return build(
            type_="det-properties", lesson=LESSON, level=level, seed=seed,
            prompt=Prompt(text=f"{pre}Với $\\mathbf{{A}}, \\mathbf{{B}}$ dưới đây, khẳng định ${stmt}$ đúng hay sai?", display=disp),
            answer=bool_ans(truth),
            hints=["Tính cả hai vế bằng số rồi so sánh.", "Nhớ các tính chất: $\\det(\\mathbf{A}\\T)$, $\\det(\\mathbf{AB})$, $\\det(k\\mathbf{A})$, đổi hàng."],
            steps=sol, verified=["sympy", "numpy"], tags=["claim"],
        )
    if level == 3:  # tính det(kA), det(A^T), det(A^2), det(A^-1) cho A thật
        n = rng.choice([2, 3])
        a = nontrivial_matrix(rng, n, n, -4, 4)
        while a.det() == 0:
            a = nontrivial_matrix(rng, n, n, -4, 4)
        k = Rational(rng.choice([2, 3, -2]))
        which = rng.choice(["k", "T", "sq", "inv"])
        d = a.det()
        if which == "k":
            expr, val, why = f"\\det({L.num(k)}\\mathbf{{A}})", (k * a).det(), f"{L.num(k)}^{{{n}}}\\det(\\mathbf{{A}})"
            ans_f = k ** n * d
        elif which == "T":
            expr, val, why = "\\det(\\mathbf{A}\\T)", a.T.det(), "\\det(\\mathbf{A})"
            ans_f = d
        elif which == "sq":
            expr, val, why = "\\det(\\mathbf{A}^2)", (a * a).det(), "\\det(\\mathbf{A})^2"
            ans_f = d ** 2
        else:
            expr, val, why = "\\det(\\mathbf{A}^{-1})", a.inv().det(), "1/\\det(\\mathbf{A})"
            ans_f = 1 / d
        check(val == ans_f, "tính chất det")
        check(np.isclose(float(val), float(np.linalg.det(to_np(k * a if which == 'k' else a.T if which == 'T' else a * a if which == 'sq' else a.inv())))), "numpy")
        return build(
            type_="det-properties", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text=f"Tính ${expr}$ (gợi ý: dùng tính chất của định thức).", display=[L.mat(a, B("A"))]),
            answer=num_ans(val),
            hints=["Tính $\\det(\\mathbf{A})$ trước.", f"Dùng tính chất: ${expr} = {why}$."],
            steps=[Step(text="Định thức của $\\mathbf{A}$:", latex=f"\\det(\\mathbf{{A}}) = {L.num(d)}"),
                   Step(text="Áp dụng tính chất:", latex=f"{expr} = {why} = {L.num(val)}")],
            verified=["sympy", "numpy"], tags=[size_tag(a), which],
        )
    if level == 4:  # chỉ cho det A, det B
        n = rng.choice([2, 3])
        a, b = nontrivial_matrix(rng, n, n, -3, 3), nontrivial_matrix(rng, n, n, -3, 3)
        while a.det() == 0 or b.det() == 0:
            a, b = nontrivial_matrix(rng, n, n, -3, 3), nontrivial_matrix(rng, n, n, -3, 3)
        da, db = a.det(), b.det()
        k = Rational(rng.choice([2, 3, -2]))
        val = (k * a.T * b.inv()).det()
        formula = k ** n * da / db
        check(val == formula, "det(kA^T B^-1)")
        np_close(Matrix([[val]]), np.array([[np.linalg.det(float(k) * to_np(a).T @ np.linalg.inv(to_np(b)))]]))
        return build(
            type_="det-properties", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"Cho $\\mathbf{{A}}, \\mathbf{{B}}$ là hai ma trận ${n}\\times {n}$ với $\\det(\\mathbf{{A}}) = {L.num(da)}$ và "
                               f"$\\det(\\mathbf{{B}}) = {L.num(db)}$. Tính $\\det({L.num(k)}\\mathbf{{A}}\\T\\mathbf{{B}}^{{-1}})$."),
            answer=num_ans(val),
            hints=["Tách từng thừa số: $\\det(k\\mathbf{M}) = k^n\\det(\\mathbf{M})$, $\\det(\\mathbf{XY}) = \\det\\mathbf{X}\\det\\mathbf{Y}$.",
                   "$\\det(\\mathbf{A}\\T) = \\det(\\mathbf{A})$ và $\\det(\\mathbf{B}^{-1}) = 1/\\det(\\mathbf{B})$."],
            steps=[Step(text="Áp dụng lần lượt các tính chất:",
                        latex=f"\\det({L.num(k)}\\mathbf{{A}}\\T\\mathbf{{B}}^{{-1}}) = {L.num(k)}^{{{n}}}\\cdot \\det(\\mathbf{{A}}) \\cdot \\frac{{1}}{{\\det(\\mathbf{{B}})}} = "
                              f"{L.num(k ** n)}\\cdot {N(da)}\\cdot \\frac{{1}}{{{N(db)}}} = {L.num(val)}")],
            verified=["sympy", "numpy", "property"], tags=[f"{n}x{n}"],
        )
    # level 5: đa cộng tuyến, det(X^T X)
    n = 3
    for _ in range(300):
        x = nontrivial_matrix(rng, 3, 3, -3, 3)
        if rng.random() < 0.6:
            c = rng.sample([-1, 1, 2, -2], 2)
            x[:, 2] = c[0] * x[:, 0] + c[1] * x[:, 1]
            if any(all(v == 0 for v in x[:, j]) for j in range(3)):
                continue
        break
    g = x.T * x
    val = g.det()
    verify_det(g, val)
    check(np.isclose(np.linalg.det(to_np(x).T @ to_np(x)), float(val), atol=1e-8), "numpy gram det")
    dep = x.rank() < 3
    return build(
        type_="det-properties", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text="Ma trận dữ liệu $\\mathbf{X}$ ($3$ mẫu, $3$ đặc trưng) được dùng để lập ma trận $\\mathbf{X}\\T\\mathbf{X}$ "
                           "trong hồi quy tuyến tính. Tính $\\det(\\mathbf{X}\\T\\mathbf{X})$.",
                      display=[L.mat(x, B("X"))]),
        answer=num_ans(val),
        hints=["$\\det(\\mathbf{X}\\T\\mathbf{X}) = \\det(\\mathbf{X}\\T)\\det(\\mathbf{X}) = \\det(\\mathbf{X})^2$ vì $\\mathbf{X}$ vuông.",
               "Nếu các cột của $\\mathbf{X}$ phụ thuộc tuyến tính (đa cộng tuyến) thì $\\det(\\mathbf{X}) = 0$."],
        steps=[Step(text="Tính $\\det(\\mathbf{X})$:", latex=f"\\det(\\mathbf{{X}}) = {L.num(x.det())}"),
               Step(text="Vì $\\mathbf{X}$ vuông, $\\det(\\mathbf{X}\\T\\mathbf{X}) = \\det(\\mathbf{X})^2$:", latex=f"{N(x.det())}^2 = {L.num(val)}")]
              + ([Step(text="Các cột phụ thuộc tuyến tính nên định thức bằng $0$: $\\mathbf{X}\\T\\mathbf{X}$ không khả nghịch, bài toán hồi quy không có nghiệm duy nhất.")] if dep else []),
        verified=["sympy", "numpy", "laplace", "gauss"], tags=["ml", "dependent" if dep else "independent"],
    )


# ---------------------------------------------------------------- det-param
def _sym_mat(rows: list[list[str]]) -> str:
    return L.expr_mat(rows)


def gen_param(rng: random.Random, level: int, seed: int) -> Exercise:
    m_sym = Symbol("m")
    n = rng.choice([2, 3]) if level == 3 else 3
    for _ in range(300):
        a = nontrivial_matrix(rng, n, n, -4, 4)
        if level == 3:
            i, j = rng.randrange(n), rng.randrange(n)
            cells = [(i, j)]
        else:  # m ở hai ô cùng hàng: định thức vẫn bậc nhất theo m
            i = rng.randrange(n)
            js = rng.sample(range(n), 2)
            cells = [(i, js[0]), (i, js[1])]
        sym = a.copy()
        for (r, c) in cells:
            sym[r, c] = m_sym * rng.choice([1, 1, -1, 2]) if level == 4 else m_sym
        dm = sym.det().expand()
        alpha, beta = dm.coeff(m_sym, 1), dm.coeff(m_sym, 0)
        if dm.coeff(m_sym, 2) == 0 and alpha != 0:
            break
    sol = solve(dm, m_sym)
    check(len(sol) == 1, "det-param: một nghiệm")
    mv = Rational(sol[0])
    sub = sym.subs(m_sym, mv)
    check(sub.det() == 0 and np.isclose(np.linalg.det(to_np(sub)), 0, atol=1e-8), "det(m)=0")
    # kiểm: ngoài m thì khả nghịch
    check(sym.subs(m_sym, mv + 1).det() != 0, "det khác 0 ở m khác")
    rows = [[(L.num(sym[r, c]) if not sym[r, c].has(m_sym) else (("-" if sym[r, c].coeff(m_sym) < 0 else "") + (str(abs(sym[r, c].coeff(m_sym))) if abs(sym[r, c].coeff(m_sym)) != 1 else "") + "m"))
             for c in range(n)] for r in range(n)]
    lin = f"{L.num(alpha)}m" + (f" + {L.num(beta)}" if beta > 0 else (f" - {L.num(-beta)}" if beta < 0 else ""))
    return build(
        type_="det-param", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tìm giá trị của $m$ để ma trận $\\mathbf{A}$ không khả nghịch (tức $\\det(\\mathbf{A}) = 0$).",
                      display=["\\mathbf{A} = " + _sym_mat(rows)]),
        answer=num_ans(mv),
        hints=["Ma trận không khả nghịch khi và chỉ khi định thức bằng $0$.", "Tính $\\det(\\mathbf{A})$ theo $m$: nó là biểu thức bậc nhất."],
        steps=[Step(text="Định thức theo $m$ (khai triển theo hàng hoặc Sarrus):", latex=f"\\det(\\mathbf{{A}}) = {lin}"),
               Step(text="Giải $\\det(\\mathbf{A}) = 0$:", latex=f"{lin} = 0 \\;\\Rightarrow\\; m = {L.num(mv)}")],
        verified=["sympy", "numpy", "substitution"], tags=[f"{n}x{n}", "param"],
    )


SPECS = [
    TopicSpec("det", LESSON, (1, 2, 3, 4), gen_det),
    TopicSpec("det-laplace", LESSON, (2, 3, 4), gen_laplace),
    TopicSpec("det-rowops", LESSON, (2, 3, 4), gen_rowops),
    TopicSpec("det-properties", LESSON, (1, 2, 3, 4, 5), gen_properties),
    TopicSpec("det-param", LESSON, (3, 4), gen_param),
]
