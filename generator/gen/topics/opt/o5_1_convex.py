"""Bài 5.1 Hàm lồi: convex-1d, convex-hessian, quad-min."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sympy import Matrix, Rational, Symbol, hessian, symbols

from ._o import (lf, F, NUM_HINT, VEC_HINT, Answer, L, Prompt, Step, TopicSpec, check, choice_exercise, fs, ft, num_ans, ob, pick, random, vec_ans, vt)

L51 = "5.1"
STRICT, CONV, CONC, NEITHER = "Lồi chặt", "Lồi nhưng không lồi chặt", "Lõm", "Không lồi cũng không lõm"
ALL_LABELS = [STRICT, CONV, CONC, NEITHER]
GRID = [F(k, 10) for k in range(-20, 21)]


def _classify_1d(f) -> str:
    """Phân loại bằng bất đẳng thức Jensen với điểm giữa, thử chính xác trên lưới (Fraction)."""
    conv = strict = conc = True
    strict_conc = True
    for u, v in combinations(GRID, 2):
        m = f((u + v) / 2)
        avg = (f(u) + f(v)) / 2
        if m > avg:
            conv = False
        if m >= avg:
            strict = False
        if m < avg:
            conc = False
    if conv:
        return STRICT if strict else CONV
    return CONC if conc else NEITHER


def _abs(x):
    return abs(x)


# ------------------------------------------------------------------ convex-1d
def gen_convex_1d(rng: random.Random, level: int, seed: int):
    kinds = {1: ["quad"], 2: ["quad", "abs", "relu"], 3: ["quartic", "abs", "relu", "quad"], 4: ["quartic", "cubic", "absquad", "abs"], 5: ["quartic", "cubic", "absquad", "relu"]}[level]
    kind = rng.choice(kinds)
    if kind == "quad":
        a, b, c = pick(rng, -6, 6), pick(rng, -8, 8, False), pick(rng, -8, 8, False)
        f = lambda x: a * x * x + b * x + c
        tex = lf([a, b, c], ["x^{2}", "x", "1"])
        label = STRICT if a > 0 else CONC
        why = f"$f''(x) = {2 * a}$ không đổi dấu: " + ("dương nên $f$ lồi chặt." if a > 0 else "âm nên $f$ lõm.")
    elif kind == "quartic":
        a, b, c = pick(rng, -3, 3), pick(rng, -6, 6, False), pick(rng, -5, 5, False)
        f = lambda x: a * x**4 + b * x * x + c
        tex = lf([a, b, c], ["x^{4}", "x^{2}", "1"])
        fpp = lf([12 * a, 2 * b], ["x^{2}", "1"])
        if a > 0 and b > 0:
            label, why = STRICT, f"$f''(x) = {fpp} > 0$ với mọi $x \\neq 0$ (và $f''(0) = {2 * b} > 0$), nên $f$ lồi chặt."
        elif a < 0 and b < 0:
            label, why = CONC, f"$f''(x) = {fpp} < 0$ với mọi $x$, nên $f$ lõm."
        else:
            label, why = NEITHER, f"$f''(x) = {fpp}$ đổi dấu (âm gần $x = 0$ nếu $b$ ngược dấu $a$ hoặc ngược lại ở xa), nên $f$ không lồi cũng không lõm."
            if a > 0 and b < 0:
                why = f"$f''(x) = {fpp}$ âm khi $x^2 < {ft(F(-2 * b, 12 * a))}$ và dương ngoài khoảng đó, nên $f$ không lồi cũng không lõm."
            else:
                why = f"$f''(x) = {fpp}$ dương gần $x = 0$ và âm ở xa, nên $f$ không lồi cũng không lõm."
    elif kind == "abs":
        a, h, c = pick(rng, -5, 5), pick(rng, -3, 3, False), pick(rng, -5, 5, False)
        f = lambda x: a * abs(x - h) + c
        tex = lf([a], [f"|x {'-' if h > 0 else '+'} {abs(h)}|"]) + (f" {'+' if c > 0 else '-'} {abs(c)}")
        label = CONV if a > 0 else CONC
        why = "Giá trị tuyệt đối là hàm lồi nhưng gồm hai nửa đường thẳng, không lồi chặt; " + ("hệ số dương giữ nguyên tính lồi." if a > 0 else "hệ số âm đổi thành hàm lõm.")
    elif kind == "relu":
        a, b = pick(rng, -4, 4), pick(rng, -5, 5, False)
        f = lambda x: max(0, a * x + b)
        tex = f"\\max(0,\\ {lf([a, b], ['x', '1'])})"
        label, why = CONV, "$\\max$ của hai hàm tuyến tính (lồi) là hàm lồi; nó gồm hai đoạn thẳng nên không lồi chặt."
    elif kind == "cubic":
        a, b, c = pick(rng, -3, 3), pick(rng, -4, 4, False), pick(rng, -5, 5, False)
        f = lambda x: a * x**3 + b * x * x + c * x
        tex = lf([a, b, c], ["x^{3}", "x^{2}", "x"])
        label, why = NEITHER, f"$f''(x) = {lf([6 * a, 2 * b], ['x', '1'])}$ là hàm bậc nhất, đổi dấu tại một điểm, nên $f$ không lồi cũng không lõm."
    else:  # absquad: a|x| + b x^2
        a, b = pick(rng, -5, 5), pick(rng, -3, 3)
        f = lambda x: a * abs(x) + b * x * x
        tex = lf([a, b], ["|x|", "x^{2}"])
        if a > 0 and b > 0:
            label, why = STRICT, "Tổng của hàm lồi $a|x|$ và hàm lồi chặt $bx^2$ là hàm lồi chặt."
        elif a < 0 and b < 0:
            label, why = CONC, "Tổng của hai hàm lõm là hàm lõm."
        else:
            label, why = NEITHER, "Hai số hạng có tính cong ngược chiều nhau; kiểm tra $f(-t), f(0), f(t)$ cho thấy bất đẳng thức Jensen bị vi phạm theo cả hai hướng."
    got = _classify_1d(f)
    if got != label:  # thiết kế và kiểm chứng trên lưới không khớp: loại đề (ví dụ vi phạm quá nhỏ để lưới thấy)
        return gen_convex_1d(random.Random(rng.random()), level, seed)
    choices, ans = choice_exercise(rng, label, [x for x in ALL_LABELS if x != label])
    return ob(type_="convex-1d", lesson=L51, level=level, seed=seed,
              prompt=Prompt(text="Hàm số $f : \\mathbb{R} \\to \\mathbb{R}$ dưới đây thuộc loại nào?", display=[f"f(x) = {tex}"]),
              answer=ans, choices=choices,
              hints=["Hàm lồi: đoạn thẳng nối hai điểm bất kỳ trên đồ thị nằm trên đồ thị. Lồi chặt: nằm hẳn trên (không có đoạn thẳng nào nằm trên đồ thị).",
                     "Nếu $f$ hai lần khả vi: $f'' \\ge 0$ ở mọi nơi thì lồi, $f'' > 0$ thì lồi chặt, $f'' \\le 0$ thì lõm."],
              steps=[Step(text=why)], verified=["jensen-grid-exact"], tags=[kind])


# ---------------------------------------------------------------- convex-hessian
def _eig_label(h: Matrix) -> str:
    npv = np.linalg.eigvalsh(np.array(h.tolist(), dtype=float))
    if h.rows <= 2:  # trị riêng dạng căn thức chính xác; ma trận 3x3 có thể ra số phức dạng căn bậc ba nên chỉ dùng NumPy
        ev = [complex(v).real for v in h.eigenvals(multiple=True)]
        check(np.allclose(sorted(ev), sorted(npv), atol=1e-8), "trị riêng SymPy ≠ NumPy")
    pos, neg, zero = sum(v > 1e-9 for v in npv), sum(v < -1e-9 for v in npv), sum(abs(v) <= 1e-9 for v in npv)
    if neg == 0:
        return STRICT if zero == 0 else CONV
    if pos == 0:
        return CONC
    return NEITHER


def _minors_label(h: Matrix) -> str:
    """Kiểm chứng thứ hai bằng định thức: Sylvester (chính) và mọi định thức con chính (nửa xác định)."""
    n = h.rows
    lead = [h[:k, :k].det() for k in range(1, n + 1)]
    if all(d > 0 for d in lead):
        return STRICT
    if all((-1) ** k * d > 0 for k, d in enumerate(lead, 1)):
        return CONC
    from itertools import combinations as comb

    psd = all(h.extract(list(s), list(s)).det() >= 0 for k in range(1, n + 1) for s in comb(range(n), k))
    nsd = all((-1) ** k * h.extract(list(s), list(s)).det() >= 0 for k in range(1, n + 1) for s in comb(range(n), k))
    return CONV if psd else CONC if nsd else NEITHER


def gen_convex_hessian(rng: random.Random, level: int, seed: int):
    target = rng.choice(ALL_LABELS)
    x, y, z = symbols("x y z")
    while True:
        if level == 5:
            vs = [x, y, z]
            diag = [pick(rng, -3, 3) for _ in range(3)]
            off = {(i, j): rng.randint(-3, 3) for i, j in [(0, 1), (0, 2), (1, 2)]}
            f = sum(diag[i] * vs[i] ** 2 for i in range(3)) + sum(c * vs[i] * vs[j] for (i, j), c in off.items())
        else:
            vs = [x, y]
            if level == 1:
                a, b, c = pick(rng, -4, 4), 0, pick(rng, -4, 4)
            elif level == 2:
                a, b, c = pick(rng, -4, 4), rng.randint(-2, 2), pick(rng, -4, 4)
            elif level == 3 and rng.random() < 0.75:
                a, b, c = pick(rng, -5, 5), rng.randint(-6, 6), pick(rng, -5, 5)
            else:  # có nhiều ca nửa xác định: f = m (p x + q y)^2
                if level == 3 or rng.random() < 0.6:
                    m, p, q = pick(rng, -3, 3), pick(rng, -3, 3), pick(rng, -3, 3)
                    a, b, c = m * p * p, 2 * m * p * q, m * q * q
                else:
                    a, b, c = pick(rng, -6, 6), rng.randint(-8, 8), pick(rng, -6, 6)
            f = a * x**2 + b * x * y + c * y**2
        d = [rng.randint(-5, 5) for _ in vs]
        lin = sum(di * v for di, v in zip(d, vs))
        fe = f + lin
        h = hessian(fe, vs)
        if h.is_zero_matrix:
            continue
        label = _eig_label(h)
        check(label == _minors_label(h), "nhãn trị riêng ≠ nhãn định thức")
        if label == target or rng.random() < 0.05:
            break
    n = len(vs)
    if n == 2:
        coeffs = [fe.coeff(x, 2), fe.coeff(x, 1).coeff(y, 1), fe.coeff(y, 2), fe.coeff(x, 1).subs(y, 0), fe.coeff(y, 1).subs(x, 0)]
        tex = lf(coeffs, ["x^{2}", "xy", "y^{2}", "x", "y"])
        fname = "f(x, y)"
    else:
        cs = [fe.coeff(x, 2), fe.coeff(y, 2), fe.coeff(z, 2), fe.coeff(x, 1).coeff(y, 1), fe.coeff(x, 1).coeff(z, 1), fe.coeff(y, 1).coeff(z, 1)]
        tex = lf(cs + [fe.coeff(x, 1).subs({y: 0, z: 0}), fe.coeff(y, 1).subs({x: 0, z: 0}), fe.coeff(z, 1).subs({x: 0, y: 0})],
                            ["x^{2}", "y^{2}", "z^{2}", "xy", "xz", "yz", "x", "y", "z"])
        fname = "f(x, y, z)"
    has_zero = any(abs(v) <= 1e-9 for v in np.linalg.eigvalsh(np.array(h.tolist(), dtype=float)))
    why_lab = {STRICT: "mọi trị riêng dương (ma trận xác định dương)", CONV: "mọi trị riêng $\\ge 0$ và có trị riêng bằng $0$ (nửa xác định dương)",
               CONC: ("mọi trị riêng $\\le 0$ và có trị riêng bằng $0$ (nửa xác định âm), nên $-f$ lồi và $f$ lõm" if has_zero else "mọi trị riêng âm (ma trận xác định âm)"),
               NEITHER: "có cả trị riêng dương lẫn âm (điểm yên ngựa)"}
    if n == 2:
        from sympy import latex as sym_latex

        ev = sorted(h.eigenvals(multiple=True), key=lambda v: float(v))
        info = "Các trị riêng của $H$: $" + ", ".join(sym_latex(v) for v in ev) + "$"
    else:
        lead = [h[:k, :k].det() for k in range(1, n + 1)]
        npv = np.linalg.eigvalsh(np.array(h.tolist(), dtype=float))
        info = ("Các định thức con chính dẫn đầu của $H$ là $" + ", ".join(str(d) for d in lead) + "$; dấu của các trị riêng: "
                f"{int(sum(v > 1e-9 for v in npv))} dương, {int(sum(v < -1e-9 for v in npv))} âm, {int(sum(abs(v) <= 1e-9 for v in npv))} bằng $0$")
    choices, ans = choice_exercise(rng, label, [lb for lb in ALL_LABELS if lb != label])
    return ob(type_="convex-hessian", lesson=L51, level=level, seed=seed,
              prompt=Prompt(text="Xét hàm nhiều biến sau. Dựa vào ma trận Hessian, $f$ thuộc loại nào?", display=[f"{fname} = {tex}"]),
              answer=ans, choices=choices,
              hints=["Hessian của hàm bậc hai là ma trận hằng số: $H_{ij} = \\partial^2 f / \\partial x_i \\partial x_j$ (số hạng $x^2$ cho phần tử đường chéo gấp đôi hệ số).",
                     "Xác định dương $\\Rightarrow$ lồi chặt; nửa xác định dương $\\Rightarrow$ lồi; xác định âm $\\Rightarrow$ lõm; có trị riêng trái dấu $\\Rightarrow$ không lồi cũng không lõm.",
                     "Với ma trận $2\\times 2$: so dấu $H_{11}$ và $\\det H$."],
              steps=[Step(text="Tính Hessian:", latex="H = " + L.mat(h)), Step(text=f"{info}, tức là {why_lab[label]}."),
                     Step(text=f"Kết luận: **{label}**.")],
              verified=["sympy-eigenvalues", "numpy-eigvalsh", "principal-minors"], tags=[f"{n}var"])


# ------------------------------------------------------------------- quad-min
def gen_quad_min(rng: random.Random, level: int, seed: int):
    if level in (1, 2):
        while True:
            a, b, c = pick(rng, 1, 6), pick(rng, -12, 12, False), pick(rng, -9, 9, False)
            if level == 2:
                a, c = pick(rng, 1, 9), pick(rng, -20, 20)
            xs = F(-b, 2 * a)
            if level == 1 and xs.denominator <= 2 or level == 2:
                break
        f = lambda t: a * t * t + b * t + c
        fstar = f(xs)
        for d in (F(1, 7), F(-1, 3), F(2), F(-5, 11)):  # mọi điểm lân cận đều có giá trị lớn hơn
            check(f(xs + d) > fstar, "x* không phải cực tiểu")
        check(2 * a * xs + b == 0, "f'(x*) ≠ 0")
        tex = lf([a, b, c], ["x^{2}", "x", "1"])
        return ob(type_="quad-min", lesson=L51, level=level, seed=seed,
                  prompt=Prompt(text="Hàm lồi chặt $f$ dưới đây đạt cực tiểu tại duy nhất một điểm $x^\\star$. Tìm $x^\\star$. " + NUM_HINT, display=[f"f(x) = {tex}"]),
                  answer=num_ans(xs), hints=["Hàm lồi khả vi: cực tiểu toàn cục tại nơi $f'(x) = 0$.", "$f'(x) = 2ax + b$."],
                  steps=[Step(text="Giải $f'(x) = 0$:", latex=f"{lf([2 * a, b], ['x', '1'])} = 0 \\;\\Rightarrow\\; x^\\star = {ft(xs)}")],
                  verified=["derivative-zero", "neighbour-points"], tags=["1d"])
    n = 2 if level in (3, 4) else 3
    x, y, z = symbols("x y z")
    vs = [x, y, z][:n]
    while True:
        diag = [pick(rng, 1, 4) for _ in range(n)]
        off = {(i, j): rng.randint(-2, 2) for i in range(n) for j in range(i + 1, n)} if level != 3 else {(0, 1): 0}
        q = [[0] * n for _ in range(n)]
        f = 0
        for i in range(n):
            q[i][i] = 2 * diag[i]
            f += diag[i] * vs[i] ** 2
        for (i, j), c in off.items():
            q[i][j] = q[j][i] = c
            f += c * vs[i] * vs[j]
        qm = Matrix(q)
        if _minors_label(qm) != STRICT:
            continue
        d = [rng.randint(-8, 8) for _ in range(n)]
        fe = f + sum(di * v for di, v in zip(d, vs))
        xs = qm.LUsolve(Matrix([-v for v in d]))  # Q x = -d
        xf = [F(int(v.p), int(v.q)) for v in xs]
        if all(abs(v) <= 12 and v.denominator <= 24 for v in xf):
            break
    grad = Matrix([fe.diff(v) for v in vs]).subs(dict(zip(vs, xs)))
    check(grad.is_zero_matrix, "gradient tại x* ≠ 0")
    npx = np.linalg.solve(np.array(q, dtype=float), -np.array(d, dtype=float))
    check(np.allclose(npx, [float(v) for v in xf], atol=1e-9), "NumPy solve ≠ SymPy")
    fx = lambda p: fe.subs(dict(zip(vs, p)))
    base = fx(xs)
    for dv in ([Rational(1, 3)] + [0] * (n - 1), [0] * (n - 1) + [Rational(-1, 2)], [Rational(1, 5)] * n):
        check(fx([a + b for a, b in zip(xs, dv)]) > base, "x* không phải cực tiểu")
    if n == 2:
        coeffs = [fe.coeff(x, 2), fe.coeff(x, 1).coeff(y, 1), fe.coeff(y, 2), fe.coeff(x, 1).subs(y, 0), fe.coeff(y, 1).subs(x, 0)]
        tex = lf(coeffs, ["x^{2}", "xy", "y^{2}", "x", "y"])
        head = "f(x, y)"
    else:
        cs = [fe.coeff(x, 2), fe.coeff(y, 2), fe.coeff(z, 2), fe.coeff(x, 1).coeff(y, 1), fe.coeff(x, 1).coeff(z, 1), fe.coeff(y, 1).coeff(z, 1),
              fe.coeff(x, 1).subs({y: 0, z: 0}), fe.coeff(y, 1).subs({x: 0, z: 0}), fe.coeff(z, 1).subs({x: 0, y: 0})]
        tex = lf(cs, ["x^{2}", "y^{2}", "z^{2}", "xy", "xz", "yz", "x", "y", "z"])
        head = "f(x, y, z)"
    return ob(type_="quad-min", lesson=L51, level=level, seed=seed,
              prompt=Prompt(text="Hàm lồi chặt $f$ dưới đây có đúng một điểm cực tiểu $\\mathbf{x}^\\star$. Tìm $\\mathbf{x}^\\star$. " + VEC_HINT, display=[f"{head} = {tex}"]),
              answer=vec_ans(xf),
              hints=["Hàm lồi khả vi: cực tiểu toàn cục tại nơi $\\nabla f = \\mathbf{0}$.", "$\\nabla f = H\\mathbf{x} + \\mathbf{d}$ với $H$ là Hessian (hằng số), $\\mathbf{d}$ là vector hệ số của các số hạng bậc nhất.",
                     "Giải hệ $H\\mathbf{x} = -\\mathbf{d}$ bằng khử Gauss."],
              steps=[Step(text="Hessian và hệ $H\\mathbf{x} = -\\mathbf{d}$:", latex="H = " + L.mat(qm) + ",\\quad \\mathbf{d} = " + L.vec_inline(Matrix(d))),
                     Step(text="Nghiệm:", latex="\\mathbf{x}^\\star = " + vt(xf))],
              verified=["sympy-solve", "numpy-solve", "gradient-zero", "neighbour-points"], tags=[f"{n}var"])


SPECS = [
    TopicSpec("convex-1d", L51, (1, 2, 3, 4, 5), gen_convex_1d),
    TopicSpec("convex-hessian", L51, (1, 2, 3, 4, 5), gen_convex_hessian),
    TopicSpec("quad-min", L51, (1, 2, 3, 4, 5), gen_quad_min),
]
