"""Bài 3.3–3.7: partial-deriv, gradient, jacobian, hessian, vec-deriv, graph-forward, graph-backward."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational, Symbol, diff, exp, hessian, log, simplify, solve, sqrt, symbols

from ...core.nice import nontrivial_matrix
from ...core.verify import check, np_close, to_np
from ._c import (Answer, Check, L, Prompt, Step, TopicSpec, cb, expr_answer, fd_check, num, random, tex, x, y, z)

L33, L34, L35, L36, L37 = "3.3", "3.4", "3.5", "3.6", "3.7"
B = L.bold


def _c(rng, lo=-5, hi=5):
    while True:
        v = rng.randint(lo, hi)
        if v:
            return v


def _vec_ans(vals) -> Answer:
    return Answer(kind="vector", rows=len(vals), cols=1, entries=[[str(Rational(v))] for v in vals])


def _mat_ans(m: Matrix) -> Answer:
    return Answer(kind="matrix", rows=m.rows, cols=m.cols, entries=[[str(Rational(m[i, j])) for j in range(m.cols)] for i in range(m.rows)])


def _fd_grad_check(f, vars_, pt, grad):
    """Sai phân hữu hạn của f tại điểm pt (dạng float) so với gradient chính xác."""
    from sympy import lambdify

    ff = lambdify(vars_, f, "numpy")
    h = 1e-6
    for i in range(len(vars_)):
        hi, lo = [float(p) for p in pt], [float(p) for p in pt]
        hi[i] += h
        lo[i] -= h
        num_ = (float(ff(*hi)) - float(ff(*lo))) / (2 * h)
        check(abs(num_ - float(grad[i])) <= 1e-5 * max(1.0, abs(float(grad[i]))), f"gradient lệch sai phân ở biến {i}")


# --------------------------------------------------------------- partial-deriv
def gen_partial(rng: random.Random, level: int, seed: int):
    a, b, c = _c(rng, -5, 5), _c(rng, -5, 5), _c(rng, 1, 4)
    if level == 1:
        f = a * x ** rng.randint(1, 3) * y ** rng.randint(1, 3) + b * x ** rng.randint(1, 3) + c * y ** rng.randint(1, 3) + _c(rng, -5, 5)
    elif level == 2:
        f = rng.choice([x**2 * exp(y), a * x * y**2 + b * exp(x) * y, x * y * (x + c * y), (a * x + y) ** 3, x**2 * y + y**3 * x])
    elif level == 3:
        f = rng.choice([exp(x * y) * c, log(x**2 + c * y**2), sqrt(x**2 + y**2 + c), x * exp(-c * y**2), (x + y) ** 2 * exp(c * x), exp(a * x + b * y), log(c * x + y**2), x * y * exp(a * x) + c * y])
    else:
        f = rng.choice([x**2 * y**3 + a * x * y, exp(x * y) + c * x, b * x * log(y) + y * log(x), (x**2 + y**2) ** 2 * c, a * x**3 * y**2 + b * x * y**3,
                        exp(a * x + c * y) * x, x**2 * exp(y) + a * y**2 * exp(x), c * x * y * log(x + y) + a * x])
    var = rng.choice([x, y])
    second = level == 4
    if second:
        g = diff(f, x, y)
        text = "Tính đạo hàm riêng hỗn hợp $\\dfrac{\\partial^2 f}{\\partial x\\,\\partial y}$ (đạo hàm theo $y$ trước, rồi theo $x$)."
        fd_check(diff(f, y), g, [x, y])
    else:
        g = diff(f, var)
        text = f"Tính đạo hàm riêng $\\dfrac{{\\partial f}}{{\\partial {var}}}$ (coi biến còn lại là hằng số)."
        fd_check(f, g, [var] + [v for v in (x, y) if v != var])
    return cb(type_="partial-deriv", lesson=L33, level=level, seed=seed,
              prompt=Prompt(text=text + " Nhập biểu thức theo `x` và `y`.", display=[f"f(x, y) = {tex(f)}"]),
              answer=expr_answer(g, ["x", "y"]),
              hints=["Đạo hàm riêng theo một biến: coi mọi biến khác là hằng số rồi dùng các quy tắc đạo hàm một biến."],
              steps=[Step(text="Kết quả:", latex=tex(g))], verified=["sympy", "finite-difference"], tags=["second" if second else f"d{var}"])


# -------------------------------------------------------------------- gradient
def gen_gradient(rng: random.Random, level: int, seed: int):
    if level == 5:  # ML: mất mát bình phương của một mô hình hai trọng số
        x1, x2, yv = _c(rng, -4, 4), _c(rng, -4, 4), _c(rng, -6, 6)
        w1, w2 = symbols("w1 w2")
        f = Rational(1, 2) * (w1 * x1 + w2 * x2 - yv) ** 2
        pt = (_c(rng, -3, 3), _c(rng, -3, 3))
        g = [diff(f, v).subs({w1: pt[0], w2: pt[1]}) for v in (w1, w2)]
        _fd_grad_check(f, [w1, w2], pt, g)
        r = pt[0] * x1 + pt[1] * x2 - yv
        return cb(type_="gradient", lesson=L33, level=5, seed=seed,
                  prompt=Prompt(text=f"Mô hình tuyến tính $\\hat{{y}} = w_1 x_1 + w_2 x_2$ với mẫu $(x_1, x_2) = ({x1}, {x2})$ và nhãn $y = {yv}$. "
                                     f"Mất mát $L = \\frac{{1}}{{2}}(\\hat{{y}} - y)^2$. Tính $\\nabla L$ tại $(w_1, w_2) = ({pt[0]}, {pt[1]})$.", display=[]),
                  answer=_vec_ans(g),
                  hints=["Chain rule: $\\partial L/\\partial w_j = (\\hat{y} - y)\\,x_j$."],
                  steps=[Step(text="Sai số dự đoán:", latex=f"\\hat{{y}} - y = {pt[0]}\\cdot {x1} + {pt[1]}\\cdot {x2} - {yv} = {r}"),
                         Step(text="Gradient:", latex=f"\\nabla L = ({r}\\cdot {x1},\\; {r}\\cdot {x2}) = ({g[0]}, {g[1]})")],
                  verified=["sympy", "finite-difference"], tags=["ml"])
    nvars = 2 if level in (2, 4) else 3
    vs = [x, y, z][:nvars]
    if level == 2:
        f = _c(rng) * x**2 + _c(rng) * x * y + _c(rng) * y**2 + _c(rng) * x + _c(rng) * y
        pt = (_c(rng, -3, 3), _c(rng, -3, 3))
    elif level == 3:
        f = _c(rng) * x * y + _c(rng) * y * z**2 + _c(rng) * x**2 * z + _c(rng, 1, 4) * z
        pt = tuple(rng.randint(-2, 2) for _ in range(3))
    else:
        f = rng.choice([exp(x) * y**2 + _c(rng) * x * y, y * log(x + 1) * _c(rng, 1, 3) + x**2, exp(x) * y * _c(rng, 1, 3) + x * y, (x**2 + _c(rng) * y) * exp(x),
                        _c(rng) * y**3 * exp(x) + x * y, (y + _c(rng)) * log(x + 1) + x * y**2])
        pt = (0, rng.choice([0, 1, 2]))
    g = [diff(f, v).subs(dict(zip(vs, pt))) for v in vs]
    g = [Rational(simplify(t)) for t in g]
    _fd_grad_check(f, vs, pt, g)
    names = "(" + ", ".join(str(v) for v in vs) + ")"
    return cb(type_="gradient", lesson=L33, level=level, seed=seed,
              prompt=Prompt(text=f"Tính gradient $\\nabla f$ tại điểm ${names} = ({', '.join(str(p) for p in pt)})$. Nhập vector các đạo hàm riêng theo thứ tự biến.",
                            display=[f"f{names} = {tex(f)}"]),
              answer=_vec_ans(g),
              hints=["$\\nabla f = (\\partial f/\\partial x_1, \\dots, \\partial f/\\partial x_n)$: tính từng đạo hàm riêng rồi thay điểm."],
              steps=[Step(text="Các đạo hàm riêng:", latex=", ".join(f"\\frac{{\\partial f}}{{\\partial {v}}} = {tex(diff(f, v))}" for v in vs)),
                     Step(text="Thay điểm:", latex=f"\\nabla f = ({', '.join(num(t) for t in g)})")],
              verified=["sympy", "finite-difference"], tags=[f"{nvars}vars"])


# -------------------------------------------------------------------- jacobian
def gen_jacobian(rng: random.Random, level: int, seed: int):
    if level == 5:
        if rng.random() < 0.5:  # lớp tuyến tính: Jacobian là W
            m, n = rng.choice([(2, 3), (2, 2), (3, 2)])
            W = nontrivial_matrix(rng, m, n, -4, 4)
            bb = Matrix([_c(rng) for _ in range(m)])
            xs = symbols(f"x1:{n + 1}")
            F = W * Matrix(xs) + bb
            J = F.jacobian(Matrix(xs))
            check(J == W, "J = W")
            return cb(type_="jacobian", lesson=L34, level=5, seed=seed,
                      prompt=Prompt(text="Lớp tuyến tính $\\mathbf{F}(\\mathbf{x}) = \\mathbf{W}\\mathbf{x} + \\mathbf{b}$. Tính ma trận Jacobian $\\mathbf{J} = \\partial\\mathbf{F}/\\partial\\mathbf{x}$.",
                                    display=[L.mat(W, B("W")), L.mat(bb, B("b"))]),
                      answer=_mat_ans(W), hints=["Phần tử $J_{ij} = \\partial F_i / \\partial x_j$; với $\\mathbf{F} = \\mathbf{W}\\mathbf{x} + \\mathbf{b}$ đó chính là $W_{ij}$."],
                      steps=[Step(text="$\\partial F_i/\\partial x_j = W_{ij}$ nên $\\mathbf{J} = \\mathbf{W}$:", latex=L.mat(W, "\\mathbf{J}"))], verified=["sympy"], tags=["ml", "linear"])
        n = rng.choice([3, 4])
        zs = [rng.randint(-4, 4) for _ in range(n)]
        J = Matrix.diag(*[1 if v > 0 else 0 for v in zs])
        return cb(type_="jacobian", lesson=L34, level=5, seed=seed,
                  prompt=Prompt(text=f"Hàm kích hoạt ReLU áp dụng từng phần tử: $\\mathbf{{a}} = \\max(\\mathbf{{0}}, \\mathbf{{z}})$ với $\\mathbf{{z}} = ({', '.join(str(v) for v in zs)})\\T$. "
                                     f"Tính Jacobian $\\partial\\mathbf{{a}}/\\partial\\mathbf{{z}}$ (đạo hàm ReLU tại $0$ quy ước là $0$)."),
                  answer=_mat_ans(J), hints=["Hàm theo từng phần tử nên Jacobian là ma trận **đường chéo**; phần tử $i$ là $\\mathrm{ReLU}'(z_i)$."],
                  steps=[Step(text="$\\mathrm{ReLU}'(z) = 1$ nếu $z > 0$, ngược lại $0$:", latex=L.mat(J, "\\mathbf{J}"))], verified=["sympy"], tags=["ml", "relu"])
    n_in, n_out = {2: (2, 2), 3: (2, 3), 4: (3, 2)}[level]
    vs = [x, y, z][:n_in]
    pool = [x * y, x**2 + y, y**2 * x, x - y**2, x * y + y, 3 * x + y**2, x**2 - x * y, y * x**2 + 1]
    if n_in == 3:
        pool = [x * y * z, x**2 + y * z, x + y**2 + z, y * z - x, x * z + y**2, x**2 * y - z]
    fs = rng.sample(pool, n_out)
    F = Matrix(fs)
    pt = tuple(rng.randint(-2, 3) for _ in range(n_in))
    Jsym = F.jacobian(Matrix(vs))
    J = Jsym.subs(dict(zip(vs, pt)))
    # kiểm chứng sai phân từng cột
    from sympy import lambdify

    ff = lambdify(vs, list(F), "numpy")
    for j in range(n_in):
        hi, lo = [float(p) for p in pt], [float(p) for p in pt]
        hi[j] += 1e-6
        lo[j] -= 1e-6
        col = (np.array(ff(*hi), dtype=float) - np.array(ff(*lo), dtype=float)) / 2e-6
        check(np.allclose(col, np.array(J[:, j].tolist(), dtype=float).ravel(), atol=1e-4), "Jacobian lệch sai phân")
    names = "(" + ", ".join(str(v) for v in vs) + ")"
    return cb(type_="jacobian", lesson=L34, level=level, seed=seed,
              prompt=Prompt(text=f"Tính ma trận Jacobian $\\mathbf{{J}}$ của $\\mathbf{{F}}$ tại điểm ${names} = ({', '.join(str(p) for p in pt)})$ "
                                 f"(hàng $i$ là gradient của $F_i$; kích thước ${n_out}\\times {n_in}$).",
                            display=["\\mathbf{F}" + names + " = " + L.expr_mat([[tex(f)] for f in fs])]),
              answer=_mat_ans(J), hints=["$J_{ij} = \\partial F_i / \\partial x_j$: hàng $i$, cột $j$.", "Tính ma trận đạo hàm riêng theo ký hiệu rồi thay điểm."],
              steps=[Step(text="Ma trận đạo hàm riêng:", latex="\\mathbf{J} = " +L.expr_mat([[tex(Jsym[i, j]) for j in range(n_in)] for i in range(n_out)])),
                     Step(text="Thay điểm:", latex=L.mat(J, "\\mathbf{J}"))],
              verified=["sympy", "finite-difference"], tags=[f"{n_out}x{n_in}"])


# --------------------------------------------------------------------- hessian
CLS = ["cực tiểu địa phương", "cực đại địa phương", "điểm yên ngựa", "chưa kết luận được (Hessian suy biến)"]


def gen_hessian(rng: random.Random, level: int, seed: int):
    if level == 5:
        n, d = rng.choice([(3, 2), (4, 2), (3, 3)])
        X = nontrivial_matrix(rng, n, d, -3, 3)
        w = symbols(f"w1:{d + 1}")
        yv = Matrix([_c(rng) for _ in range(n)])
        f = Rational(1, 2) * sum(((X * Matrix(w) - yv)[i]) ** 2 for i in range(n))
        H = hessian(f, w)
        check(H == X.T * X, "Hessian = XᵀX")
        return cb(type_="hessian", lesson=L35, level=5, seed=seed,
                  prompt=Prompt(text="Mất mát bình phương tối thiểu $L(\\mathbf{w}) = \\frac{1}{2}\\lVert\\mathbf{X}\\mathbf{w} - \\mathbf{y}\\rVert_2^2$. Tính Hessian $\\mathbf{H} = \\nabla^2 L$.",
                                display=[L.mat(X, B("X")), L.mat(yv, B("y"))]),
                  answer=_mat_ans(H), hints=["$\\nabla L = \\mathbf{X}\\T(\\mathbf{X}\\mathbf{w} - \\mathbf{y})$, đạo hàm thêm lần nữa theo $\\mathbf{w}$."],
                  steps=[Step(text="Hessian không phụ thuộc $\\mathbf{w}$ và $\\mathbf{y}$:", latex=L.mat(H, "\\nabla^2 L = \\mathbf{X}\\T\\mathbf{X}"))], verified=["sympy"], tags=["ml"])
    if level == 4:  # phân loại điểm dừng của hàm bậc hai
        for _ in range(500):
            a, b, c = _c(rng, -4, 4), _c(rng, -4, 4), _c(rng, -4, 4)
            d, e = rng.randint(-6, 6), rng.randint(-6, 6)
            f = a * x**2 + b * x * y + c * y**2 + d * x + e * y
            H = hessian(f, (x, y))
            if H.det() == 0:
                continue
            sol = solve([diff(f, x), diff(f, y)], [x, y], dict=True)
            if len(sol) != 1:
                continue
            break
        s = sol[0]
        ev = [float(v) for v in H.eigenvals()]
        np_ev = np.linalg.eigvalsh(to_np(H))
        cls = 0 if all(v > 0 for v in np_ev) else 1 if all(v < 0 for v in np_ev) else 2
        pt = f"({s[x]}, {s[y]})"
        return cb(type_="hessian", lesson=L35, level=4, seed=seed,
                  prompt=Prompt(text=f"Hàm $f$ có điểm dừng (gradient bằng $\\mathbf{{0}}$) tại ${pt.replace('(', '(').replace(')', ')')}$. Điểm đó là gì?", display=[f"f(x, y) = {tex(f)}"]),
                  answer=Answer(kind="choice", correct=[cls]), choices=CLS,
                  hints=["Tính Hessian rồi xét dấu trị riêng: cả hai dương: cực tiểu; cả hai âm: cực đại; trái dấu: yên ngựa.", "Với $2\\times 2$: $\\det > 0$ và $f_{xx} > 0$: cực tiểu; $\\det > 0$ và $f_{xx} < 0$: cực đại; $\\det < 0$: yên ngựa."],
                  steps=[Step(text="Hessian (không đổi vì $f$ bậc hai):", latex=L.mat(H, "\\mathbf{H}")),
                         Step(text=f"$\\det(\\mathbf{{H}}) = {L.num(H.det())}$, $f_{{xx}} = {L.num(H[0, 0])}$: " + CLS[cls] + ".")],
                  verified=["sympy", "numpy"], tags=["classify"])
    nv = 2 if level == 2 else 3
    vs = [x, y, z][:nv]
    f = sum(_c(rng) * x**i * y**j * (z**k if nv == 3 else 1) for i, j, k in [(rng.randint(0, 3), rng.randint(0, 3), rng.randint(0, 2)) for _ in range(3)]) + _c(rng) * x * y
    pt = tuple(rng.randint(-2, 3) for _ in range(nv))
    Hs = hessian(f, vs)
    H = Hs.subs(dict(zip(vs, pt)))
    check(H == H.T, "đối xứng")
    # kiểm chứng sai phân bậc hai bằng gradient
    from sympy import lambdify

    gg = lambdify(vs, [diff(f, v) for v in vs], "numpy")
    for j in range(nv):
        hi, lo = [float(p) for p in pt], [float(p) for p in pt]
        hi[j] += 1e-6
        lo[j] -= 1e-6
        col = (np.array(gg(*hi), dtype=float) - np.array(gg(*lo), dtype=float)) / 2e-6
        check(np.allclose(col, np.array(H[:, j].tolist(), dtype=float).ravel(), atol=1e-3), "Hessian lệch sai phân")
    names = "(" + ", ".join(str(v) for v in vs) + ")"
    return cb(type_="hessian", lesson=L35, level=level, seed=seed,
              prompt=Prompt(text=f"Tính ma trận Hessian $\\mathbf{{H}}$ của $f$ tại điểm ${names} = ({', '.join(str(p) for p in pt)})$.", display=[f"f{names} = {tex(f)}"]),
              answer=_mat_ans(H), hints=["$H_{ij} = \\partial^2 f/\\partial x_i\\partial x_j$; ma trận đối xứng.", "Tính gradient trước, rồi đạo hàm từng thành phần."],
              steps=[Step(text="Hessian theo ký hiệu:", latex="\\mathbf{H} = " + L.expr_mat([[tex(Hs[i, j]) for j in range(nv)] for i in range(nv)])),
                     Step(text="Thay điểm:", latex=L.mat(H, "\\mathbf{H}"))],
              verified=["sympy", "finite-difference"], tags=[f"{nv}x{nv}"])


# --------------------------------------------------------------------- vec-deriv
def gen_vec_deriv(rng: random.Random, level: int, seed: int):
    if level == 2:  # ∇(aᵀx) = a
        n = rng.choice([2, 3])
        a = Matrix([_c(rng, -6, 6) for _ in range(n)])
        xs = symbols(f"x1:{n + 1}")
        f = (a.T * Matrix(xs))[0, 0]
        g = Matrix([diff(f, v) for v in xs])
        check(g == a, "∇ aᵀx = a")
        return cb(type_="vec-deriv", lesson=L36, level=2, seed=seed,
                  prompt=Prompt(text="Cho $f(\\mathbf{x}) = \\mathbf{a}\\T\\mathbf{x}$. Tính $\\nabla_{\\mathbf{x}} f$.", display=[L.mat(a, B("a"))]),
                  answer=_vec_ans(list(a)), hints=["$f = \\sum_i a_ix_i$ nên $\\partial f/\\partial x_i = a_i$."], steps=[Step(text="Quy tắc: $\\nabla(\\mathbf{a}\\T\\mathbf{x}) = \\mathbf{a}$.", latex=L.mat(a, "\\nabla f"))],
                  verified=["sympy"], tags=["linear"])
    if level == 3:  # ∇(xᵀAx) = (A + Aᵀ)x
        n = 2
        A = nontrivial_matrix(rng, n, n, -4, 4)
        xv = Matrix([_c(rng, -4, 4) for _ in range(n)])
        xs = symbols("x1:3")
        f = (Matrix(xs).T * A * Matrix(xs))[0, 0]
        g = Matrix([diff(f, v) for v in xs]).subs(dict(zip(xs, xv)))
        check(g == (A + A.T) * xv, "∇ xᵀAx = (A+Aᵀ)x")
        return cb(type_="vec-deriv", lesson=L36, level=3, seed=seed,
                  prompt=Prompt(text="Cho $f(\\mathbf{x}) = \\mathbf{x}\\T\\mathbf{A}\\mathbf{x}$. Tính $\\nabla f$ tại $\\mathbf{x}$ cho dưới đây.", display=[L.mat(A, B("A")), L.mat(xv, B("x"))]),
                  answer=_vec_ans(list(g)), hints=["$\\nabla(\\mathbf{x}\\T\\mathbf{A}\\mathbf{x}) = (\\mathbf{A} + \\mathbf{A}\\T)\\mathbf{x}$; nếu $\\mathbf{A}$ đối xứng thì bằng $2\\mathbf{A}\\mathbf{x}$."],
                  steps=[Step(text="$\\mathbf{A} + \\mathbf{A}\\T$:", latex=L.mat(A + A.T)), Step(text="Nhân với $\\mathbf{x}$:", latex=L.mat(g, "\\nabla f"))],
                  verified=["sympy"], tags=["quadratic"])
    if level == 4:  # ∇ ½‖Ax − b‖² = Aᵀ(Ax − b)
        m, n = rng.choice([(3, 2), (2, 2), (3, 3)])
        A = nontrivial_matrix(rng, m, n, -3, 3)
        b = Matrix([_c(rng, -5, 5) for _ in range(m)])
        xv = Matrix([_c(rng, -3, 3) for _ in range(n)])
        xs = symbols(f"x1:{n + 1}")
        f = Rational(1, 2) * sum(((A * Matrix(xs) - b)[i]) ** 2 for i in range(m))
        g = Matrix([diff(f, v) for v in xs]).subs(dict(zip(xs, xv)))
        check(g == A.T * (A * xv - b), "∇ = Aᵀ(Ax − b)")
        return cb(type_="vec-deriv", lesson=L36, level=4, seed=seed,
                  prompt=Prompt(text="Cho $f(\\mathbf{x}) = \\frac{1}{2}\\lVert\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\rVert_2^2$. Tính $\\nabla f$ tại $\\mathbf{x}$ cho dưới đây.",
                                display=[L.mat(A, B("A")), L.mat(b, B("b")), L.mat(xv, B("x"))]),
                  answer=_vec_ans(list(g)), hints=["$\\nabla f = \\mathbf{A}\\T(\\mathbf{A}\\mathbf{x} - \\mathbf{b})$.", "Tính phần dư $\\mathbf{A}\\mathbf{x} - \\mathbf{b}$ trước."],
                  steps=[Step(text="Phần dư:", latex=L.mat(A * xv - b, "\\mathbf{r}")), Step(text="$\\mathbf{A}\\T\\mathbf{r}$:", latex=L.mat(g, "\\nabla f"))],
                  verified=["sympy"], tags=["least-squares"])
    # level 5: gradient MSE
    N, d = rng.choice([(3, 2), (2, 2), (4, 2)])
    X = nontrivial_matrix(rng, N, d, -3, 3)
    yv = Matrix([_c(rng, -5, 5) for _ in range(N)])
    w = Matrix([_c(rng, -3, 3) for _ in range(d)])
    ws = symbols(f"w1:{d + 1}")
    f = sum(((X * Matrix(ws) - yv)[i]) ** 2 for i in range(N)) / N
    g = Matrix([diff(f, v) for v in ws]).subs(dict(zip(ws, w)))
    check(g == Rational(2, N) * X.T * (X * w - yv), "∇ MSE")
    return cb(type_="vec-deriv", lesson=L36, level=5, seed=seed,
              prompt=Prompt(text=f"Mất mát MSE $L(\\mathbf{{w}}) = \\frac{{1}}{{N}}\\lVert\\mathbf{{X}}\\mathbf{{w}} - \\mathbf{{y}}\\rVert_2^2$ với $N = {N}$. Tính $\\nabla_{{\\mathbf{{w}}}} L$ tại $\\mathbf{{w}}$ cho dưới đây.",
                            display=[L.mat(X, B("X")), L.mat(yv, B("y")), L.mat(w, B("w"))]),
              answer=_vec_ans(list(g)), hints=["$\\nabla L = \\frac{2}{N}\\mathbf{X}\\T(\\mathbf{X}\\mathbf{w} - \\mathbf{y})$."],
              steps=[Step(text="Phần dư:", latex=L.mat(X * w - yv, "\\mathbf{r}")), Step(text="$\\frac{2}{N}\\mathbf{X}\\T\\mathbf{r}$:", latex=L.mat(g, "\\nabla L"))],
              verified=["sympy"], tags=["ml", "mse"])


# ---------------------------------------------------------- đồ thị tính toán
OPS = ["add", "sub", "mul", "sq", "relu"]


def _fmt(op, a, b=None):
    return {"add": f"{a} + {b}", "sub": f"{a} - {b}", "mul": f"{a} * {b}", "sq": f"{a} ** 2", "relu": f"max(0, {a})"}[op]


def _apply(op, a, b=None):
    return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "sq": lambda: a**2, "relu": lambda: max(0, a)}[op]()


def _rand_graph(rng: random.Random, n_in: int, n_ops: int, relu: bool):
    names = ["a", "b", "c"][:n_in]
    vals = {n: Rational(_c(rng, -3, 3)) for n in names}
    nodes: list[tuple] = []
    avail = list(names)
    inter = ["u", "v", "w", "t"]
    for k in range(n_ops):
        out = inter[k] if k < n_ops - 1 else "L"
        op = rng.choice(OPS if relu else ["add", "sub", "mul", "sq"])
        if k == n_ops - 1 and op in ("relu",):
            op = "sq"
        if op in ("sq", "relu"):
            arg = (avail[-1] if rng.random() < 0.7 else rng.choice(avail),)
        else:
            a, b = rng.choice(avail), rng.choice(avail)
            if k == 0 and len(names) > 1:
                a, b = names[0], names[1]
            arg = (a, b)
        nodes.append((out, op, arg))
        avail.append(out)
    return names, vals, nodes


def _forward(vals, nodes):
    v = dict(vals)
    for out, op, args in nodes:
        v[out] = _apply(op, *[v[a] for a in args])
    return v


def _backward(v, nodes, wrt):
    g = {n: Rational(0) for n in v}
    g["L"] = Rational(1)
    steps = []
    for out, op, args in reversed(nodes):
        go = g[out]
        if op == "add":
            contrib = [go, go]
        elif op == "sub":
            contrib = [go, -go]
        elif op == "mul":
            contrib = [go * v[args[1]], go * v[args[0]]]
        elif op == "sq":
            contrib = [2 * v[args[0]] * go]
        else:
            contrib = [go if v[args[0]] > 0 else Rational(0)]
        for a, c in zip(args, contrib):
            g[a] += c
        steps.append((out, op, args, go, contrib))
    return g, steps


def _sympy_grad(names, vals, nodes):
    syms = {n: Symbol(n) for n in names}
    env = dict(syms)
    from sympy import Max

    for out, op, args in nodes:
        a = [env[x_] for x_ in args]
        env[out] = {"add": lambda: a[0] + a[1], "sub": lambda: a[0] - a[1], "mul": lambda: a[0] * a[1], "sq": lambda: a[0] ** 2, "relu": lambda: Max(0, a[0])}[op]()
    Lexpr = env["L"]
    return {n: diff(Lexpr, syms[n]).subs({syms[k]: vals[k] for k in names}) for n in names}, Lexpr


def _code(names, vals, nodes):
    lines = [f"{n} = {vals[n]}" for n in names]
    for out, op, args in nodes:
        lines.append(f"{out} = {_fmt(op, *args)}")
    return "\n".join(lines)


def gen_graph_forward(rng: random.Random, level: int, seed: int):
    if level == 5:
        xv, wv, bv, yv = _c(rng, -3, 3), _c(rng, -3, 3), _c(rng, -3, 3), _c(rng, -4, 4)
        zz = wv * xv + bv
        aa = max(0, zz)
        Lv = (aa - yv) ** 2
        code = f"x = {xv}\nw = {wv}\nb = {bv}\ny = {yv}\n\nz = w * x + b\na = max(0, z)      # ReLU\nL = (a - y) ** 2"
        return cb(type_="graph-forward", lesson=L37, level=5, seed=seed,
                  prompt=Prompt(text="Đồ thị tính toán của một nơ-ron ReLU với hàm mất mát bình phương. Chạy pha **forward** và nhập giá trị $L$.", code=code),
                  answer=Answer(kind="number", value=str(Rational(Lv))), hints=["Tính lần lượt từng nút theo thứ tự trong code."],
                  steps=[Step(text=f"$z = {wv}\\cdot {xv} + {bv} = {zz}$, $a = \\max(0, z) = {aa}$, $L = ({aa} - {yv})^2 = {Lv}$.")], verified=["python"], tags=["ml"])
    n_in, n_ops = {2: (2, 3), 3: (2, 4), 4: (3, 5)}[level]
    for _ in range(100):
        names, vals, nodes = _rand_graph(rng, n_in, n_ops, relu=level >= 3)
        v = _forward(vals, nodes)
        if all(abs(val) < 500 for val in v.values()) and v["L"] != 0:
            break
    target = "L" if level < 4 else rng.choice([nodes[-2][0], "L"])
    return cb(type_="graph-forward", lesson=L37, level=level, seed=seed,
              prompt=Prompt(text=f"Chạy pha **forward** của đồ thị tính toán sau và nhập giá trị của `{target}`.", code=_code(names, vals, nodes)),
              answer=Answer(kind="number", value=str(v[target])), hints=["Tính các nút theo thứ tự xuất hiện; mỗi nút chỉ dùng các nút đã tính."],
              steps=[Step(text=f"`{out}` = {_fmt(op, *args)} = {v[out]}") for out, op, args in nodes],
              verified=["python"], tags=["graph"])


def gen_graph_backward(rng: random.Random, level: int, seed: int):
    if level == 5:
        xv, wv, bv, yv = _c(rng, -3, 3), _c(rng, -3, 3), _c(rng, -3, 3), _c(rng, -4, 4)
        zz = wv * xv + bv
        while zz <= 0:
            wv, bv = _c(rng, -3, 3), _c(rng, -3, 3)
            zz = wv * xv + bv
        aa = zz
        dL_da = 2 * (aa - yv)
        gw, gb = dL_da * xv, dL_da
        w_, b_ = symbols("w b")
        Lx = (w_ * xv + b_ - yv) ** 2
        check([diff(Lx, w_).subs({w_: wv, b_: bv}), diff(Lx, b_).subs({w_: wv, b_: bv})] == [gw, gb], "backprop = sympy")
        code = f"x = {xv}\nw = {wv}\nb = {bv}\ny = {yv}\n\nz = w * x + b      # z > 0 nên ReLU cho a = z\na = max(0, z)\nL = (a - y) ** 2"
        return cb(type_="graph-backward", lesson=L37, level=5, seed=seed,
                  prompt=Prompt(text="Chạy **backpropagation** cho nơ-ron ReLU trên. Nhập vector $(\\partial L/\\partial w,\\; \\partial L/\\partial b)$.", code=code),
                  answer=_vec_ans([gw, gb]),
                  hints=["Đi ngược: $\\partial L/\\partial a = 2(a - y)$; ReLU với $z > 0$ truyền nguyên; $\\partial z/\\partial w = x$, $\\partial z/\\partial b = 1$."],
                  steps=[Step(text=f"$\\partial L/\\partial a = 2({aa} - {yv}) = {dL_da}$; qua ReLU (z > 0) giữ nguyên: $\\partial L/\\partial z = {dL_da}$."),
                         Step(text="Chain rule:", latex=f"\\partial L/\\partial w = {dL_da}\\cdot {xv} = {gw},\\quad \\partial L/\\partial b = {dL_da}\\cdot 1 = {gb}")],
                  verified=["sympy"], tags=["ml"])
    n_in, n_ops = {3: (2, 3), 4: (2, 4)}[level]
    for _ in range(300):
        names, vals, nodes = _rand_graph(rng, n_in, n_ops, relu=True)
        v = _forward(vals, nodes)
        g, steps_ = _backward(v, nodes, names)
        used = [n for n in names if g[n] != 0]
        if len(used) >= 1 and all(abs(val) < 500 for val in v.values()) and (level == 3 or len(used) == n_in):
            break
    sg, _L = _sympy_grad(names, vals, nodes)
    for n in names:
        check(Rational(sg[n]) == g[n], f"backprop ≠ sympy tại {n}")
    lines = [Step(text=f"Nút `{out}` = {_fmt(op, *args)} (gradient đến nút này: {go}); đóng góp cho " + ", ".join(f"`{a}`: {c}" for a, c in zip(args, contrib)) + ".")
             for out, op, args, go, contrib in steps_]
    if level == 3:
        target = rng.choice(names)
        return cb(type_="graph-backward", lesson=L37, level=3, seed=seed,
                  prompt=Prompt(text=f"Tính $\\partial L/\\partial {target}$ bằng backpropagation (nhớ cộng đóng góp từ mọi đường đi; $\\mathrm{{ReLU}}'$ tại $0$ quy ước là $0$).", code=_code(names, vals, nodes)),
                  answer=Answer(kind="number", value=str(g[target])),
                  hints=["Đi ngược từ `L` với $\\partial L/\\partial L = 1$.", "Nút dùng ở nhiều chỗ nhận gradient **cộng dồn** từ mỗi chỗ dùng."],
                  steps=lines + [Step(text=f"Kết quả: $\\partial L/\\partial {target} = {g[target]}$.")], verified=["sympy", "python"], tags=["graph"])
    return cb(type_="graph-backward", lesson=L37, level=4, seed=seed,
              prompt=Prompt(text="Tính vector gradient $\\nabla L = (\\partial L/\\partial a,\\; \\partial L/\\partial b)$ bằng backpropagation.", code=_code(names, vals, nodes)),
              answer=_vec_ans([g[n] for n in names[:2]]) if n_in >= 2 else _vec_ans([g[names[0]]]),
              hints=["Đi ngược theo thứ tự đảo của các dòng; cộng gradient khi một nút có nhiều đường đi tới."],
              steps=lines, verified=["sympy", "python"], tags=["graph", "vector"])


SPECS = [
    TopicSpec("partial-deriv", L33, (1, 2, 3, 4), gen_partial),
    TopicSpec("gradient", L33, (2, 3, 4, 5), gen_gradient),
    TopicSpec("jacobian", L34, (2, 3, 4, 5), gen_jacobian),
    TopicSpec("hessian", L35, (2, 3, 4, 5), gen_hessian),
    TopicSpec("vec-deriv", L36, (2, 3, 4, 5), gen_vec_deriv),
    TopicSpec("graph-forward", L37, (2, 3, 4, 5), gen_graph_forward),
    TopicSpec("graph-backward", L37, (3, 4, 5), gen_graph_backward),
]
