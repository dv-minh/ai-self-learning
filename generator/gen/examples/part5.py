r"""Ví dụ cho Phần 5 (Tối ưu). Mọi số do chương trình tính bằng Fraction và kiểm chứng; MDX chỉ chứa kết quả."""
from __future__ import annotations

import math
from fractions import Fraction as F
from itertools import combinations

from sympy import Matrix, Rational, diff, hessian, solve, symbols

from ..core.verify import check
from ..topics.opt._o import fp, ft, lf, vt
from .common import display, write_examples

x, y = symbols("x y")
EX: dict[str, dict] = {}


def lesson(code: str):
    def deco(fn):
        EX.setdefault(code, {})[fn.__name__.replace("_", "-")] = fn
        return fn
    return deco


def table(header: list[str], rows: list[list[str]]) -> str:
    cols = "c" * len(header)
    body = " \\\\ ".join(" & ".join(r) for r in rows)
    return f"\\begin{{array}}{{{cols}}} {' & '.join(header)} \\\\ \\hline {body} \\end{{array}}"


def _q(v):
    v = F(v)
    return Rational(v.numerator, v.denominator)


# ------------------------------------------------------------------ 5.1
@lesson("5.1")
def jensen():
    f = lambda t: t * t
    a, b, t = F(-1), F(3), F(1, 2)
    mid = f(t * a + (1 - t) * b)
    chord = t * f(a) + (1 - t) * f(b)
    check(mid <= chord, "Jensen")
    return display(f"f(x) = x^2,\\quad a = {ft(a)},\\ b = {ft(b)},\\ \\theta = {ft(t)}",
                   f"f\\big(\\theta a + (1-\\theta)b\\big) = f({ft(t * a + (1 - t) * b)}) = {ft(mid)} \\;\\le\\; \\theta f(a) + (1-\\theta)f(b) = {ft(chord)}")


@lesson("5.1")
def hessian_quad():
    f = 2 * x**2 + x * y + y**2
    h = hessian(f, (x, y))
    ev = h.eigenvals()
    check(all(v > 0 for v in ev), "xác định dương")
    from sympy import latex

    return display(f"f(x, y) = {latex(f)}", f"\\mathbf{{H}} = {_m(h)},\\qquad \\det\\mathbf{{H}} = {h.det()} > 0,\\quad H_{{11}} = {h[0, 0]} > 0",
                   "\\lambda_1, \\lambda_2 = " + ",\\ ".join(latex(v) for v in sorted(ev, key=float)) + " > 0") + "\n\nHessian xác định dương ở mọi điểm nên $f$ **lồi chặt**."


def _m(m: Matrix) -> str:
    return "\\begin{bmatrix} " + " \\\\ ".join(" & ".join(str(m[i, j]) for j in range(m.cols)) for i in range(m.rows)) + " \\end{bmatrix}"


@lesson("5.1")
def minimise_quadratic():
    f = 3 * x**2 + x * y + y**2 - 5 * x - y
    g = [diff(f, v) for v in (x, y)]
    sol = solve(g, [x, y], dict=True)[0]
    h = hessian(f, (x, y))
    check(h.is_positive_definite, "Hessian xác định dương")
    check(all(gi.subs(sol) == 0 for gi in g), "gradient bằng 0")
    from sympy import latex

    return display(f"f(x, y) = {latex(f)}", f"\\nabla f = {_m(Matrix(g))} = \\mathbf{{0}}", f"\\mathbf{{x}}^\\star = ({latex(sol[x])}, {latex(sol[y])})\\T,\\qquad f(\\mathbf{{x}}^\\star) = {latex(f.subs(sol))}") \
        + f"\n\nHessian $\\mathbf{{H}} = {_m(h)}$ xác định dương nên $f$ lồi chặt và điểm này là cực tiểu **toàn cục duy nhất**."


@lesson("5.1")
def nonconvex_double_well():
    f = lambda t: t**4 - 2 * t**2
    a, b = F(-1), F(1)
    mid = f((a + b) / 2)
    avg = (f(a) + f(b)) / 2
    check(mid > avg, "vi phạm Jensen")
    return display("f(x) = x^4 - 2x^2,\\quad f''(x) = 12x^2 - 4",
                   f"f(-1) = {f(a)},\\quad f(1) = {f(b)},\\quad f(0) = {f(F(0))}",
                   f"f\\!\\left(\\tfrac{{-1 + 1}}{{2}}\\right) = {mid} \\;>\\; \\tfrac{{f(-1) + f(1)}}{{2}} = {avg}") \
        + "\n\nBất đẳng thức Jensen bị vi phạm nên $f$ **không lồi**. Nó có hai cực tiểu địa phương (tại $\\pm 1$) và một cực đại địa phương tại $0$."


@lesson("5.1")
def prop_sum_convex():
    f = lambda t: t * t
    g = lambda t: abs(t)
    a, b, t = F(-2), F(5), F(1, 3)
    h = lambda z: f(z) + g(z)
    mid = t * a + (1 - t) * b
    lhs, rhs = h(mid), t * h(a) + (1 - t) * h(b)
    check(lhs <= rhs, "tổng hàm lồi")
    return display(f"h(x) = x^2 + |x|,\\quad a = {ft(a)},\\ b = {ft(b)},\\ \\theta = {ft(t)}", f"h({ft(mid)}) = {ft(lhs)} \\;\\le\\; {ft(rhs)} = \\theta h(a) + (1-\\theta)h(b)")


@lesson("5.1")
def mistake_second_derivative_zero():
    f = lambda t: t**4
    check(all(F(12) * F(k, 4) ** 2 >= 0 for k in range(-8, 9)), "f'' >= 0")
    return ("**Sai:** $f(x) = x^4$ có $f''(0) = 0$ nên \"không lồi\".\n\n"
            + display("f''(x) = 12x^2 \\ge 0 \\quad \\forall x") + "\n\n**Đúng:** $f'' \\ge 0$ ở mọi nơi, nên $f$ lồi (thậm chí lồi chặt). $f''$ chỉ cần không âm; việc bằng $0$ tại một điểm không làm mất tính lồi.")


# ------------------------------------------------------------------ 5.2
def _gd_table(grad, f, x0, eta, k):
    xs = [x0]
    rows = []
    for t in range(k):
        g = grad(xs[-1])
        rows.append([str(t), ft(xs[-1]), ft(g), ft(f(xs[-1]))])
        xs.append(xs[-1] - eta * g)
    rows.append([str(k), ft(xs[-1]), "", ft(f(xs[-1]))])
    return rows, xs


@lesson("5.2")
def gd_1d():
    f = lambda t: t * t - 4 * t
    grad = lambda t: 2 * t - 4
    eta = F(1, 4)
    rows, xs = _gd_table(grad, f, F(0), eta, 4)
    check(all(f(xs[i + 1]) < f(xs[i]) for i in range(4)), "f giảm")
    check(abs(xs[-1] - 2) < abs(xs[0] - 2), "tiến gần x*")
    return (display(f"f(x) = x^2 - 4x,\\quad f'(x) = 2x - 4,\\quad x_0 = 0,\\quad \\eta = {ft(eta)}", table(["t", "x_t", "f'(x_t)", "f(x_t)"], rows))
            + "\n\nCực tiểu thật là $x^\\star = 2$; dãy $x_t$ tiến dần về đó và $f(x_t)$ giảm ở mọi bước.")


@lesson("5.2")
def gd_2d():
    grad = lambda p: [2 * p[0], 6 * p[1]]
    p0, eta = [F(4), F(2)], F(1, 4)
    p1 = [a - eta * g for a, g in zip(p0, grad(p0))]
    p2 = [a - eta * g for a, g in zip(p1, grad(p1))]
    return display("f(x, y) = x^2 + 3y^2,\\quad \\nabla f = (2x,\\ 6y)\\T,\\quad \\eta = " + ft(eta),
                   f"\\mathbf{{x}}_1 = {vt(p0)} - {fp(eta)}\\cdot {vt(grad(p0))} = {vt(p1)}", f"\\mathbf{{x}}_2 = {vt(p1)} - {fp(eta)}\\cdot {vt(grad(p1))} = {vt(p2)}")


@lesson("5.2")
def lr_effect():
    rows_all = []
    labels = []
    for eta in (F(1, 10), F(1, 2), F(1), F(3, 2)):
        xs = [F(1)]
        for _ in range(4):
            xs.append(xs[-1] - eta * 2 * xs[-1])
        r = 1 - 2 * eta
        check(all(xs[i + 1] == r * xs[i] for i in range(4)), "x_{t+1} = r x_t")
        labels.append(ft(eta))
        rows_all.append([ft(eta), ft(r)] + [ft(v) for v in xs])
    return display("f(x) = x^2,\\quad x_{t+1} = x_t - \\eta\\cdot 2x_t = (1 - 2\\eta)\\,x_t,\\quad x_0 = 1",
                   table(["\\eta", "r = 1-2\\eta", "x_0", "x_1", "x_2", "x_3", "x_4"], rows_all)) \
        + "\n\n$\\eta$ nhỏ: hội tụ chậm; $\\eta = 1/2$: về đúng $0$ chỉ sau một bước; $\\eta = 1$: nhảy qua lại giữa $1$ và $-1$ mãi mãi; $\\eta = 3/2$: $|r| > 1$ nên phân kỳ."


@lesson("5.2")
def linreg_step():
    xs, ys = [F(1), F(2), F(3)], [F(2), F(2), F(4)]
    w, b, eta = F(0), F(0), F(1, 4)
    n = len(xs)
    res = [w * xi + b - yi for xi, yi in zip(xs, ys)]
    gw = sum(r * xi for r, xi in zip(res, xs)) / n
    gb = sum(res) / n
    w1, b1 = w - eta * gw, b - eta * gb
    loss = lambda ww, bb: sum((ww * xi + bb - yi) ** 2 for xi, yi in zip(xs, ys)) / (2 * n)
    check(loss(w1, b1) < loss(w, b), "mất mát giảm")
    return display("L(w, b) = \\dfrac{1}{2n}\\sum_i (wx_i + b - y_i)^2", "x = (1, 2, 3),\\ y = (2, 2, 4),\\ (w_0, b_0) = (0, 0),\\ \\eta = " + ft(eta),
                   "e = \\hat{y} - y = (" + ", ".join(ft(r) for r in res) + ")",
                   f"\\dfrac{{\\partial L}}{{\\partial w}} = \\dfrac{{1}}{{3}}\\sum e_ix_i = {ft(gw)},\\qquad \\dfrac{{\\partial L}}{{\\partial b}} = \\dfrac{{1}}{{3}}\\sum e_i = {ft(gb)}",
                   f"(w_1, b_1) = ({ft(w1)},\\ {ft(b1)}),\\qquad L: {ft(loss(w, b))} \\to {ft(loss(w1, b1))}")


@lesson("5.2")
def prop_descent():
    f = lambda t: t * t - 4 * t
    x0 = F(0)
    out = []
    for eta in (F(1, 4), F(1, 2), F(3, 4)):
        x1 = x0 - eta * (2 * x0 - 4)
        out.append((eta, x1, f(x1)))
        check(f(x1) < f(x0), "f giảm với eta nhỏ")
    return display("f(x) = x^2 - 4x,\\ x_0 = 0,\\ f(x_0) = 0", table(["\\eta", "x_1", "f(x_1)"], [[ft(e), ft(a), ft(b)] for e, a, b in out])) \
        + "\n\nVới $\\eta$ đủ nhỏ, bước đi ngược gradient luôn làm $f$ giảm."


@lesson("5.2")
def mistake_sign():
    f = lambda t: t * t
    x0, eta = F(3), F(1, 4)
    good = x0 - eta * 2 * x0
    bad = x0 + eta * 2 * x0
    check(f(good) < f(x0) < f(bad), "đúng dấu thì giảm")
    return (f"**Sai:** cập nhật $x \\leftarrow x + \\eta f'(x)$ (đi theo gradient). Với $f(x) = x^2$, $x_0 = 3$, $\\eta = {ft(eta)}$: $x_1 = {ft(bad)}$, $f$ tăng từ ${f(x0)}$ lên ${f(bad)}$.\n\n"
            f"**Đúng:** $x_1 = x_0 - \\eta f'(x_0) = {ft(good)}$ và $f$ giảm xuống ${ft(f(good))}$.")


# ------------------------------------------------------------------ 5.3
_DATA = ([F(1), F(2), F(3), F(4)], [F(1), F(3), F(2), F(5)])


def _g(xs, ys, idx, w):
    return sum((w * xs[i] - ys[i]) * xs[i] for i in idx) / len(idx)


@lesson("5.3")
def minibatch_step():
    xs, ys = _DATA
    w, eta = F(1), F(1, 10)
    full = _g(xs, ys, range(4), w)
    mb = _g(xs, ys, [0, 2], w)
    return display("x = (1, 2, 3, 4),\\ y = (1, 3, 2, 5),\\ \\hat{y} = wx,\\ w_0 = 1,\\ \\eta = " + ft(eta),
                   f"g_i = (wx_i - y_i)\\,x_i = ({', '.join(ft((w * a - b) * a) for a, b in zip(xs, ys))})",
                   f"\\text{{full batch: }} \\dfrac{{1}}{{4}}\\sum g_i = {ft(full)}",
                   f"\\text{{mini-batch }}\\{{1, 3\\}}: \\dfrac{{g_1 + g_3}}{{2}} = {ft(mb)} \\;\\Rightarrow\\; w_1 = {ft(w)} - {fp(eta)}\\cdot {fp(mb)} = {ft(w - eta * mb)}")


@lesson("5.3")
def epoch_two_updates():
    xs, ys = _DATA
    w, eta = F(1), F(1, 10)
    rows = []
    for k, idx in enumerate([[0, 1], [2, 3]], 1):
        g = _g(xs, ys, idx, w)
        wn = w - eta * g
        rows.append([str(k), "\\{" + ",".join(str(i + 1) for i in idx) + "\\}", ft(w), ft(g), ft(wn)])
        w = wn
    return display("|\\mathcal{B}| = 2,\\ \\eta = " + ft(eta), table(["k", "\\mathcal{B}", "w", "g", "w'"], rows)) + "\n\nCột $k$ là số thứ tự cập nhật, $\\mathcal{B}$ là mini-batch, $w'$ là giá trị sau cập nhật. Một epoch gồm $4/2 = 2$ lần cập nhật; lần sau dùng $w$ vừa cập nhật."


@lesson("5.3")
def unbiased():
    xs, ys = _DATA
    w = F(1)
    full = _g(xs, ys, range(4), w)
    batches = list(combinations(range(4), 2))
    means = [_g(xs, ys, b, w) for b in batches]
    avg = sum(means) / len(means)
    check(avg == full, "không chệch")
    return ("Có $\\binom{4}{2} = 6$ mini-batch cỡ $2$. Gradient của từng batch (theo thứ tự các tập con) rồi trung bình:\n\n"
            + display("(" + ", ".join(ft(m) for m in means) + ")", f"\\dfrac{{1}}{{6}}\\sum_{{\\mathcal{{B}}}} g_{{\\mathcal{{B}}}} = {ft(avg)} = \\dfrac{{1}}{{4}}\\sum_i g_i = {ft(full)}"))


@lesson("5.3")
def mistake_divide():
    xs, ys = _DATA
    w = F(1)
    idx = [0, 2]
    wrong = sum((w * xs[i] - ys[i]) * xs[i] for i in idx) / 4
    right = _g(xs, ys, idx, w)
    check(wrong != right, "khác nhau")
    return f"**Sai:** chia tổng gradient của batch $\\{{1, 3\\}}$ cho $N = 4$ (kích thước toàn bộ dữ liệu): ${ft(wrong)}$.\n\n**Đúng:** chia cho $|\\mathcal{{B}}| = 2$: ${ft(right)}$. Nếu chia cho $N$, gradient bị co lại theo tỉ lệ $|\\mathcal{{B}}|/N$ và bước đi nhỏ hơn dự định."


# ------------------------------------------------------------------ 5.4
@lesson("5.4")
def momentum_vs_gd():
    grad = lambda t: t  # f = x^2 / 2
    eta, beta = F(1, 2), F(1, 2)
    xg, xm, v = F(8), F(8), F(0)
    rows = []
    for t in range(5):
        rows.append([str(t), ft(xg), ft(xm), ft(v)])
        xg = xg - eta * grad(xg)
        v = beta * v + grad(xm)
        xm = xm - eta * v
    rows.append(["5", ft(xg), ft(xm), ft(v)])
    return display("f(x) = \\tfrac{1}{2}x^2,\\ \\eta = " + ft(eta) + ",\\ \\beta = " + ft(beta) + ",\\ x_0 = 8,\\ v_0 = 0", table(["t", "x_t\\ (\\text{GD})", "x_t\\ (\\text{momentum})", "v_t"], rows))


@lesson("5.4")
def momentum_two_steps():
    eta, beta = F(1, 4), F(1, 2)
    x0, v = F(4), F(0)
    f = lambda t: t * t / 2  # f' = t
    g1 = x0
    v1 = beta * v + g1
    x1 = x0 - eta * v1
    g2 = x1
    v2 = beta * v1 + g2
    x2 = x1 - eta * v2
    return display(f"f(x) = \\tfrac{{1}}{{2}}x^2,\\ \\eta = {ft(eta)},\\ \\beta = {ft(beta)},\\ x_0 = {ft(x0)},\\ v_0 = 0",
                   f"g_1 = {ft(g1)},\\ v_1 = {ft(beta)}\\cdot 0 + {ft(g1)} = {ft(v1)},\\ x_1 = {ft(x0)} - {fp(eta)}\\cdot {ft(v1)} = {ft(x1)}",
                   f"g_2 = {ft(g2)},\\ v_2 = {ft(beta)}\\cdot {ft(v1)} + {ft(g2)} = {ft(v2)},\\ x_2 = {ft(x1)} - {fp(eta)}\\cdot {ft(v2)} = {ft(x2)}")


@lesson("5.4")
def nesterov_lookahead():
    eta, beta = F(1, 4), F(1, 2)
    x0, v0 = F(4), F(2)
    classic = x0
    look = x0 - eta * beta * v0
    check(look != classic, "điểm nhìn trước khác")
    return display(f"x_0 = {ft(x0)},\\ v_0 = {ft(v0)},\\ \\eta = {ft(eta)},\\ \\beta = {ft(beta)},\\ f(x) = \\tfrac{{1}}{{2}}x^2",
                   f"\\text{{momentum: }} g_1 = f'(x_0) = {ft(x0)}", f"\\text{{Nesterov: }} g_1 = f'\\big(x_0 - \\eta\\beta v_0\\big) = f'({ft(look)}) = {ft(look)}") \
        + "\n\nNesterov tính gradient tại vị trí mà momentum sắp đưa tới, nên \"nhìn trước\" và hiệu chỉnh sớm hơn."


@lesson("5.4")
def mistake_v0():
    eta, beta = F(1, 2), F(9, 10)
    g1 = F(4)
    v1 = beta * 0 + g1
    wrong = beta * g1  # nhầm v_1 = beta*g_1
    return f"**Sai:** coi $v_1 = \\beta g_1 = {ft(wrong)}$ (hoặc nhầm $v_0 = g_1$).\n\n**Đúng:** $v_0 = 0$ nên $v_1 = \\beta\\cdot 0 + g_1 = {ft(v1)}$: bước đầu tiên của momentum **giống hệt** gradient descent."


# ------------------------------------------------------------------ 5.5
@lesson("5.5")
def ema_bias():
    beta = F(9, 10)
    g = F(10)
    m1 = (1 - beta) * g
    mh1 = m1 / (1 - beta)
    check(mh1 == g, "m_hat_1 = g_1")
    g2 = F(20)
    m2 = beta * m1 + (1 - beta) * g2
    mh2 = m2 / (1 - beta**2)
    w = (beta * g + g2) / (1 + beta)
    check(w == mh2, "trung bình có trọng số")
    return display(f"\\beta_1 = {ft(beta)},\\ g_1 = {ft(g)},\\ g_2 = {ft(g2)}",
                   f"m_1 = (1-\\beta_1)g_1 = {ft(m1)},\\qquad \\hat{{m}}_1 = \\dfrac{{m_1}}{{1 - \\beta_1}} = {ft(mh1)}",
                   f"m_2 = {ft(beta)}\\cdot {ft(m1)} + {ft(1 - beta)}\\cdot {ft(g2)} = {ft(m2)},\\qquad \\hat{{m}}_2 = \\dfrac{{m_2}}{{1 - \\beta_1^2}} = {ft(mh2)}") \
        + f"\n\nCó thể kiểm tra: $\\hat{{m}}_2 = \\dfrac{{\\beta_1 g_1 + g_2}}{{1 + \\beta_1}} = {ft(w)}$, là trung bình có trọng số của $g_1, g_2$."


@lesson("5.5")
def adam_first_step():
    eta = F(1, 10)
    rows = []
    for g in (F(3), F(-50), F(1, 100)):
        m, v = (1 - F(9, 10)) * g, (1 - F(999, 1000)) * g * g
        mh, vh = m / (1 - F(9, 10)), v / (1 - F(999, 1000))
        s = mh / F(math.isqrt(vh.numerator), math.isqrt(vh.denominator))
        check(abs(s) == 1, "bước đầu có độ lớn 1")
        rows.append([ft(g), ft(mh), ft(vh), ft(s), ft(-eta * s)])
    return display("\\beta_1 = \\tfrac{9}{10},\\ \\beta_2 = \\tfrac{999}{1000},\\ \\epsilon = 0,\\ \\eta = " + ft(eta),
                   table(["g_1", "\\hat{m}_1", "\\hat{v}_1", "\\hat{m}_1/\\sqrt{\\hat{v}_1}", "\\Delta\\theta"], rows)) \
        + "\n\nBước đầu của Adam luôn có độ lớn $\\eta$ bất kể gradient lớn hay nhỏ: Adam chuẩn hóa theo từng tham số."


@lesson("5.5")
def adagrad_345():
    eta = F(1, 2)
    gs = [F(3), F(4)]
    th = F(0)
    G = F(0)
    lines = []
    for t, g in enumerate(gs, 1):
        G += g * g
        r = F(math.isqrt(G.numerator))
        check(r * r == G, "G chính phương")
        th -= eta * g / r
        lines.append(f"G_{t} = {ft(G)},\\ \\sqrt{{G_{t}}} = {ft(r)},\\ \\theta_{t} = {ft(th)}")
    return display(f"\\eta = {ft(eta)},\\ \\theta_0 = 0,\\ g_1 = 3,\\ g_2 = 4", *lines) + "\n\nBước đầu có độ lớn $\\eta$; bước hai nhỏ hơn vì $G$ đã tích lũy: AdaGrad giảm learning rate hiệu dụng theo thời gian."


@lesson("5.5")
def rmsprop_forgets():
    rho, eta = F(3, 4), F(1, 2)
    gs = [F(4), F(2)]
    s = F(0)
    lines = []
    th = F(0)
    for t, g in enumerate(gs, 1):
        s = rho * s + (1 - rho) * g * g
        r = F(math.isqrt(s.numerator), math.isqrt(s.denominator))
        check(r * r == s, "căn chính xác")
        th -= eta * g / r
        lines.append(f"s_{t} = {ft(s)},\\ \\sqrt{{s_{t}}} = {ft(r)},\\ \\theta_{t} = {ft(th)}")
    return display(f"\\rho = {ft(rho)},\\ \\eta = {ft(eta)},\\ \\theta_0 = 0,\\ g_1 = 4,\\ g_2 = 2", *lines)


@lesson("5.5")
def mistake_bias_correction():
    beta1 = F(9, 10)
    g = F(10)
    m1 = (1 - beta1) * g
    return f"**Sai:** dùng $m_1 = {ft(m1)}$ trực tiếp trong cập nhật (và vẫn hiệu chỉnh $\\hat{{v}}$): bước đầu chỉ bằng $1/10$ đúng kích cỡ. Nếu bỏ hiệu chỉnh cả $m$ lẫn $v$ thì bước đầu lại bị phóng đại (với $\\beta_2$ gần $1$, $v_1$ rất nhỏ nên chia cho $\\sqrt{{v_1}}$ làm bước lớn hơn mức đúng).\n\n**Đúng:** hiệu chỉnh $\\hat{{m}}_1 = m_1/(1-\\beta_1) = {ft(m1 / (1 - beta1))}$ trước khi dùng. Vì khởi tạo $m_0 = 0$, các giá trị đầu bị kéo về $0$."


# ------------------------------------------------------------------ 5.6
@lesson("5.6")
def schedules_table():
    eta0 = F(1, 10)
    rows = []
    for t in range(0, 13, 2):
        step = eta0 * F(1, 2) ** (t // 4)
        expo = eta0 * F(9, 10) ** t
        inv = eta0 / (1 + F(1, 2) * t)
        rows.append([str(t), ft(step), ft(expo), ft(inv)])
    return display("\\eta_0 = " + ft(eta0), table(["t", "\\text{step } (s=4,\\gamma=\\tfrac12)", "\\text{exp } (\\gamma=\\tfrac{9}{10})", "\\text{inverse-time } (k=\\tfrac12)"], rows))


@lesson("5.6")
def cosine_table():
    eta_max, eta_min, T = F(1, 10), F(0), 12
    cosv = {0: F(1), 4: F(1, 2), 6: F(0), 8: F(-1, 2), 12: F(-1)}
    rows = []
    for t, c in cosv.items():
        val = eta_min + (eta_max - eta_min) * (1 + c) / 2
        flt = float(eta_min) + 0.5 * float(eta_max - eta_min) * (1 + math.cos(math.pi * t / T))
        check(abs(flt - float(val)) < 1e-12, "cosine")
        rows.append([str(t), ft(F(t, T)) + "\\pi", ft(c), ft(val)])
    return display("\\eta_{\\max} = " + ft(eta_max) + ",\\ \\eta_{\\min} = 0,\\ T = 12", table(["t", "\\pi t/T", "\\cos", "\\eta_t"], rows))


@lesson("5.6")
def warmup_cosine():
    eta_max, w, span = F(1, 10), 4, 12
    rows = []
    for t in (1, 2, 4, 10, 16):
        if t < w:
            val = eta_max * F(t, w)
        else:
            ratio = F(t - w, span)
            c = {F(0): F(1), F(1, 2): F(0), F(1): F(-1)}.get(ratio)
            if c is None:
                continue
            val = eta_max * (1 + c) / 2
            check(abs(float(val) - 0.5 * float(eta_max) * (1 + math.cos(math.pi * float(ratio)))) < 1e-12, "warmup+cosine")
        rows.append([str(t), ft(val)])
    return display(f"\\eta_{{\\max}} = {ft(eta_max)},\\ W = {w},\\ T = {w + span},\\ \\eta_{{\\min}} = 0", table(["t", "\\eta_t"], rows))


@lesson("5.6")
def noam():
    d, w = 64, 16
    rows = []
    for t in (1, 4, 16, 36, 64):
        a, b = F(1, math.isqrt(t)), F(t, math.isqrt(w) ** 3)
        val = F(1, math.isqrt(d)) * min(a, b)
        flt = d ** -0.5 * min(t ** -0.5, t * w ** -1.5)
        check(abs(flt - float(val)) < 1e-12, "Noam")
        rows.append([str(t), ft(a), ft(b), ft(val)])
    return display(f"d = {d},\\ W = {w},\\ \\eta_t = d^{{-1/2}}\\min(t^{{-1/2}},\\ tW^{{-3/2}})", table(["t", "t^{-1/2}", "tW^{-3/2}", "\\eta_t"], rows)) \
        + "\n\nGiá trị tăng tuyến tính đến $t = W$, đạt đỉnh, rồi giảm theo $1/\\sqrt{t}$."


@lesson("5.6")
def mistake_floor():
    eta0, g, s, t = F(1, 2), F(1, 2), 4, 9
    wrong = eta0 * g ** F(t, s)
    right = eta0 * g ** (t // s)
    return f"**Sai:** thay $\\lfloor t/s \\rfloor$ bằng $t/s$: $\\eta_{{{t}}} = {ft(eta0)}\\cdot {ft(g)}^{{{ft(F(t, s))}}}$ (số mũ phân số, learning rate đổi liên tục).\n\n**Đúng:** step decay giảm theo bậc thang: $\\lfloor {t}/{s} \\rfloor = {t // s}$ nên $\\eta_{{{t}}} = {ft(eta0)}\\cdot {ft(g)}^{{{t // s}}} = {ft(right)}$."


# ------------------------------------------------------------------ 5.7
@lesson("5.7")
def ridge_small():
    xs, ys = [F(1), F(2), F(3)], [F(2), F(3), F(5)]
    sxy = sum(a * b for a, b in zip(xs, ys))
    sxx = sum(a * a for a in xs)
    rows = []
    for lam in (F(0), F(2), F(14), F(100)):
        w = sxy / (sxx + lam)
        rows.append([ft(lam), ft(w)])
    vals = [sxy / (sxx + lam) for lam in (F(0), F(2), F(14), F(100))]
    check(all(vals[i] > vals[i + 1] > 0 for i in range(3)), "w co dần về 0")
    return display("x = (1, 2, 3),\\ y = (2, 3, 5):\\quad \\sum xy = " + ft(sxy) + ",\\ \\sum x^2 = " + ft(sxx),
                   "w^\\star = \\dfrac{\\sum x_i y_i}{\\sum x_i^2 + \\lambda}", table(["\\lambda", "w^\\star"], rows)) + "\n\n$\\lambda$ càng lớn, $w^\\star$ càng bị co về $0$."


@lesson("5.7")
def weight_decay_form():
    eta, lam = F(1, 10), F(1, 2)
    w, e, xv = F(5), F(2), F(3)
    g = e * xv
    a = w - eta * (g + lam * w)
    b = (1 - eta * lam) * w - eta * g
    check(a == b, "hai cách viết")
    return display(f"w = {ft(w)},\\ e\\,x = {ft(g)},\\ \\eta = {ft(eta)},\\ \\lambda = {ft(lam)}",
                   f"w - \\eta\\,(ex + \\lambda w) = {ft(a)} = (1 - \\eta\\lambda)\\,w - \\eta\\,ex = {ft(b)}")


@lesson("5.7")
def l1_soft_threshold():
    lam = F(3)
    rows = []
    for z in (F(-5), F(-2), F(0), F(2), F(7)):
        w = F((z > 0) - (z < 0)) * max(abs(z) - lam, F(0))
        ok = abs(z) <= lam if w == 0 else (w - z) + lam * ((w > 0) - (w < 0)) == 0
        check(ok, "tối ưu")
        rows.append([ft(z), ft(w)])
    return display("S_\\lambda(z) = \\mathrm{sign}(z)\\max(|z| - \\lambda,\\ 0),\\quad \\lambda = " + ft(lam), table(["z", "S_\\lambda(z)"], rows)) + "\n\nMọi $|z| \\le \\lambda$ bị đưa về đúng $0$: đó là lý do L1 cho nghiệm **thưa**."


@lesson("5.7")
def adam_l2_vs_adamw():
    eta, lam = F(1, 10), F(1, 2)
    th, g = F(4), F(-3)
    gp = g + lam * th
    s_l2 = (gp > 0) - (gp < 0)
    s_w = (g > 0) - (g < 0)
    new_l2 = th - eta * s_l2
    new_w = th - eta * (s_w + lam * th)
    check(new_l2 != new_w, "hai cách khác nhau")
    return display(f"\\theta_0 = {ft(th)},\\ g_1 = {ft(g)},\\ \\eta = {ft(eta)},\\ \\lambda = {ft(lam)},\\ \\epsilon = 0",
                   f"\\text{{Adam + L2: }} g' = g + \\lambda\\theta = {ft(gp)},\\ \\mathrm{{sign}} = {s_l2},\\ \\theta_1 = {ft(new_l2)}",
                   f"\\text{{AdamW: }} \\mathrm{{sign}}(g) = {s_w},\\ \\theta_1 = {ft(th)} - {fp(eta)}\\left({s_w} + {fp(lam)}\\cdot {ft(th)}\\right) = {ft(new_w)}") \
        + "\n\nTrong Adam + L2, số hạng $\\lambda\\theta$ bị chuẩn hóa cùng gradient (ở đây còn đổi cả dấu của bước); AdamW tách weight decay ra nên co đều theo $\\theta$."


@lesson("5.7")
def mistake_l1_l2():
    lam, w = F(1, 2), F(4)
    return ("**Sai:** nghĩ rằng L2 cũng làm trọng số bằng đúng $0$ như L1.\n\n"
            + display(f"w \\leftarrow (1 - \\eta\\lambda)\\,w \\;\\Rightarrow\\; \\text{{factor}} = {ft(1 - F(1, 10) * lam)}") +
            "\n\n**Đúng:** L2 co trọng số theo hệ số (không bao giờ đúng bằng $0$ nếu không có gradient kéo), còn L1 (soft-thresholding) cắt hẳn về $0$ khi $|z| \\le \\lambda$.")


def build():
    paths = []
    for code, examples in EX.items():
        paths += write_examples(code, examples)
    return paths
