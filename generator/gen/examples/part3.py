r"""Ví dụ cho Phần 3 (Giải tích). Mọi đạo hàm do SymPy tính và được kiểm bằng sai phân hữu hạn; MDX chỉ chứa kết quả đã tính."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational, diff, exp, hessian, lambdify, log, simplify, sin, symbols

from ..core import latex as L
from ..core.verify import check
from ..topics.calc._c import fd_check, tex
from ..topics.calc.c3_3_7_multivariable import _backward, _code, _forward
from .common import display, write_examples

x, y, z = symbols("x y z")
EX: dict[str, dict] = {}


def lesson(code: str):
    def deco(fn):
        EX.setdefault(code, {})[fn.__name__.replace("_", "-")] = fn
        return fn
    return deco


def _d(f, v=x):
    g = diff(f, v)
    fd_check(f, g, [v] + [s for s in (x, y) if s != v and f.has(s)])
    return g


# ------------------------------------------------------------------ 3.1
@lesson("3.1")
def poly():
    f = 4 * x**3 - 6 * x**2 + x - 2
    g = _d(f)
    return display(f"f(x) = {tex(f)}", f"f'(x) = 3\\cdot 4x^{{2}} - 2\\cdot 6x + 1 - 0 = {tex(g)}")


@lesson("3.1")
def product():
    u, v = x**2, exp(x)
    f = u * v
    g = _d(f)
    return display(f"f(x) = x^2 e^x", f"f'(x) = u'v + uv' = 2x\\cdot e^x + x^2\\cdot e^x = {tex(g)}") + "\n\nVới $u = x^2$ và $v = e^x$."


@lesson("3.1")
def quotient():
    u, v = x**2 + 1, x + 2
    f = u / v
    g = _d(f)
    num_ = diff(u, x) * v - u * diff(v, x)
    return display(f"f(x) = \\dfrac{{{tex(u)}}}{{{tex(v)}}}", f"f'(x) = \\dfrac{{u'v - uv'}}{{v^2}} = \\dfrac{{{tex(num_.expand())}}}{{{tex(v ** 2)}}}")


@lesson("3.1")
def value_at_point():
    f = x**3 * log(x)
    g = _d(f)
    return display(f"f(x) = x^3\\ln x", f"f'(x) = {tex(g)}", f"f'(1) = {tex(g.subs(x, 1))}")


@lesson("3.1")
def prop_linearity():
    f, g = x**3, sin(x)
    a, b = 2, -3
    lhs = diff(a * f + b * g, x)
    rhs = a * diff(f, x) + b * diff(g, x)
    check(simplify(lhs - rhs) == 0, "tuyến tính")
    return display(f"\\dfrac{{d}}{{dx}}\\left({a}x^3 {b:+d}\\sin x\\right) = {tex(lhs)} = {a}\\cdot {tex(diff(f, x))} {b:+d}\\cdot {tex(diff(g, x))}")


@lesson("3.1")
def mistake_product():
    f = x**2 * exp(x)
    wrong = diff(x**2, x) * diff(exp(x), x)
    right = diff(f, x)
    return (display(f"f(x) = x^2 e^x") + f"\n\n**Sai:** $(uv)' = u'v'$, cho $2x\\cdot e^x = {tex(wrong)}$.\n\n"
            + f"**Đúng:** $(uv)' = u'v + uv'$:\n\n" + display(f"f'(x) = {tex(right)}")
            + f"\n\nTại $x = 1$, đáp án sai cho ${tex(wrong.subs(x, 1))}$, đáp án đúng cho ${tex(right.subs(x, 1))}$.")


# ------------------------------------------------------------------ 3.2
@lesson("3.2")
def chain_simple():
    f = (3 * x + 1) ** 5
    g = _d(f)
    return display("f(x) = (3x+1)^5", "f'(x) = 5(3x+1)^4\\cdot (3x+1)' = 5(3x+1)^4\\cdot 3", f"f'(x) = {tex(g)}")


@lesson("3.2")
def chain_nested():
    f = sin(x**2 + 1) ** 3
    g = _d(f)
    return display("f(x) = \\left(\\sin(x^2+1)\\right)^3",
                   "f'(x) = 3\\sin^2(x^2+1)\\cdot \\cos(x^2+1)\\cdot 2x", f"f'(x) = {tex(g)}") + "\n\nBa tầng: lũy thừa ngoài cùng, rồi $\\sin$, rồi $x^2+1$; nhân đạo hàm của từng tầng từ ngoài vào trong."


@lesson("3.2")
def chain_sigmoid():
    f = 1 / (1 + exp(-x))
    g = _d(f)
    s = f
    check(simplify(g - s * (1 - s)) == 0, "σ' = σ(1−σ)")
    return display("\\sigma(t) = \\dfrac{1}{1 + e^{-t}}", "\\sigma'(t) = \\sigma(t)\\,(1 - \\sigma(t))", f"\\sigma'(0) = \\sigma(0)(1 - \\sigma(0)) = \\dfrac{{1}}{{2}}\\cdot\\dfrac{{1}}{{2}} = {tex(g.subs(x, 0))}")


@lesson("3.2")
def chain_loss():
    w = symbols("w")
    xv, yv = 3, 5
    Lw = (w * xv - yv) ** 2
    g = diff(Lw, w)
    fd_check(Lw, g, [w])
    return display(f"L(w) = (wx - y)^2,\\quad x = {xv},\\; y = {yv}", f"\\dfrac{{dL}}{{dw}} = 2(wx - y)\\cdot x = {tex(g)}", f"\\dfrac{{dL}}{{dw}}\\Big|_{{w=0}} = {tex(g.subs(w, 0))}")


@lesson("3.2")
def prop_numeric_check():
    f = sin(x**2)
    g = diff(f, x)
    a = 1.3
    h = 1e-6
    fn = lambdify(x, f)
    gn = lambdify(x, g)
    num_ = (fn(a + h) - fn(a - h)) / (2 * h)
    check(abs(num_ - gn(a)) < 1e-6, "sai phân")
    return display(f"f(x) = \\sin(x^2),\\quad f'(x) = {tex(g)}") + f"\n\nKiểm tra số tại $x = {a}$: sai phân hữu hạn $\\dfrac{{f(x+h) - f(x-h)}}{{2h}} \\approx {num_:.6f}$, đạo hàm giải tích cho ${gn(a):.6f}$ (khớp)."


@lesson("3.2")
def mistake_inner():
    f = exp(3 * x)
    wrong = exp(3 * x)
    right = diff(f, x)
    return display("f(x) = e^{3x}") + f"\n\n**Sai:** $(e^{{3x}})' = e^{{3x}}$ (quên đạo hàm của hàm trong).\n\n**Đúng:** $(e^{{3x}})' = e^{{3x}}\\cdot 3 = {tex(right)}$."


# ------------------------------------------------------------------ 3.3
@lesson("3.3")
def partial():
    f = x**2 * y + 3 * x * y**2 - y
    fx, fy = diff(f, x), diff(f, y)
    return display(f"f(x, y) = {tex(f)}", f"\\dfrac{{\\partial f}}{{\\partial x}} = {tex(fx)}", f"\\dfrac{{\\partial f}}{{\\partial y}} = {tex(fy)}") + \
        "\n\nKhi lấy đạo hàm riêng theo $x$, coi $y$ là một hằng số (và ngược lại)."


@lesson("3.3")
def gradient_value():
    f = x**2 * y + y**3
    pt = (2, -1)
    g = [diff(f, v).subs({x: pt[0], y: pt[1]}) for v in (x, y)]
    return display(f"f(x, y) = {tex(f)}", f"\\nabla f(x, y) = ({tex(diff(f, x))},\\; {tex(diff(f, y))})", f"\\nabla f{pt} = ({g[0]},\\; {g[1]})")


@lesson("3.3")
def steepest():
    f = x**2 + 2 * y**2
    pt = Matrix([1, 1])
    g = Matrix([diff(f, v).subs({x: 1, y: 1}) for v in (x, y)])
    h = Rational(1, 100)
    along = lambda d: f.subs({x: 1 + h * d[0], y: 1 + h * d[1]}) - f.subs({x: 1, y: 1})
    unit_g = g / g.norm()
    d1 = along([float(unit_g[0]), float(unit_g[1])])
    d2 = along([1.0, 0.0])
    check(float(d1) > float(d2) > 0, "hướng gradient tăng nhanh nhất")
    return display(f"f(x, y) = {tex(f)},\\quad \\nabla f(1, 1) = ({g[0]}, {g[1]})") + \
        f"\n\nDi chuyển một đoạn nhỏ $0{{,}}01$ từ $(1, 1)$: theo hướng gradient $f$ tăng khoảng ${float(d1):.5f}$, còn theo hướng trục $x$ chỉ tăng ${float(d2):.5f}$. Gradient chỉ hướng $f$ tăng nhanh nhất."


@lesson("3.3")
def prop_zero_gradient():
    f = (x - 2) ** 2 + (y + 1) ** 2
    g = [diff(f, v) for v in (x, y)]
    at = [t.subs({x: 2, y: -1}) for t in g]
    check(all(v == 0 for v in at), "gradient 0 tại cực tiểu")
    return display(f"f(x, y) = {tex(f)},\\quad \\nabla f = ({tex(g[0])},\\; {tex(g[1])})", f"\\nabla f(2, -1) = ({at[0]},\\; {at[1]})") + "\n\nTại điểm cực tiểu $(2, -1)$ gradient bằng $\\mathbf{0}$."


@lesson("3.3")
def mistake_partial():
    f = x * y**2
    return display(f"f(x, y) = x y^2") + f"\n\n**Sai:** $\\partial f/\\partial x = y^2 + 2xy$ (coi $y$ cũng thay đổi theo $x$).\n\n**Đúng:** coi $y$ là hằng số: $\\partial f/\\partial x = {tex(diff(f, x))}$, còn $\\partial f/\\partial y = {tex(diff(f, y))}$."


# ------------------------------------------------------------------ 3.4
@lesson("3.4")
def jacobian():
    F = Matrix([x * y, x + y**2])
    J = F.jacobian([x, y])
    pt = {x: 2, y: 3}
    return display(f"\\mathbf{{F}}(x, y) = {L.expr_mat([[tex(f)] for f in F])}", f"\\mathbf{{J}} = {L.expr_mat([[tex(J[i, j]) for j in range(2)] for i in range(2)])}",
                   f"\\mathbf{{J}}(2, 3) = {L.mat(J.subs(pt))}")


@lesson("3.4")
def jacobian_chain():
    F = Matrix([x + y, x * y])
    a, b = symbols("a b")
    G = Matrix([a**2, a + b])
    comp = F.subs({x: a**2, y: a + b}, simultaneous=True)
    Jc = comp.jacobian([a, b]).subs({a: 1, b: 2})
    JF = F.jacobian([x, y]).subs({x: 1, y: 3})
    JG = G.jacobian([a, b]).subs({a: 1, b: 2})
    check(Jc == JF * JG, "Jacobian của hợp = tích Jacobian")
    return display(f"\\mathbf{{J}}_{{\\mathbf{{F}}\\circ\\mathbf{{G}}}} = \\mathbf{{J}}_{{\\mathbf{{F}}}}\\mathbf{{J}}_{{\\mathbf{{G}}}} = {L.mat(JF)}{L.mat(JG)} = {L.mat(Jc)}") + \
        "\n\nTính tại $(a, b) = (1, 2)$, khi đó $\\mathbf{G}(1, 2) = (1, 3)$. Đây chính là quy tắc chuỗi dạng ma trận."


@lesson("3.4")
def prop_linear_map():
    W = Matrix([[2, -1, 0], [1, 3, 4]])
    xs = symbols("x1:4")
    F = W * Matrix(xs)
    check(F.jacobian(Matrix(xs)) == W, "J = W")
    return display(f"\\mathbf{{F}}(\\mathbf{{x}}) = \\mathbf{{W}}\\mathbf{{x}} = {L.expr_mat([[tex(f)] for f in F])},\\quad \\mathbf{{J}} = {L.mat(W)} = \\mathbf{{W}}")


@lesson("3.4")
def mistake_shape():
    F = Matrix([x * y, x + y, x - y])
    J = F.jacobian([x, y])
    return display(f"\\mathbf{{F}}:\\; \\R^{{2}} \\to \\R^{{3}},\\quad \\mathbf{{J}} = {L.expr_mat([[tex(J[i, j]) for j in range(2)] for i in range(3)])}") + \
        f"\n\n**Sai:** xếp $\\mathbf{{J}}$ cỡ $2\\times 3$ (hàng theo biến). **Đúng:** hàng ứng với **đầu ra**, cột ứng với **đầu vào**: $\\mathbf{{J}}$ cỡ ${J.rows}\\times {J.cols}$ (số đầu ra nhân số đầu vào)."


# ------------------------------------------------------------------ 3.5
@lesson("3.5")
def hessian_example():
    f = x**3 + x * y**2 - 3 * x * y
    H = hessian(f, (x, y))
    return display(f"f(x, y) = {tex(f)}", f"\\mathbf{{H}} = {L.expr_mat([[tex(H[i, j]) for j in range(2)] for i in range(2)])}", f"\\mathbf{{H}}(1, 2) = {L.mat(H.subs({x: 1, y: 2}))}")


@lesson("3.5")
def classify_min():
    f = x**2 + x * y + y**2 - 3 * x
    g = [diff(f, v) for v in (x, y)]
    sol = {x: 2, y: -1}
    check(all(t.subs(sol) == 0 for t in g), "điểm dừng")
    H = hessian(f, (x, y))
    return display(f"f(x, y) = {tex(f)}", f"\\nabla f = ({tex(g[0])},\\; {tex(g[1])}) = \\mathbf{{0}} \\Rightarrow (x, y) = (2, -1)", f"\\mathbf{{H}} = {L.mat(H)},\\quad \\det = {H.det()},\\; f_{{xx}} = {H[0, 0]}") + \
        "\n\n$\\det > 0$ và $f_{xx} > 0$ nên Hessian xác định dương: điểm dừng là **cực tiểu địa phương**."


@lesson("3.5")
def classify_saddle():
    f = x**2 - y**2
    H = hessian(f, (x, y))
    return display(f"f(x, y) = {tex(f)},\\quad \\nabla f(0, 0) = \\mathbf{{0}},\\quad \\mathbf{{H}} = {L.mat(H)}") + "\n\nHai trị riêng trái dấu ($2$ và $-2$): $\\mathbf{0}$ là **điểm yên ngựa**: $f$ tăng dọc trục $x$ nhưng giảm dọc trục $y$."


@lesson("3.5")
def prop_symmetry():
    f = x**3 * y**2 + sin(x * y)
    fxy, fyx = diff(f, x, y), diff(f, y, x)
    check(simplify(fxy - fyx) == 0, "Schwarz")
    return display(f"f(x, y) = {tex(f)}", f"f_{{xy}} = {tex(fxy)},\\qquad f_{{yx}} = {tex(fyx)}") + "\n\nHai đạo hàm hỗn hợp bằng nhau, nên Hessian đối xứng."


@lesson("3.5")
def mistake_second():
    f = x**4 + y**2
    H = hessian(f, (x, y))
    return display(f"f(x, y) = {tex(f)}") + "\n\n**Sai:** kết luận điểm dừng $(0, 0)$ là cực tiểu chỉ vì Hessian tại đó có phần tử dương. " + \
        f"Hessian tại $(0, 0)$ là ${L.mat(H.subs({x: 0, y: 0}))}$ (suy biến: $\\det = 0$), nên tiêu chuẩn **chưa kết luận được**; cần xét bậc cao hơn. (Ở đây $f \\ge 0$ nên thực ra là cực tiểu, nhưng không suy ra được từ Hessian.)"


# ------------------------------------------------------------------ 3.6
@lesson("3.6")
def grad_quadratic():
    A = Matrix([[2, 1], [3, 4]])
    xv = Matrix([1, -1])
    g = (A + A.T) * xv
    xs = symbols("x1 x2")
    f = (Matrix(xs).T * A * Matrix(xs))[0, 0]
    check(Matrix([diff(f, v) for v in xs]).subs(dict(zip(xs, xv))) == g, "∇ xᵀAx")
    return display(f"f(\\mathbf{{x}}) = \\mathbf{{x}}\\T\\mathbf{{A}}\\mathbf{{x}} = {tex(f)}", f"\\nabla f = (\\mathbf{{A}} + \\mathbf{{A}}\\T)\\mathbf{{x}} = {L.mat(A + A.T)}{L.mat(xv)} = {L.mat(g)}")


@lesson("3.6")
def grad_least_squares():
    X = Matrix([[1, 0], [1, 1], [1, 2]])
    yv = Matrix([1, 3, 4])
    w = (X.T * X).inv() * X.T * yv
    g = X.T * (X * w - yv)
    check(g == Matrix([0, 0]), "∇ = 0 tại nghiệm")
    return display(f"\\nabla L = \\mathbf{{X}}\\T(\\mathbf{{X}}\\mathbf{{w}} - \\mathbf{{y}})", f"\\nabla L = \\mathbf{{0}} \\;\\Rightarrow\\; \\mathbf{{X}}\\T\\mathbf{{X}}\\mathbf{{w}} = \\mathbf{{X}}\\T\\mathbf{{y}} \\;\\Rightarrow\\; \\mathbf{{w}} = {L.mat(w)}") + \
        "\n\nĐặt gradient bằng $\\mathbf{0}$ cho lại phương trình chuẩn của bài 1.15."


@lesson("3.6")
def prop_linear_grad():
    a = Matrix([3, -2, 5])
    xs = symbols("x1:4")
    f = (a.T * Matrix(xs))[0, 0]
    check(Matrix([diff(f, v) for v in xs]) == a, "∇ aᵀx = a")
    return display(f"f(\\mathbf{{x}}) = \\mathbf{{a}}\\T\\mathbf{{x}} = {tex(f)},\\quad \\nabla f = {L.mat(a)} = \\mathbf{{a}}")


@lesson("3.6")
def mistake_transpose():
    A = Matrix([[1, 2], [0, 3]])
    xv = Matrix([1, 1])
    wrong = 2 * A * xv
    right = (A + A.T) * xv
    return display(f"f(\\mathbf{{x}}) = \\mathbf{{x}}\\T\\mathbf{{A}}\\mathbf{{x}},\\quad \\mathbf{{A}} = {L.mat(A)}") + \
        f"\n\n**Sai:** $\\nabla f = 2\\mathbf{{A}}\\mathbf{{x}} = {L.vec_inline(wrong)}$ (chỉ đúng khi $\\mathbf{{A}}$ đối xứng).\n\n**Đúng:** $\\nabla f = (\\mathbf{{A}} + \\mathbf{{A}}\\T)\\mathbf{{x}} = {L.vec_inline(right)}$."


# ------------------------------------------------------------------ 3.7
def _graph_example(names, vals, nodes):
    v = _forward(vals, nodes)
    g, steps = _backward(v, nodes, names)
    return v, g, steps


@lesson("3.7")
def graph_forward_backward():
    names = ["a", "b"]
    vals = {"a": Rational(2), "b": Rational(-3)}
    nodes = [("u", "mul", ("a", "b")), ("v", "add", ("u", "a")), ("L", "sq", ("v",))]
    v, g, steps = _graph_example(names, vals, nodes)
    code = _code(names, vals, nodes)
    back = "\n".join(f"- Nút `{out}`: gradient đến nút này là ${go}$; truyền về " + ", ".join(f"`{a}` thêm ${c}$" for a, c in zip(args, contrib)) + "." for out, op, args, go, contrib in steps)
    return (f"```python\n{code}\n```\n\n**Forward:** " + ", ".join(f"`{k}` = {val}" for k, val in v.items() if k not in names) + f".\n\n**Backward** (bắt đầu từ $\\partial L/\\partial L = 1$, đi ngược):\n\n{back}\n\n"
            + f"Kết quả: $\\partial L/\\partial a = {g['a']}$ và $\\partial L/\\partial b = {g['b']}$. Chú ý `a` được dùng ở **hai chỗ** (`u` và `v`) nên nhận gradient cộng dồn từ cả hai.")


@lesson("3.7")
def graph_neuron():
    xv, wv, bv, yv = 2, 1, 1, 5
    zz = wv * xv + bv
    aa = max(0, zz)
    dLda = 2 * (aa - yv)
    return (f"```python\nx = {xv}\nw = {wv}\nb = {bv}\ny = {yv}\n\nz = w * x + b\na = max(0, z)\nL = (a - y) ** 2\n```\n\n"
            f"Forward: $z = {zz}$, $a = {aa}$, $L = ({aa} - {yv})^2 = {(aa - yv) ** 2}$.\n\n"
            + display(f"\\dfrac{{\\partial L}}{{\\partial a}} = 2(a - y) = {dLda}", f"\\dfrac{{\\partial L}}{{\\partial z}} = {dLda}\\cdot \\mathrm{{ReLU}}'(z) = {dLda}\\cdot {1 if zz > 0 else 0}",
                       f"\\dfrac{{\\partial L}}{{\\partial w}} = {dLda}\\cdot x = {dLda * xv},\\qquad \\dfrac{{\\partial L}}{{\\partial b}} = {dLda}\\cdot 1 = {dLda}"))


@lesson("3.7")
def prop_accumulate():
    a = symbols("a")
    f = a * a + a
    return display("L = a\\cdot a + a", f"\\dfrac{{dL}}{{da}} = a + a + 1 = {tex(diff(f, a))}") + "\n\nMỗi lần `a` xuất hiện trong đồ thị cho một đóng góp; gradient là **tổng** các đóng góp."


@lesson("3.7")
def mistake_accumulate():
    names = ["a", "b"]
    vals = {"a": Rational(3), "b": Rational(2)}
    nodes = [("u", "mul", ("a", "b")), ("v", "add", ("u", "a")), ("L", "sq", ("v",))]
    v, g, steps = _graph_example(names, vals, nodes)
    only_first = [c for out, op, args, go, contrib in steps if out == "u" for a_, c in zip(args, contrib) if a_ == "a"][0]
    return (f"```python\n{_code(names, vals, nodes)}\n```\n\n**Sai:** chỉ lấy đóng góp của `a` qua nút `u`, được ${only_first}$.\n\n"
            f"**Đúng:** cộng cả đóng góp qua nút `v` (nơi `a` được cộng trực tiếp): $\\partial L/\\partial a = {g['a']}$.")


def build():
    paths = []
    for code, examples in EX.items():
        paths += write_examples(code, examples)
    return paths
