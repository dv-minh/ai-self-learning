"""Bài 3.1 Đạo hàm, 3.2 Quy tắc chuỗi: deriv-poly, deriv-rules, deriv-eval, chain-rule."""
from __future__ import annotations

from ._c import (Answer, Matrix, Prompt, Rational, Step, TopicSpec, cb, check, cos, diff, display_f, exp, expr_answer, fd_check, log, num, random,
                 sin, sqrt, tex, to_input, x)
from ._c import L

L31, L32 = "3.1", "3.2"


def _c(rng, lo=-9, hi=9, nonzero=True):
    while True:
        v = rng.randint(lo, hi)
        if v or not nonzero:
            return v


def _poly(rng, terms: int, maxdeg: int, frac: bool = False, neg: bool = False):
    exps = rng.sample(range(0, maxdeg + 1), min(terms, maxdeg + 1))
    e = 0
    for k in exps:
        c = Rational(_c(rng, -9, 9), rng.choice([1, 2, 3, 4]) if frac else 1)
        e += c * x**k
    if neg:
        e += Rational(_c(rng, -6, 6)) * x ** (-1) if rng.random() < 0.5 else Rational(_c(rng, 1, 6)) * sqrt(x)
    return e


def _steps(f, g, rule: str):
    return [Step(text=rule, latex=f"{tex(f)}\\;\\Rightarrow\\; f'(x) = {tex(g)}")]


# ------------------------------------------------------------------ deriv-poly
def gen_deriv_poly(rng: random.Random, level: int, seed: int):
    if level == 1:
        f = _c(rng, -9, 9) * x ** rng.randint(2, 6)
        rule = "Quy tắc lũy thừa: $(x^n)' = n x^{n-1}$, hằng số đứng ngoài:"
    elif level == 2:
        f = _poly(rng, rng.randint(2, 4), 5)
        rule = "Đạo hàm từng số hạng (đạo hàm của tổng là tổng các đạo hàm; hằng số có đạo hàm $0$):"
    else:
        f = _poly(rng, rng.randint(2, 3), 4, frac=True, neg=True)
        rule = "Đạo hàm từng số hạng, dùng $(x^n)' = nx^{n-1}$ cả khi $n$ âm hoặc phân số (ví dụ $\\sqrt{x} = x^{1/2}$, $1/x = x^{-1}$):"
    g = diff(f, x)
    fd_check(f, g, [x])
    return cb(type_="deriv-poly", lesson=L31, level=level, seed=seed,
              prompt=Prompt(text="Tính đạo hàm $f'(x)$. Nhập biểu thức theo cú pháp như `3x^2 + 2x` (dấu `^` là lũy thừa).", display=[display_f("f", "x", f)]),
              answer=expr_answer(g, ["x"]),
              hints=["$(x^n)' = n\\,x^{n-1}$ và $(c)' = 0$.", "Đạo hàm là tuyến tính: đạo hàm từng số hạng rồi cộng lại."],
              steps=_steps(f, g, rule), verified=["sympy", "finite-difference"], tags=["poly"])


# ----------------------------------------------------------------- deriv-rules
def _fn_pool(rng):
    k = _c(rng, 2, 4)
    return [(x**rng.randint(1, 3) + _c(rng, 1, 5), "đa thức"), (exp(rng.choice([1, 2, 3, -1]) * x), "hàm mũ"), (log(x), "logarit"),
            (sin(x), "sin"), (cos(x), "cos"), (x**k, "lũy thừa")]


def gen_deriv_rules(rng: random.Random, level: int, seed: int):
    while True:
        if level == 2:
            u = x**rng.randint(1, 3) + _c(rng, -5, 5)
            v = x**rng.randint(1, 3) + _c(rng, -5, 5)
            kind = rng.choice(["product", "quotient"])
        else:
            pool = _fn_pool(rng)
            (u, _), (v, _) = rng.sample(pool, 2)
            kind = rng.choice(["product", "quotient"])
            if level == 4:
                u = u * x**rng.randint(1, 2)
        f = u * v if kind == "product" else u / v
        if kind == "quotient" and v.has(sin, cos) and False:
            continue
        break
    g = diff(f, x)
    fd_check(f, g, [x])
    if kind == "product":
        rule = f"Quy tắc tích $(uv)' = u'v + uv'$ với $u = {tex(u)}$, $v = {tex(v)}$:"
    else:
        rule = f"Quy tắc thương $(u/v)' = (u'v - uv')/v^2$ với $u = {tex(u)}$, $v = {tex(v)}$:"
    return cb(type_="deriv-rules", lesson=L31, level=level, seed=seed,
              prompt=Prompt(text="Tính $f'(x)$ bằng quy tắc tích hoặc thương. Nhập biểu thức (dùng `exp(x)`, `ln(x)`, `sin(x)`, `cos(x)`).", display=[display_f("f", "x", f)]),
              answer=expr_answer(g, ["x"]),
              hints=["Tích: $(uv)' = u'v + uv'$.", "Thương: $(u/v)' = (u'v - uv')/v^2$.", "Đạo hàm cơ bản: $(e^x)' = e^x$, $(\\ln x)' = 1/x$, $(\\sin x)' = \\cos x$, $(\\cos x)' = -\\sin x$."],
              steps=_steps(f, g, rule), verified=["sympy", "finite-difference"], tags=[kind])


# ------------------------------------------------------------------ deriv-eval
def gen_deriv_eval(rng: random.Random, level: int, seed: int):
    if level == 5:
        kind = rng.choice(["relu", "sigmoid", "tanh", "softplus", "mse"])
        xv, bv, wv, yv = _c(rng, -4, 4), _c(rng, -3, 3), _c(rng, -3, 3), _c(rng, -4, 4)
        if kind == "relu":
            z = wv * xv + bv
            ans = xv if z > 0 else 0
            text = f"ReLU: $a = \\max(0, wx + b)$ với $x = {xv}$, $b = {bv}$. Tính $\\dfrac{{\\partial a}}{{\\partial w}}$ tại $w = {wv}$ (quy ước đạo hàm của ReLU tại $0$ là $0$)."
            step = f"$wx+b = {z}$, " + ("dương nên $a = wx + b$ và $\\partial a/\\partial w = x$." if z > 0 else "không dương nên $a = 0$ và đạo hàm bằng $0$.")
        elif kind == "sigmoid":
            ans = Rational(xv, 4)
            text = f"Sigmoid $\\sigma(t) = 1/(1+e^{{-t}})$. Tính $\\dfrac{{\\partial}}{{\\partial w}}\\sigma(wx)$ tại $w = 0$ với $x = {xv}$."
            step = "$\\sigma'(t) = \\sigma(t)(1-\\sigma(t))$ và $\\sigma(0) = 1/2$, nên $\\sigma'(0) = 1/4$; chain rule nhân thêm $x$."
        elif kind == "tanh":
            ans = Rational(xv)
            text = f"$\\tanh$ có $\\tanh'(t) = 1 - \\tanh^2(t)$. Tính $\\dfrac{{\\partial}}{{\\partial w}}\\tanh(wx)$ tại $w = 0$ với $x = {xv}$."
            step = "$\\tanh(0) = 0$ nên $\\tanh'(0) = 1$; chain rule nhân thêm $x$."
        elif kind == "softplus":
            ans = Rational(xv, 2)
            text = f"Softplus $s(t) = \\ln(1 + e^{{t}})$ có $s'(t) = \\sigma(t)$. Tính $\\dfrac{{\\partial}}{{\\partial w}}s(wx)$ tại $w = 0$ với $x = {xv}$."
            step = "$s'(0) = \\sigma(0) = 1/2$; chain rule nhân thêm $x$."
        else:
            ans = Rational(wv * xv + bv - yv) * xv
            text = f"Mất mát $L = \\frac{{1}}{{2}}(wx + b - y)^2$ với $x = {xv}$, $b = {bv}$, $y = {yv}$. Tính $\\dfrac{{\\partial L}}{{\\partial w}}$ tại $w = {wv}$."
            step = f"$\\partial L/\\partial w = (wx+b-y)\\,x = ({wv * xv + bv - yv})\\cdot {xv}$."
        return cb(type_="deriv-eval", lesson=L31, level=5, seed=seed, prompt=Prompt(text=text),
                  answer=Answer(kind="number", value=str(Rational(ans))),
                  hints=["Dùng chain rule: đạo hàm hàm ngoài nhân đạo hàm hàm trong."], steps=[Step(text=step)], verified=["sympy"], tags=["ml", kind])
    a = _c(rng, -4, 4)
    if level == 1:
        f = _poly(rng, 3, 4)
    elif level == 2:
        f = (_c(rng, 1, 4) * x**2 + _c(rng, -4, 4)) / (x + abs(a) + rng.choice([1, 2]))
    elif level == 3:  # hàm mũ và logarit, đánh giá tại điểm mà $e^0 = 1$ hoặc $\ln 1 = 0$
        if rng.random() < 0.5:
            f, a = _c(rng, -9, 9) * x ** rng.randint(1, 4) * log(x), 1
        else:
            f, a = (x + _c(rng, -5, 5)) * exp(_c(rng, -2, 3) * x) * _c(rng, 1, 4), 0
    else:
        c = _c(rng, 1, 5)
        f, a = rng.choice([
            (c * exp(_c(rng, 1, 3) * x**2 + x), 0), (c * log(x**2 + _c(rng, 1, 5)), 0), (c * (x**2 + 1) ** rng.randint(3, 5), rng.choice([0, 1])),
            (c * exp(x) * (x**2 + _c(rng, 1, 4)), 0), (c * log(x) * (x**2 + _c(rng, 1, 4)), 1), (c * log(x) * x ** rng.randint(2, 4), 1),
        ])
    g = diff(f, x)
    val = Rational(g.subs(x, a).simplify())
    fd_check(f, g, [x])
    return cb(type_="deriv-eval", lesson=L31, level=level, seed=seed,
              prompt=Prompt(text=f"Tính giá trị đạo hàm $f'({a})$ (một số hữu tỉ).", display=[display_f("f", "x", f)]),
              answer=Answer(kind="number", value=str(val)),
              hints=["Tính $f'(x)$ trước rồi thay $x$.", "$e^{0} = 1$ và $\\ln 1 = 0$."],
              steps=[Step(text="Đạo hàm:", latex=f"f'(x) = {tex(g)}"), Step(text=f"Thay $x = {a}$:", latex=f"f'({a}) = {num(val)}")],
              verified=["sympy", "finite-difference"], tags=["value"])


# ------------------------------------------------------------------ chain-rule
def gen_chain(rng: random.Random, level: int, seed: int):
    a, b = _c(rng, 2, 5), _c(rng, -5, 5)
    inner_lin = a * x + b
    if level == 2:
        f = inner_lin ** rng.randint(3, 6) if rng.random() < 0.7 else sqrt(inner_lin + abs(b) + 3)
    elif level == 3:
        outer = rng.choice([exp, sin, cos, lambda u: log(u + abs(b) + 3), lambda u: u**4])
        inner = rng.choice([inner_lin, x**2 + b, x**3 - a * x])
        f = outer(inner)
    elif level == 4:
        f = rng.choice([sin(a * x) ** rng.randint(2, 4), exp(sin(x) * a), log(exp(a * x) + b * b + 1), (x**2 + b * b + 1) ** 3 * exp(a * x), cos(x**2 + b) ** 2,
                        sqrt(x**2 + a) * sin(b * x + 1), exp(a * x) * cos(b * x), (a * x + b) ** 3 * log(x**2 + a)])
    else:  # ML: đạo hàm theo tham số w của một mất mát, các hằng số là số cụ thể
        xv, yv = _c(rng, 1, 5), _c(rng, -5, 5)
        from ._c import symbols
        w = symbols("w")
        f = rng.choice([(w * xv - yv) ** 2, (w * xv - yv) ** 4, exp(-w * xv) * yv, log(1 + exp(w * xv)), (exp(-w * xv) + 1) ** -1])
        g = diff(f, w)
        fd_check(f, g, [w])
        from ._c import Answer as A
        return cb(type_="chain-rule", lesson=L32, level=5, seed=seed,
                  prompt=Prompt(text="Hàm mất mát một tham số $w$ (các hằng số là số cụ thể). Tính $\\dfrac{dL}{dw}$ bằng quy tắc chuỗi. Nhập biểu thức theo `w`.",
                                display=[f"L(w) = {tex(f)}"]),
                  answer=A(kind="expression", value=to_input(g), variables=["w"]),
                  hints=["Đạo hàm hàm ngoài (giữ nguyên hàm trong) nhân đạo hàm của hàm trong theo $w$."],
                  steps=[Step(text="Quy tắc chuỗi:", latex=f"\\dfrac{{dL}}{{dw}} = {tex(g)}")], verified=["sympy", "finite-difference"], tags=["ml"])
    g = diff(f, x)
    fd_check(f, g, [x])
    return cb(type_="chain-rule", lesson=L32, level=level, seed=seed,
              prompt=Prompt(text="Tính $f'(x)$ bằng quy tắc chuỗi $\\dfrac{d}{dx}f(g(x)) = f'(g(x))\\,g'(x)$. Nhập biểu thức.", display=[display_f("f", "x", f)]),
              answer=expr_answer(g, ["x"]),
              hints=["Xác định hàm trong $g(x)$ và hàm ngoài.", "Đạo hàm hàm ngoài tại $g(x)$ rồi nhân với $g'(x)$; với hàm lồng nhiều tầng, nhân liên tiếp."],
              steps=[Step(text="Áp dụng quy tắc chuỗi từ ngoài vào trong:", latex=f"f'(x) = {tex(g)}")], verified=["sympy", "finite-difference"], tags=["chain"])


SPECS = [
    TopicSpec("deriv-poly", L31, (1, 2, 3), gen_deriv_poly),
    TopicSpec("deriv-rules", L31, (2, 3, 4), gen_deriv_rules),
    TopicSpec("deriv-eval", L31, (1, 2, 3, 4, 5), gen_deriv_eval),
    TopicSpec("chain-rule", L32, (2, 3, 4, 5), gen_chain),
]
