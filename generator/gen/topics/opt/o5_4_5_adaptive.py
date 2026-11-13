"""Bài 5.4 Momentum (momentum-steps) và 5.5 Adam, RMSProp, AdaGrad (adam-bias, adam-steps, adagrad-steps, rmsprop-steps).

Với Adam/RMSProp/AdaGrad, đề đặt $\\epsilon = 0$ và chọn gradient sao cho căn bậc hai là số hữu tỉ, để người học tính tay chính xác.
"""
from __future__ import annotations

from functools import lru_cache
from math import isqrt

from sympy import Rational, symbols

from ._o import F, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, fp, ft, lf, nice, num_ans, ob, pick, random, vec_ans, vt

L54, L55 = "5.4", "5.5"
X, Y = symbols("x y")


def _q(v: F) -> Rational:
    return Rational(v.numerator, v.denominator)


def _sqrt(v: F) -> F | None:
    """Căn bậc hai hữu tỉ chính xác (hoặc None)."""
    v = F(v)
    if v < 0:
        return None
    a, b = isqrt(v.numerator), isqrt(v.denominator)
    return F(a, b) if a * a == v.numerator and b * b == v.denominator else None


# ------------------------------------------------------------- momentum-steps
def _quad(rng, level):
    """(mô tả LaTeX, hàm gradient Fraction, biểu thức SymPy, biến, điểm đầu)."""
    if level <= 2:
        a, b = pick(rng, 1, 5), rng.randint(-8, 8)
        return (lf([a, b], ["x^{2}", "x"]), "f(x)", lambda p: [2 * a * p[0] + b], a * X**2 + b * X, [X], [F(pick(rng, -6, 6))])
    if level == 3:
        a, c, d, e = pick(rng, 1, 5), pick(rng, 1, 5), rng.randint(-8, 8), rng.randint(-8, 8)
        return (lf([a, c, d, e], ["x^{2}", "y^{2}", "x", "y"]), "f(x, y)", lambda p: [2 * a * p[0] + d, 2 * c * p[1] + e],
                a * X**2 + c * Y**2 + d * X + e * Y, [X, Y], [F(pick(rng, -5, 5)), F(pick(rng, -5, 5))])
    a, c, b, d, e = pick(rng, 1, 4), pick(rng, 1, 4), rng.randint(-3, 3), rng.randint(-6, 6), rng.randint(-6, 6)
    return (lf([a, b, c, d, e], ["x^{2}", "xy", "y^{2}", "x", "y"]), "f(x, y)", lambda p: [2 * a * p[0] + b * p[1] + d, b * p[0] + 2 * c * p[1] + e],
            a * X**2 + b * X * Y + c * Y**2 + d * X + e * Y, [X, Y], [F(pick(rng, -5, 5)), F(pick(rng, -5, 5))])


def gen_momentum_steps(rng: random.Random, level: int, seed: int):
    nesterov = level == 5
    k = {1: 2, 2: 3, 3: 2, 4: 2, 5: 2}[level]
    for _ in range(500):
        f_tex, head, grad, fexpr, vs, x0 = _quad(rng, level if level < 5 else 4)
        beta = rng.choice([F(1, 2), F(1, 4), F(3, 4), F(4, 5), F(9, 10), F(1, 3)])
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 8), F(1, 20)])
        x, v = list(x0), [F(0)] * len(x0)
        rows = []
        for _t in range(k):
            look = [xi - eta * beta * vi for xi, vi in zip(x, v)] if nesterov else x
            g = grad(look)
            sg = [F(int(_q(F(0)) + fexpr.diff(s).subs(dict(zip(vs, [_q(p) for p in look]))).p), int(fexpr.diff(s).subs(dict(zip(vs, [_q(p) for p in look]))).q)) for s in vs]
            check(g == sg, "gradient viết tay ≠ SymPy")
            v = [beta * vi + gi for vi, gi in zip(v, g)]
            x = [xi - eta * vi for xi, vi in zip(x, v)]
            rows.append((look, g, list(v), list(x)))
        if nice([c for r in rows for part in r for c in part], 500, 200):
            break
    one = len(vs) == 1
    fm = (lambda a: ft(a[0])) if one else vt
    steps = []
    for t, (look, g, vv, xx) in enumerate(rows, 1):
        txt = f"Bước {t}: " + (f"gradient tại điểm nhìn trước $x_{t - 1} - \\eta\\beta v_{t - 1}$." if nesterov else f"gradient tại $x_{t - 1}$.")
        steps.append(Step(text=txt, latex=f"g_{t} = {fm(g)},\\quad v_{t} = {ft(beta)}\\cdot v_{t - 1} + g_{t} = {fm(vv)},\\quad x_{t} = x_{t - 1} - {fp(eta)}\\,v_{t} = {fm(xx)}"))
    if nesterov:
        rule = ["g_t = \\nabla f\\left(x_{t-1} - \\eta\\beta\\,v_{t-1}\\right)", "v_t = \\beta\\,v_{t-1} + g_t", "x_t = x_{t-1} - \\eta\\,v_t"]
        name = "Nesterov momentum"
    else:
        rule = ["g_t = \\nabla f(x_{t-1})", "v_t = \\beta\\,v_{t-1} + g_t", "x_t = x_{t-1} - \\eta\\,v_t"]
        name = "momentum"
    return ob(type_="momentum-steps", lesson=L54, level=level, seed=seed,
              prompt=Prompt(text=f"Chạy {name} ($v_0 = {'0' if one else '\\mathbf{0}'}$) trên hàm dưới đây với $\\beta = {ft(beta)}$, $\\eta = {ft(eta)}$, bắt đầu từ $x_0 = {fm(x0)}$. Tính $x_{k}$. "
                                 + (NUM_HINT if one else VEC_HINT), display=[f"{head} = {f_tex}"] + rule),
              answer=num_ans(x[0]) if one else vec_ans(x),
              hints=["Cập nhật theo đúng thứ tự: tính $g_t$, rồi $v_t$, rồi $x_t$. $v$ tích lũy các gradient cũ nhân $\\beta$.", "Bước đầu $v_0 = 0$ nên $v_1 = g_1$ (giống gradient descent).",
                     "Với Nesterov, gradient được tính tại điểm **nhìn trước** $x_{t-1} - \\eta\\beta v_{t-1}$, không phải tại $x_{t-1}$." if nesterov else "Giữ nguyên $\\eta$ và $\\beta$ ở mọi bước."],
              steps=steps, verified=["sympy-gradient", "fraction-simulation"], tags=["nesterov" if nesterov else "heavy-ball", f"{len(vs)}var"])


# ---------------------------------------------------------------- adam-bias
def gen_adam_bias(rng: random.Random, level: int, seed: int):
    t = {1: 2, 2: 2, 3: 3, 4: 3, 5: 4}[level]
    corrected = level >= 2
    kind = rng.choice(["m", "v"]) if level >= 3 else "m"
    beta = rng.choice([F(1, 2), F(3, 4), F(9, 10), F(4, 5), F(1, 4), F(99, 100)] if level >= 4 else [F(1, 2), F(3, 4), F(1, 4), F(4, 5), F(1, 3)])
    gs = [F(rng.randint(-9, 9)) for _ in range(t)]
    while len(set(gs)) < 2 or 0 in gs:
        gs = [F(rng.randint(-9, 9)) for _ in range(t)]
    base = [g if kind == "m" else g * g for g in gs]
    m = F(0)
    seq = []
    for z in base:
        m = beta * m + (1 - beta) * z
        seq.append(m)
    hat = m / (1 - beta**t)
    # kiểm chứng độc lập: công thức đóng, trung bình có trọng số beta^(t-i)
    closed = (1 - beta) * sum(beta ** (t - 1 - i) * z for i, z in enumerate(base))
    check(closed == m, "công thức đóng EMA sai")
    weighted = sum(beta ** (t - 1 - i) * z for i, z in enumerate(base)) / sum(beta ** (t - 1 - i) for i in range(t))
    check(weighted == hat, "bias correction ≠ trung bình có trọng số")
    ans = hat if corrected else m
    sym = "m" if kind == "m" else "v"
    gsq = "g_t" if kind == "m" else "g_t^2"
    steps = [Step(text=f"Khởi tạo ${sym}_0 = 0$ và cập nhật ${sym}_t = \\beta {sym}_{{t-1}} + (1 - \\beta){gsq}$:",
                  latex=", ".join(f"{sym}_{i + 1} = {ft(s)}" for i, s in enumerate(seq)))]
    if corrected:
        steps.append(Step(text=f"Hiệu chỉnh độ chệch: chia cho $1 - \\beta^{{{t}}} = {ft(1 - beta**t)}$:", latex=f"\\hat{{{sym}}}_{{{t}}} = \\dfrac{{{ft(m)}}}{{{ft(1 - beta**t)}}} = {ft(hat)}"))
    sub = "t"
    target = f"\\hat{{{sym}}}_{{{t}}}" if corrected else f"{sym}_{{{t}}}"
    return ob(type_="adam-bias", lesson=L55, level=level, seed=seed,
              prompt=Prompt(text=f"Trung bình động mũ (EMA) trong Adam: $m_t = \\beta_1 m_{{t-1}} + (1-\\beta_1)g_t$ và $v_t = \\beta_2 v_{{t-1}} + (1-\\beta_2)g_t^2$ với $m_0 = v_0 = 0$"
                                 + (", hiệu chỉnh $\\hat{m}_t = m_t/(1-\\beta_1^t)$, $\\hat{v}_t = v_t/(1-\\beta_2^t)$" if corrected else "")
                                 + f". Gradient lần lượt là $g_1, \\dots, g_{{{t}}} = {', '.join(ft(g) for g in gs)}$. Với "
                                 + (f"$\\beta_1 = {ft(beta)}$" if kind == "m" else f"$\\beta_2 = {ft(beta)}$") + f", tính ${target}$. " + NUM_HINT),
              answer=num_ans(ans),
              hints=["Cập nhật lần lượt từ $t = 1$; luôn bắt đầu từ $0$.", "Bước đầu: $m_1 = (1-\\beta)g_1$ rất nhỏ so với $g_1$ — đó là lý do cần hiệu chỉnh độ chệch." if corrected else "$m_1 = (1-\\beta)g_1$.",
                     ("$\\hat{m}_t$ chính là trung bình có trọng số $\\beta^{t-i}$ của các $g_i$ (tổng trọng số bằng $1$)." if kind == "m" else "$\\hat{v}_t$ chính là trung bình có trọng số $\\beta^{t-i}$ của các $g_i^2$ (tổng trọng số bằng $1$).") if corrected else "Mỗi bước: nhân giá trị cũ với $\\beta$ rồi cộng $(1-\\beta)$ lần giá trị mới."],
              steps=steps, verified=["closed-form", "weighted-average"], tags=[kind, "bias-corrected" if corrected else "raw", f"t{t}"])


# ---------------------------------------------------------------- adam-steps
@lru_cache(maxsize=None)
def _adam_pairs(beta2: F):
    """(g1, g2) nguyên với v_hat_2 = (b2 g1^2 + g2^2)/(1+b2) là bình phương hữu tỉ."""
    return [(g1, g2) for g1 in range(-12, 13) for g2 in range(-12, 13) if g1 and g2 and abs(g1) != abs(g2) and _sqrt((beta2 * g1 * g1 + g2 * g2) / (1 + beta2)) is not None]


def gen_adam_steps(rng: random.Random, level: int, seed: int):
    t = 1 if level == 1 else 2
    p = 1 if level <= 3 else 2
    for _ in range(500):
        beta1 = rng.choice([F(1, 2), F(1, 4), F(3, 4), F(4, 5), F(1, 3)])
        beta2 = rng.choice([F(1, 2), F(1, 3), F(3, 4), F(1, 4), F(4, 5), F(3, 5), F(2, 3)]) if t == 2 else rng.choice([F(1, 2), F(9, 10), F(99, 100)])
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 8), F(1, 20), F(1)])
        theta0 = [F(rng.randint(-9, 9)) for _ in range(p)]
        if t == 1:
            gcols = [[F(pick(rng, -9, 9))] for _ in range(p)]
        else:
            pairs = _adam_pairs(beta2)
            gcols = [[F(a), F(b)] for a, b in (rng.choice(pairs) for _ in range(p))]
        th, ms, vs, rows = list(theta0), [F(0)] * p, [F(0)] * p, []
        for s in range(t):
            for j in range(p):
                g = gcols[j][s]
                ms[j] = beta1 * ms[j] + (1 - beta1) * g
                vs[j] = beta2 * vs[j] + (1 - beta2) * g * g
            mh = [m / (1 - beta1 ** (s + 1)) for m in ms]
            vh = [v / (1 - beta2 ** (s + 1)) for v in vs]
            sq = [_sqrt(v) for v in vh]
            check(all(q is not None for q in sq), "v_hat không phải bình phương hữu tỉ")
            th = [a - eta * m / q for a, m, q in zip(th, mh, sq)]
            rows.append((list(mh), list(vh), list(sq), list(th)))
        if nice(th, 500, 200):
            break
    # kiểm chứng độc lập: v_hat là trung bình có trọng số của g^2, căn bình phương lại khớp
    for j in range(p):
        w = [beta2 ** (t - 1 - i) for i in range(t)]
        vh_alt = sum(wi * gcols[j][i] ** 2 for i, wi in enumerate(w)) / sum(w)
        check(vh_alt == rows[-1][1][j], "v_hat sai")
    steps = []
    for s, (mh, vh, sq, tn) in enumerate(rows, 1):
        steps.append(Step(text=f"Bước {s}: $\\hat{{m}}_{s}$, $\\hat{{v}}_{s}$ (đã hiệu chỉnh độ chệch), rồi cập nhật $\\theta_{s} = \\theta_{s - 1} - \\eta\\,\\hat{{m}}_{s}/\\sqrt{{\\hat{{v}}_{s}}}$.",
                          latex=(f"\\hat{{m}}_{s} = {vt(mh) if p > 1 else ft(mh[0])},\\ \\hat{{v}}_{s} = {vt(vh) if p > 1 else ft(vh[0])},\\ \\sqrt{{\\hat{{v}}_{s}}} = {vt(sq) if p > 1 else ft(sq[0])}"
                                 f",\\ \\theta_{s} = {vt(tn) if p > 1 else ft(tn[0])}")))
    if p == 1:
        gtxt = ", ".join(f"g_{i + 1} = {ft(g)}" for i, g in enumerate(gcols[0]))
        gblock = [gtxt]
    else:
        gblock = [f"g_{i + 1} = {vt([gcols[j][i] for j in range(p)])}" for i in range(t)]
    return ob(type_="adam-steps", lesson=L55, level=level, seed=seed,
              prompt=Prompt(text=f"Chạy Adam ${t}$ bước (đặt $\\epsilon = 0$; các phép trên vector làm theo từng thành phần) với $\\beta_1 = {ft(beta1)}$, $\\beta_2 = {ft(beta2)}$, $\\eta = {ft(eta)}$, "
                                 f"$\\theta_0 = {vt(theta0) if p > 1 else ft(theta0[0])}$. Gradient (đã cho sẵn): ${'$; $'.join(gblock)}$. Tính $\\theta_{t}$. " + (NUM_HINT if p == 1 else VEC_HINT),
                            display=["m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t,\\qquad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2", "\\hat{m}_t = \\dfrac{m_t}{1-\\beta_1^t},\\qquad \\hat{v}_t = \\dfrac{v_t}{1-\\beta_2^t},\\qquad \\theta_t = \\theta_{t-1} - \\eta\\,\\dfrac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}}"]),
              answer=num_ans(th[0]) if p == 1 else vec_ans(th),
              hints=["Khởi tạo $m_0 = v_0 = 0$. Bước đầu tiên: $\\hat{m}_1 = g_1$ và $\\hat{v}_1 = g_1^2$, nên bước cập nhật có độ lớn đúng $\\eta$ (với $\\epsilon = 0$).",
                     "Tính $m_t, v_t$ trước, rồi hiệu chỉnh bằng $1 - \\beta^t$.", "Căn bậc hai của $\\hat{v}_t$ là một số hữu tỉ (đề đã chọn như vậy)."],
              steps=steps, verified=["fraction-simulation", "weighted-average", "exact-sqrt"], tags=["adam", f"t{t}", f"{p}param"])


# -------------------------------------------------------------- adagrad-steps
@lru_cache(maxsize=None)
def _triples(n: int):
    """Bộ n số nguyên khác 0, |g| <= 12, có tổng bình phương là bình phương."""
    import itertools

    out = []
    for combo in itertools.product(range(-12, 13), repeat=n):
        prefixes = [sum(c * c for c in combo[: i + 1]) for i in range(n)]  # mọi tổng riêng đều phải là số chính phương
        if all(combo) and all(_sqrt(F(v)) is not None for v in prefixes) and len({abs(c) for c in combo}) > 1:
            out.append(combo)
    return out


def gen_adagrad_steps(rng: random.Random, level: int, seed: int):
    t, p = {1: (2, 1), 2: (3, 1), 3: (2, 2)}[level]
    for _ in range(500):
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 3), F(1)])
        theta0 = [F(rng.randint(-9, 9)) for _ in range(p)]
        pool = _triples(t) if t == 2 else _triples(t)
        gcols = [list(map(F, rng.choice(pool))) for _ in range(p)]
        th = list(theta0)
        acc = [F(0)] * p
        rows = []
        for s in range(t):
            for j in range(p):
                acc[j] += gcols[j][s] ** 2
            sq = [_sqrt(a) for a in acc]
            check(all(q is not None for q in sq), "G không phải bình phương")
            th = [a - eta * gcols[j][s] / sq[j] for j, a in enumerate(th)]
            rows.append((list(acc), list(sq), list(th)))
        if nice(th, 300, 100):
            break
    steps = []
    for s, (a, sq, tn) in enumerate(rows, 1):
        steps.append(Step(text=f"Bước {s}: cộng dồn $G_{s} = G_{s - 1} + g_{s}^2$ rồi $\\theta_{s} = \\theta_{s - 1} - \\eta\\,g_{s}/\\sqrt{{G_{s}}}$.",
                          latex=f"G_{s} = {vt(a) if p > 1 else ft(a[0])},\\ \\sqrt{{G_{s}}} = {vt(sq) if p > 1 else ft(sq[0])},\\ \\theta_{s} = {vt(tn) if p > 1 else ft(tn[0])}"))
    if p == 1:
        gblock = [", ".join(f"g_{i + 1} = {ft(g)}" for i, g in enumerate(gcols[0]))]
    else:
        gblock = [f"g_{i + 1} = {vt([gcols[j][i] for j in range(p)])}" for i in range(t)]
    # kiểm chứng độc lập: tính lại từ đầu bằng công thức đóng
    for j in range(p):
        big = sum(g * g for g in gcols[j])
        theta_alt = theta0[j]
        run = F(0)
        for g in gcols[j]:
            run += g * g
            theta_alt -= eta * g / _sqrt(run)
        check(theta_alt == th[j], "AdaGrad tính lại không khớp")
    return ob(type_="adagrad-steps", lesson=L55, level=level, seed=seed,
              prompt=Prompt(text=f"Chạy AdaGrad ${t}$ bước (đặt $\\epsilon = 0$, $G_0 = 0$; các phép trên vector làm theo từng thành phần) với $\\eta = {ft(eta)}$, $\\theta_0 = {vt(theta0) if p > 1 else ft(theta0[0])}$. "
                                 f"Gradient: ${'$; $'.join(gblock)}$. Tính $\\theta_{t}$. " + (NUM_HINT if p == 1 else VEC_HINT),
                            display=["G_t = G_{t-1} + g_t^2,\\qquad \\theta_t = \\theta_{t-1} - \\eta\\,\\dfrac{g_t}{\\sqrt{G_t}}"]),
              answer=num_ans(th[0]) if p == 1 else vec_ans(th),
              hints=["$G_t$ là tổng bình phương mọi gradient từ đầu đến bước $t$ (không có trung bình động).", "Bước đầu tiên: $\\sqrt{G_1} = |g_1|$ nên $\\theta_1 = \\theta_0 - \\eta\\,\\mathrm{sign}(g_1)$.",
                     "Các tổng bình phương trong đề là số chính phương, nên căn ra số nguyên."],
              steps=steps, verified=["fraction-simulation", "recomputed-closed-form", "exact-sqrt"], tags=["adagrad", f"t{t}", f"{p}param"])


# -------------------------------------------------------------- rmsprop-steps
RHOS = [F(3, 4), F(15, 16), F(8, 9), F(24, 25), F(0)]


@lru_cache(maxsize=None)
def _rms_pairs(rho: F):
    return [(a, b) for a in range(-12, 13) for b in range(-12, 13)
            if a and b and abs(a) != abs(b) and _sqrt((1 - rho) * a * a) is not None and _sqrt((1 - rho) * (rho * a * a + b * b)) is not None]


def gen_rmsprop_steps(rng: random.Random, level: int, seed: int):
    t, p = {1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2), 5: (2, 2)}[level]
    for _ in range(1000):
        rho = rng.choice(RHOS if level <= 2 else RHOS[:4])
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 8), F(1, 20), F(1)])
        theta0 = [F(rng.randint(-9, 9)) for _ in range(p)]
        if t == 1:
            gcols = [[F(pick(rng, -9, 9))] for _ in range(p)]
        else:
            pairs = _rms_pairs(rho)
            if not pairs:
                continue
            gcols = [[F(a), F(b)] for a, b in (rng.choice(pairs) for _ in range(p))]
        th, acc, rows = list(theta0), [F(0)] * p, []
        for s in range(t):
            for j in range(p):
                acc[j] = rho * acc[j] + (1 - rho) * gcols[j][s] ** 2
            sq = [_sqrt(a) for a in acc]
            if any(q is None for q in sq):
                break
            th = [a - eta * gcols[j][s] / sq[j] for j, a in enumerate(th)]
            rows.append((list(acc), list(sq), list(th)))
        else:
            if nice(th, 500, 200):
                break
    steps = []
    for s, (a, sq, tn) in enumerate(rows, 1):
        steps.append(Step(text=f"Bước {s}: $s_{s} = \\rho\\,s_{s - 1} + (1-\\rho)g_{s}^2$ rồi $\\theta_{s} = \\theta_{s - 1} - \\eta\\,g_{s}/\\sqrt{{s_{s}}}$.",
                          latex=f"s_{s} = {vt(a) if p > 1 else ft(a[0])},\\ \\sqrt{{s_{s}}} = {vt(sq) if p > 1 else ft(sq[0])},\\ \\theta_{s} = {vt(tn) if p > 1 else ft(tn[0])}"))
    if p == 1:
        gblock = [", ".join(f"g_{i + 1} = {ft(g)}" for i, g in enumerate(gcols[0]))]
    else:
        gblock = [f"g_{i + 1} = {vt([gcols[j][i] for j in range(p)])}" for i in range(t)]
    for j in range(p):  # kiểm chứng độc lập: công thức đóng của EMA
        sj = (1 - rho) * sum(rho ** (t - 1 - i) * g * g for i, g in enumerate(gcols[j]))
        check(sj == rows[-1][0][j], "s_t sai")
    return ob(type_="rmsprop-steps", lesson=L55, level=level, seed=seed,
              prompt=Prompt(text=f"Chạy RMSProp ${t}$ bước (đặt $\\epsilon = 0$, $s_0 = 0$; vector làm theo từng thành phần) với $\\rho = {ft(rho)}$, $\\eta = {ft(eta)}$, $\\theta_0 = {vt(theta0) if p > 1 else ft(theta0[0])}$. "
                                 f"Gradient: ${'$; $'.join(gblock)}$. Tính $\\theta_{t}$. " + (NUM_HINT if p == 1 else VEC_HINT),
                            display=["s_t = \\rho\\,s_{t-1} + (1-\\rho)\\,g_t^2,\\qquad \\theta_t = \\theta_{t-1} - \\eta\\,\\dfrac{g_t}{\\sqrt{s_t}}"]),
              answer=num_ans(th[0]) if p == 1 else vec_ans(th),
              hints=["$s_t$ là trung bình động mũ của $g^2$ (khác AdaGrad: các gradient cũ bị quên dần), và **không** có hiệu chỉnh độ chệch trong RMSProp.",
                     "Bước đầu: $s_1 = (1-\\rho)g_1^2$ nên $\\sqrt{s_1} = \\sqrt{1-\\rho}\\,|g_1|$.", "Các giá trị trong đề được chọn để căn bậc hai là số hữu tỉ."],
              steps=steps, verified=["fraction-simulation", "closed-form-ema", "exact-sqrt"], tags=["rmsprop", f"t{t}", f"{p}param"])


SPECS = [
    TopicSpec("momentum-steps", L54, (1, 2, 3, 4, 5), gen_momentum_steps),
    TopicSpec("adam-bias", L55, (1, 2, 3, 4, 5), gen_adam_bias),
    TopicSpec("adam-steps", L55, (1, 2, 3, 4, 5), gen_adam_steps),
    TopicSpec("adagrad-steps", L55, (1, 2, 3), gen_adagrad_steps),
    TopicSpec("rmsprop-steps", L55, (1, 2, 3, 4, 5), gen_rmsprop_steps),
]
