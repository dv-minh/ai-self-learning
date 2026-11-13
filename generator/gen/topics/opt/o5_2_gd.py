"""Bài 5.2 Gradient descent: gd-steps, gd-linreg, gd-converge, gd-max-lr."""
from __future__ import annotations

from sympy import Matrix, Rational, symbols

from ._o import (F, NUM_HINT, VEC_HINT, Answer, L, Prompt, Step, TopicSpec, check, choice_exercise, fp, fs, ft, lf, nice, num_ans, ob, pick, random, vec_ans, vt)

L52 = "5.2"
ETAS = [F(1, 2), F(1, 3), F(1, 4), F(1, 5), F(1, 6), F(1, 8), F(1, 10), F(1, 20), F(3, 10), F(2, 5)]
X, Y = symbols("x y")


def _sym_grad(fexpr, vs, point):
    """Gradient chính xác bằng SymPy (đạo hàm biểu tượng rồi thế số hữu tỉ) — kiểm chứng độc lập với công thức gradient viết tay."""
    return [F(int(Rational(fexpr.diff(v).subs(dict(zip(vs, [Rational(p.numerator, p.denominator) for p in point])))).p),
              int(Rational(fexpr.diff(v).subs(dict(zip(vs, [Rational(p.numerator, p.denominator) for p in point])))).q)) for v in vs]


# ------------------------------------------------------------------- gd-steps
def gen_gd_steps(rng: random.Random, level: int, seed: int):
    for _ in range(500):
        eta = rng.choice(ETAS)
        if level in (1, 2):
            a, b = pick(rng, 1, 6), pick(rng, -9, 9, False)
            f_tex, fexpr, vs = lf([a, b], ["x^{2}", "x"]), a * X**2 + b * X, [X]
            grad = lambda p: [2 * a * p[0] + b]
            x0, k = [F(pick(rng, -8, 8))], (1 if level == 1 else rng.choice([2, 3]))
            gtex = lambda p: lf([2 * a, b], ["x", "1"])
            head = "f(x)"
        elif level == 3:
            a, c, d, e = pick(rng, 1, 5), pick(rng, 1, 5), rng.randint(-8, 8), rng.randint(-8, 8)
            f_tex, fexpr, vs = lf([a, c, d, e], ["x^{2}", "y^{2}", "x", "y"]), a * X**2 + c * Y**2 + d * X + e * Y, [X, Y]
            grad = lambda p: [2 * a * p[0] + d, 2 * c * p[1] + e]
            x0, k, head = [F(pick(rng, -6, 6)), F(pick(rng, -6, 6))], 1, "f(x, y)"
        elif level == 4:
            a, c, b, d, e = pick(rng, 1, 5), pick(rng, 1, 5), rng.randint(-3, 3), rng.randint(-8, 8), rng.randint(-8, 8)
            f_tex, fexpr, vs = lf([a, b, c, d, e], ["x^{2}", "xy", "y^{2}", "x", "y"]), a * X**2 + b * X * Y + c * Y**2 + d * X + e * Y, [X, Y]
            grad = lambda p: [2 * a * p[0] + b * p[1] + d, b * p[0] + 2 * c * p[1] + e]
            x0, k, head = [F(pick(rng, -6, 6)), F(pick(rng, -6, 6))], 2, "f(x, y)"
        else:  # bậc bốn, không còn là hàm bậc hai
            a, b, c = pick(rng, 1, 3), pick(rng, 1, 4), rng.randint(-3, 3)
            if rng.random() < 0.5:
                f_tex, fexpr, vs = lf([a, b, c], ["x^{4}", "x^{2}", "x"]), a * X**4 + b * X**2 + c * X, [X]
                grad = lambda p: [4 * a * p[0] ** 3 + 2 * b * p[0] + c]
                x0, k, head = [F(pick(rng, -3, 3))], 2, "f(x)"
            else:
                f_tex, fexpr, vs = lf([a, b, c], ["x^{4}", "y^{2}", "xy"]), a * X**4 + b * Y**2 + c * X * Y, [X, Y]
                grad = lambda p: [4 * a * p[0] ** 3 + c * p[1], 2 * b * p[1] + c * p[0]]
                x0, k, head = [F(pick(rng, -3, 3)), F(pick(rng, -3, 3))], 2, "f(x, y)"
        path, grads = [x0], []
        for _ in range(k):
            g = grad(path[-1])
            check(g == _sym_grad(fexpr, vs, path[-1]), "gradient viết tay ≠ SymPy")
            grads.append(g)
            path.append([xi - eta * gi for xi, gi in zip(path[-1], g)])
        if nice([v for p in path for v in p], 400, 200):
            break
    else:
        raise RuntimeError("không dựng được đề gd-steps")
    one = len(vs) == 1
    steps = []
    for t in range(k):
        gv, xv = grads[t], path[t]
        gt = vt(gv) if not one else ft(gv[0])
        xn = vt(path[t + 1]) if not one else ft(path[t + 1][0])
        xt = vt(xv) if not one else ft(xv[0])
        steps.append(Step(text=f"Bước {t + 1}: tính gradient tại $x_{t}$ rồi cập nhật.",
                          latex=f"\\nabla f(x_{t}) = {gt},\\qquad x_{t + 1} = {xt} - {fp(eta)}\\cdot {gt if one else '(' + gt + ')'} = {xn}"))
    ans = num_ans(path[-1][0]) if one else vec_ans(path[-1])
    unit = "một số" if one else "một vector"
    return ob(type_="gd-steps", lesson=L52, level=level, seed=seed,
              prompt=Prompt(text=f"Chạy gradient descent $x_{{t+1}} = x_t - \\eta\\,\\nabla f(x_t)$ cho hàm dưới đây với $\\eta = {ft(eta)}$, bắt đầu từ $x_0 = {ft(x0[0]) if one else vt(x0)}$. "
                                 f"Tính $x_{k}$ ({unit}). " + (NUM_HINT if one else VEC_HINT), display=[f"{head} = {f_tex}"]),
              answer=ans,
              hints=["Mỗi bước: tính $\\nabla f(x_t)$ tại điểm hiện tại, nhân với $\\eta$, rồi **trừ** khỏi $x_t$.",
                     "Dùng kết quả của bước trước làm điểm bắt đầu cho bước sau." if k > 1 else "$\\nabla f$ của $ax^2$ là $2ax$; của $bx$ là $b$.",
                     "Với hàm hai biến, tính từng thành phần của gradient (đạo hàm riêng) rồi cập nhật từng tọa độ."],
              steps=steps, verified=["sympy-gradient", "fraction-simulation"], tags=[f"{len(vs)}var", f"{k}step"])


# ------------------------------------------------------------------ gd-linreg
def gen_gd_linreg(rng: random.Random, level: int, seed: int):
    n = {1: 2, 2: rng.choice([3, 4]), 3: 3, 4: 4, 5: 4}[level]
    with_b = level >= 3
    k = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}[level]
    w_, b_, e_ = symbols("w b e")
    for _ in range(500):
        xs = [F(pick(rng, -4, 4, False)) for _ in range(n)]
        if len(set(xs)) < 2:
            continue
        ys = [F(rng.randint(-8, 8)) for _ in range(n)]
        w0, b0 = F(rng.randint(-3, 3)), (F(rng.randint(-3, 3)) if with_b else F(0))
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 8), F(1, 20), F(1, 6)])
        # mất mát L = 1/(2n) * sum (w x + b - y)^2
        loss = sum((w_ * Rational(int(x)) + (b_ if with_b else 0) - Rational(int(y))) ** 2 for x, y in zip(xs, ys)) / (2 * n)
        path = [(w0, b0)]
        info = []
        for _t in range(k):
            w, b = path[-1]
            res = [w * x + b - y for x, y in zip(xs, ys)]
            gw = sum(r * x for r, x in zip(res, xs)) / n
            gb = sum(res) / n if with_b else F(0)
            sw = loss.diff(w_).subs({w_: Rational(w.numerator, w.denominator), b_: Rational(b.numerator, b.denominator)})
            check(Rational(gw.numerator, gw.denominator) == sw, "gradient theo w sai")
            if with_b:
                sb = loss.diff(b_).subs({w_: Rational(w.numerator, w.denominator), b_: Rational(b.numerator, b.denominator)})
                check(Rational(gb.numerator, gb.denominator) == sb, "gradient theo b sai")
            info.append((w, b, res, gw, gb))
            path.append((w - eta * gw, b - eta * gb))
        if nice([v for p in path for v in p], 600, 300):
            break
    else:
        raise RuntimeError("không dựng được đề gd-linreg")
    steps = []
    for t, (w, b, res, gw, gb) in enumerate(info):
        steps.append(Step(text=f"Bước {t + 1}: sai số dự đoán $e_i = \\hat{{y}}_i - y_i$ tại $(w, b) = ({ft(w)}, {ft(b)})$." if with_b else f"Bước {t + 1}: sai số dự đoán $e_i = \\hat{{y}}_i - y_i$ tại $w = {ft(w)}$.",
                          latex="e = (" + ", ".join(ft(r) for r in res) + ")"))
        steps.append(Step(text="Gradient của $L$ (trung bình, có hệ số $\\tfrac{1}{2n}$ nên đạo hàm còn $\\tfrac{1}{n}$):",
                          latex=f"\\dfrac{{\\partial L}}{{\\partial w}} = \\dfrac{{1}}{{{n}}}\\sum e_i x_i = {ft(gw)}" + (f",\\qquad \\dfrac{{\\partial L}}{{\\partial b}} = \\dfrac{{1}}{{{n}}}\\sum e_i = {ft(gb)}" if with_b else "")))
        wn, bn = path[t + 1]
        steps.append(Step(text="Cập nhật:", latex=(f"w \\leftarrow {ft(w)} - {fp(eta)}\\cdot {fp(gw)} = {ft(wn)}" + (f",\\qquad b \\leftarrow {ft(b)} - {fp(eta)}\\cdot {fp(gb)} = {ft(bn)}" if with_b else ""))))
    tbl = "\\begin{array}{c|" + "c" * n + "} x & " + " & ".join(ft(x) for x in xs) + " \\\\ y & " + " & ".join(ft(y) for y in ys) + " \\end{array}"
    model = "\\hat{y} = wx + b" if with_b else "\\hat{y} = wx"
    start = f"$(w, b) = ({ft(w0)}, {ft(b0)})$" if with_b else f"$w = {ft(w0)}$"
    ans = vec_ans(path[-1]) if with_b else num_ans(path[-1][0])
    return ob(type_="gd-linreg", lesson=L52, level=level, seed=seed,
              prompt=Prompt(text=f"Hồi quy tuyến tính ${model}$ trên ${n}$ điểm dữ liệu (bảng dưới) với mất mát $L = \\dfrac{{1}}{{2n}}\\sum_{{i}}(\\hat{{y}}_i - y_i)^2$. "
                                 f"Chạy gradient descent với $\\eta = {ft(eta)}$ từ {start}, ${k}$ bước. Tính " + ("$(w, b)$ sau bước cuối. " + VEC_HINT if with_b else "$w$ sau bước cuối. " + NUM_HINT),
                            display=[tbl]),
              answer=ans,
              hints=["$\\dfrac{\\partial L}{\\partial w} = \\dfrac{1}{n}\\sum_i (\\hat{y}_i - y_i)\\,x_i$ và $\\dfrac{\\partial L}{\\partial b} = \\dfrac{1}{n}\\sum_i (\\hat{y}_i - y_i)$.",
                     "Tính trước vector sai số $e_i = wx_i + b - y_i$ bằng $w$, $b$ hiện tại, rồi mới tính gradient.", "Cập nhật **đồng thời** $w$ và $b$ bằng giá trị cũ."],
              steps=steps, verified=["sympy-gradient", "fraction-simulation"], tags=["ml", "with-b" if with_b else "no-b", f"n{n}"])


# ---------------------------------------------------------------- gd-converge
OPTS = {
    "mono": "Hội tụ về $0$, các $x_t$ giữ nguyên dấu",
    "osc": "Hội tụ về $0$, nhưng $x_t$ đổi dấu sau mỗi bước",
    "loop": "Không hội tụ: $x_t$ dao động mãi giữa hai giá trị $x_0$ và $-x_0$",
    "div": "Phân kỳ: $|x_t|$ tăng vô hạn",
}


def gen_gd_converge(rng: random.Random, level: int, seed: int):
    for _ in range(500):
        a = pick(rng, 1, 9)
        eta = rng.choice([F(1, 20), F(1, 10), F(1, 5), F(1, 4), F(1, 3), F(1, 2), F(3, 5), F(2, 3), F(3, 4), F(1, 1), F(3, 2), F(2), F(5, 2), F(1, 8), F(2, 5)])
        x0 = F(pick(rng, -9, 9))
        r = 1 - eta * a if level == 1 else 1 - 2 * a * eta
        if r == 0 or r == 1:
            continue
        break
    kind = "mono" if 0 < r < 1 else "osc" if -1 < r < 0 else "loop" if r == -1 else "div"
    # kiểm chứng độc lập: mô phỏng 60 bước bằng Fraction
    x = x0
    signs = []
    for _ in range(60):
        g = a * x if level == 1 else 2 * a * x
        x = x - eta * g
        signs.append((x > 0) - (x < 0))
    if kind == "mono":
        check(abs(x) < abs(x0) / 2 and len(set(signs)) == 1, "mô phỏng không hội tụ đơn điệu")
    elif kind == "osc":
        check(abs(x) < abs(x0) / 2 and all(signs[i] != signs[i + 1] for i in range(59)), "mô phỏng không dao động hội tụ")
    elif kind == "loop":
        check(abs(x) == abs(x0), "mô phỏng không dao động đều")
    else:
        check(abs(x) > abs(x0) * 100, "mô phỏng không phân kỳ")
    fname = f"f(x) = \\dfrac{{{a}}}{{2}}x^{{2}}" if level == 1 and a > 1 else ("f(x) = \\frac{1}{2}x^{2}" if level == 1 else f"f(x) = {lf([a], ['x^{2}'])}")
    deriv = f"f'(x) = {lf([a], ['x'])}" if level == 1 else f"f'(x) = {lf([2 * a], ['x'])}"
    choices, ans = choice_exercise(rng, OPTS[kind], [v for k, v in OPTS.items() if k != kind])
    return ob(type_="gd-converge", lesson=L52, level=level, seed=seed,
              prompt=Prompt(text=f"Chạy gradient descent trên hàm dưới đây với $\\eta = {ft(eta)}$ và $x_0 = {x0.numerator}$. Dãy $x_t$ hành xử thế nào?", display=[fname]),
              answer=ans, choices=choices,
              hints=["Với hàm bậc hai, mỗi bước nhân $x_t$ với một hệ số cố định $r$: $x_{t+1} = r\\,x_t$.", "$|r| < 1$ thì hội tụ ($r > 0$: giữ dấu; $r < 0$: đổi dấu mỗi bước); $|r| = 1$ thì dao động đều; $|r| > 1$ thì phân kỳ."],
              steps=[Step(text="Cập nhật có dạng $x_{t+1} = x_t - \\eta f'(x_t) = r\\,x_t$:", latex=f"{deriv},\\qquad r = 1 - {fp(eta)}\\cdot {a if level == 1 else 2 * a} = {ft(r)}"),
                     Step(text=f"Với $r = {ft(r)}$: {OPTS[kind]}.")],
              verified=["fraction-simulation-60-steps"], tags=[kind])


# ----------------------------------------------------------------- gd-max-lr
_ROT = [(F(3, 5), F(4, 5)), (F(5, 13), F(12, 13)), (F(8, 17), F(15, 17)), (F(7, 25), F(24, 25)), (F(20, 29), F(21, 29))]


def gen_gd_max_lr(rng: random.Random, level: int, seed: int):
    if level == 3:  # đường chéo
        lam = sorted([pick(rng, 1, 9), pick(rng, 1, 9)])
        qm = Matrix([[lam[0], 0], [0, lam[1]]]) if rng.random() < 0.5 else Matrix([[lam[1], 0], [0, lam[0]]])
        form = "diag"
    else:
        lam = sorted([pick(rng, 1, 9), pick(rng, 1, 9)])
        c, s = rng.choice(_ROT)
        s *= rng.choice([1, -1])
        p = Matrix([[Rational(c.numerator, c.denominator), -Rational(s.numerator, s.denominator)], [Rational(s.numerator, s.denominator), Rational(c.numerator, c.denominator)]])
        rng.shuffle(lam)
        qm = p * Matrix([[lam[0], 0], [0, lam[1]]]) * p.T
        form = "rot"
    ev = sorted(qm.eigenvals(multiple=True))
    check(sorted(lam) == ev, "trị riêng ≠ giá trị thiết kế")
    lmax, lmin = F(int(max(lam))), F(int(min(lam)))
    entries = [[F(int(qm[i, j].p), int(qm[i, j].q)) for j in range(2)] for i in range(2)]
    fx = f"f(\\mathbf{{x}}) = \\dfrac{{1}}{{2}}\\mathbf{{x}}\\T \\mathbf{{Q}}\\mathbf{{x}},\\qquad \\mathbf{{Q}} = {L.mat(qm)}"
    if level in (3, 4):
        ans_v = F(2) / lmax
        text = "Chạy gradient descent $\\mathbf{x}_{t+1} = \\mathbf{x}_t - \\eta\\,\\nabla f(\\mathbf{x}_t)$ với $f$ dưới đây (Q xác định dương). Gradient descent hội tụ về $\\mathbf{0}$ với mọi điểm bắt đầu khi và chỉ khi $0 < \\eta < \\eta_{\\max}$. Tìm $\\eta_{\\max}$. "
        steps = [Step(text="$\\nabla f = \\mathbf{Q}\\mathbf{x}$ nên $\\mathbf{x}_{t+1} = (\\mathbf{I} - \\eta\\mathbf{Q})\\mathbf{x}_t$, hội tụ khi mọi $|1 - \\eta\\lambda_i| < 1$ với $\\lambda_i$ là trị riêng của $\\mathbf{Q}$."),
                 Step(text=f"Trị riêng của $\\mathbf{{Q}}$: ${ft(lmin)}$ và ${ft(lmax)}$. Điều kiện $\\eta\\lambda < 2$ chặt nhất với trị riêng lớn nhất:", latex=f"\\eta_{{\\max}} = \\dfrac{{2}}{{\\lambda_{{\\max}}}} = {ft(ans_v)}")]
    else:
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 3), F(1, 6), F(1, 8), F(1, 12)])
        rho = max(abs(1 - eta * lmax), abs(1 - eta * lmin))
        ans_v = rho
        text = (f"Với $\\eta = {ft(eta)}$, gradient descent trên $f$ dưới đây thỏa $\\|\\mathbf{{x}}_{{t+1}}\\| \\le \\rho \\|\\mathbf{{x}}_t\\|$ với $\\rho = \\max_i |1 - \\eta\\lambda_i|$ ($\\lambda_i$ là trị riêng của $\\mathbf{{Q}}$). Tính $\\rho$. ")
        steps = [Step(text=f"Trị riêng của $\\mathbf{{Q}}$: ${ft(lmin)}$ và ${ft(lmax)}$."),
                 Step(text="Tính $|1 - \\eta\\lambda|$ cho từng trị riêng rồi lấy lớn nhất:", latex=f"\\rho = \\max\\left(|1 - {fp(eta)}\\cdot {ft(lmin)}|,\\ |1 - {fp(eta)}\\cdot {ft(lmax)}|\\right) = {ft(rho)}")]
    # kiểm chứng độc lập: ma trận lặp I - eta Q có bán kính phổ đúng như tính
    import numpy as np

    qn = np.array([[float(v) for v in r] for r in entries])
    if level in (3, 4):
        e = float(ans_v)
        for scale, ok in ((0.999, True), (1.001, False)):
            m = np.eye(2) - e * scale * qn
            check((max(abs(np.linalg.eigvals(m))) < 1) == ok, "ngưỡng η_max sai")
    else:
        m = np.eye(2) - float(eta) * qn
        check(abs(max(abs(np.linalg.eigvals(m))) - float(rho)) < 1e-9, "bán kính phổ sai")
    return ob(type_="gd-max-lr", lesson=L52, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT, display=[fx]), answer=num_ans(ans_v),
              hints=["Gradient của $\\frac{1}{2}\\mathbf{x}\\T\\mathbf{Q}\\mathbf{x}$ là $\\mathbf{Q}\\mathbf{x}$ (khi $\\mathbf{Q}$ đối xứng).", "Mỗi trị riêng $\\lambda$ cho một hướng mà $x$ bị nhân với $1 - \\eta\\lambda$ ở mỗi bước.",
                     "Với ma trận chéo, trị riêng chính là các phần tử đường chéo; với ma trận $2\\times 2$, giải $\\det(\\mathbf{Q} - \\lambda\\mathbf{I}) = 0$."],
              steps=steps, verified=["sympy-eigenvalues", "numpy-spectral-radius"], tags=[form])


SPECS = [
    TopicSpec("gd-steps", L52, (1, 2, 3, 4, 5), gen_gd_steps),
    TopicSpec("gd-linreg", L52, (1, 2, 3, 4, 5), gen_gd_linreg),
    TopicSpec("gd-converge", L52, (1, 2), gen_gd_converge),
    TopicSpec("gd-max-lr", L52, (3, 4, 5), gen_gd_max_lr),
]
