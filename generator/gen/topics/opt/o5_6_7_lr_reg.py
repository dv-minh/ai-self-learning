"""Bài 5.6 Lịch learning rate: lr-decay, lr-cosine. Bài 5.7 Regularization: wd-step, adamw-step, ridge-solve, l1-prox."""
from __future__ import annotations

import math

import numpy as np
from sympy import Matrix, Rational, symbols

from ._o import F, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, fp, ft, lf, nice, num_ans, ob, pick, random, vec_ans, vt

L56, L57 = "5.6", "5.7"
ETA0S = [F(1), F(1, 2), F(1, 5), F(1, 10), F(3, 10), F(1, 100), F(1, 4), F(2, 5)]


def _q(v: F) -> Rational:
    return Rational(v.numerator, v.denominator)


# ------------------------------------------------------------------- lr-decay
def gen_lr_decay(rng: random.Random, level: int, seed: int):
    eta0 = rng.choice(ETA0S)
    if level == 1:
        s = rng.choice([2, 3, 4, 5, 10])
        gamma = rng.choice([F(1, 2), F(1, 10), F(1, 3), F(1, 5), F(3, 4), F(9, 10)])
        t = rng.randint(s, 5 * s + s - 1)
        val = eta0 * gamma ** (t // s)
        loop = eta0
        for i in range(1, t + 1):  # mô phỏng: nhân $\gamma$ sau mỗi s bước
            if i % s == 0:
                loop *= gamma
        check(loop == val, "step decay mô phỏng ≠ công thức")
        text = f"Step decay: $\\eta_t = \\eta_0\\,\\gamma^{{\\lfloor t/s \\rfloor}}$ với $\\eta_0 = {ft(eta0)}$, $\\gamma = {ft(gamma)}$, $s = {s}$. Tính $\\eta_{{{t}}}$. "
        steps = [Step(text=f"$\\lfloor {t}/{s} \\rfloor = {t // s}$ nên:", latex=f"\\eta_{{{t}}} = {ft(eta0)}\\cdot \\left({ft(gamma)}\\right)^{{{t // s}}} = {ft(val)}")]
        hints = ["$\\lfloor t/s \\rfloor$ là số lần đã giảm (phần nguyên của phép chia).", "Mỗi $s$ bước, learning rate nhân với $\\gamma$."]
        tag = "step"
    elif level == 2:
        kind = rng.choice(["exp", "inv"])
        t = rng.randint(2, 6)
        if kind == "exp":
            gamma = rng.choice([F(1, 2), F(9, 10), F(4, 5), F(1, 3), F(3, 4), F(1, 10)])
            val = eta0 * gamma**t
            loop = eta0
            for _ in range(t):
                loop *= gamma
            check(loop == val, "exp decay mô phỏng ≠ công thức")
            text = f"Exponential decay: $\\eta_t = \\eta_0\\,\\gamma^{{t}}$ với $\\eta_0 = {ft(eta0)}$, $\\gamma = {ft(gamma)}$. Tính $\\eta_{{{t}}}$. "
            steps = [Step(text="Thay số:", latex=f"\\eta_{{{t}}} = {ft(eta0)}\\cdot \\left({ft(gamma)}\\right)^{{{t}}} = {ft(val)}")]
            hints = ["Mỗi bước nhân learning rate với $\\gamma$, nên sau $t$ bước là $\\eta_0\\gamma^t$."]
        else:
            k = rng.choice([F(1, 2), F(1), F(2), F(1, 4), F(1, 10), F(3)])
            val = eta0 / (1 + k * t)
            text = f"Inverse-time decay: $\\eta_t = \\dfrac{{\\eta_0}}{{1 + k\\,t}}$ với $\\eta_0 = {ft(eta0)}$, $k = {ft(k)}$. Tính $\\eta_{{{t}}}$. "
            steps = [Step(text="Thay số:", latex=f"\\eta_{{{t}}} = \\dfrac{{{ft(eta0)}}}{{1 + {fp(k)}\\cdot {t}}} = {ft(val)}")]
            hints = ["Mẫu số $1 + kt$ tăng dần nên learning rate giảm dần, nhưng chậm hơn exponential."]
        tag = kind
    elif level == 3:  # warmup tuyến tính rồi giảm tuyến tính về 0
        w = rng.choice([2, 4, 5, 10])
        total = w * rng.choice([2, 3, 4])
        t = rng.randint(1, total)
        val = eta0 * F(t, w) if t <= w else eta0 * F(total - t, total - w)
        # kiểm chứng độc lập: nội suy tuyến tính qua hai điểm (0,0), (w, eta0), (total, 0)
        xs, ys = [0, w, total], [F(0), eta0, F(0)]
        seg = 0 if t <= w else 1
        alt = ys[seg] + (ys[seg + 1] - ys[seg]) * F(t - xs[seg], xs[seg + 1] - xs[seg])
        check(alt == val, "nội suy tuyến tính ≠ công thức")
        text = (f"Lịch tăng–giảm tuyến tính: $\\eta_t = \\eta_{{\\max}}\\dfrac{{t}}{{W}}$ khi $t \\le W$ (warmup), và $\\eta_t = \\eta_{{\\max}}\\dfrac{{T - t}}{{T - W}}$ khi $W < t \\le T$. "
                f"Với $\\eta_{{\\max}} = {ft(eta0)}$, $W = {w}$, $T = {total}$, tính $\\eta_{{{t}}}$. ")
        which = "giai đoạn warmup" if t <= w else "giai đoạn giảm"
        steps = [Step(text=f"$t = {t}$ thuộc {which}:",
                      latex=(f"\\eta_{{{t}}} = {ft(eta0)}\\cdot\\dfrac{{{t}}}{{{w}}} = {ft(val)}" if t <= w else f"\\eta_{{{t}}} = {ft(eta0)}\\cdot\\dfrac{{{total} - {t}}}{{{total} - {w}}} = {ft(val)}"))]
        hints = ["Xác định $t$ thuộc đoạn nào ($t \\le W$ hay $t > W$) trước khi chọn công thức.", "Hai đoạn nối liền tại $t = W$ với $\\eta = \\eta_{\\max}$."]
        tag = "warmup-linear"
    else:  # Noam: lr = d^{-1/2} min(t^{-1/2}, t W^{-3/2}); chọn d, W, t là số chính phương để căn hữu tỉ
        d = rng.choice([4, 16, 25, 64, 100, 144, 256])
        w = rng.choice([4, 9, 16, 25])
        t = rng.choice([1, 4, 9, 16, 25, 36, 49, 64, 100]) if rng.random() < 0.6 else rng.randint(1, 40)
        tsq = math.isqrt(t)
        if tsq * tsq != t:
            t = tsq * tsq or 1
        sd, sw, st = math.isqrt(d), math.isqrt(w), math.isqrt(t)
        a = F(1, st)
        b = F(t, sw**3)
        val = F(1, sd) * min(a, b)
        # kiểm chứng độc lập bằng số thực
        flt = d ** -0.5 * min(t ** -0.5, t * w ** -1.5)
        check(abs(flt - float(val)) < 1e-12, "Noam: số thực ≠ hữu tỉ")
        text = (f"Lịch của Transformer gốc: $\\eta_t = d^{{-1/2}}\\min\\left(t^{{-1/2}},\\ t\\,W^{{-3/2}}\\right)$ với $d = {d}$, $W = {w}$ (số bước warmup). Tính $\\eta_{{{t}}}$. ")
        steps = [Step(text=f"$t^{{-1/2}} = {ft(a)}$ và $t\\,W^{{-3/2}} = {ft(b)}$; lấy giá trị nhỏ hơn rồi nhân $d^{{-1/2}} = {ft(F(1, sd))}$:",
                      latex=f"\\eta_{{{t}}} = {ft(F(1, sd))}\\cdot \\min\\left({ft(a)},\\ {ft(b)}\\right) = {ft(val)}")]
        hints = ["$d$, $W$, $t$ đều là số chính phương nên các lũy thừa $-1/2$ và $-3/2$ cho số hữu tỉ.", "Khi $t < W$ thì $t\\,W^{-3/2}$ nhỏ hơn (warmup); khi $t > W$ thì $t^{-1/2}$ nhỏ hơn (giảm theo $1/\\sqrt{t}$)."]
        tag = "noam"
    return ob(type_="lr-decay", lesson=L56, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(val), hints=hints, steps=steps,
              verified=["fraction-simulation" if level <= 2 else "independent-recompute"], tags=[tag])


# ------------------------------------------------------------------ lr-cosine
COS_OF = {F(0): F(1), F(1, 3): F(1, 2), F(1, 2): F(0), F(2, 3): F(-1, 2), F(1): F(-1)}


def _cos_lr(eta_max, eta_min, ratio):
    return eta_min + (eta_max - eta_min) * (1 + COS_OF[ratio]) / 2


def gen_lr_cosine(rng: random.Random, level: int, seed: int):
    eta_max = rng.choice(ETA0S)
    eta_min = F(0) if level <= 2 else rng.choice([F(0), F(1, 100), F(1, 1000), F(1, 20), F(1, 50)])
    if eta_min >= eta_max:
        eta_min = eta_max / 10
    T = rng.choice([6, 12, 18, 24, 30, 36, 60, 90, 120, 300])
    ratio = rng.choice([F(1, 2)] if level == 1 else [F(1, 3), F(1, 2), F(2, 3)] if level in (2, 4) else [r for r in COS_OF if r != 0] if level == 3 else list(COS_OF))
    if level <= 3:
        t = int(T * ratio)
        val = _cos_lr(eta_max, eta_min, ratio)
        flt = float(eta_min) + 0.5 * float(eta_max - eta_min) * (1 + math.cos(math.pi * t / T))
        check(abs(flt - float(val)) < 1e-12, "cosine: số thực ≠ hữu tỉ")
        text = (f"Cosine annealing: $\\eta_t = \\eta_{{\\min}} + \\dfrac{{1}}{{2}}(\\eta_{{\\max}} - \\eta_{{\\min}})\\left(1 + \\cos\\dfrac{{\\pi t}}{{T}}\\right)$ với $\\eta_{{\\max}} = {ft(eta_max)}$, $\\eta_{{\\min}} = {ft(eta_min)}$, $T = {T}$. "
                f"Tính $\\eta_{{{t}}}$. ")
        steps = [Step(text=f"$\\dfrac{{\\pi t}}{{T}} = {ft(ratio)}\\pi$ nên $\\cos = {ft(COS_OF[ratio])}$ (giá trị đặc biệt của cosine):",
                      latex=f"\\eta_{{{t}}} = {ft(eta_min)} + \\dfrac{{1}}{{2}}\\left({ft(eta_max)} - {ft(eta_min)}\\right)\\left(1 + {fp(COS_OF[ratio])}\\right) = {ft(val)}")]
        hints = ["$\\cos 0 = 1$, $\\cos\\frac{\\pi}{3} = \\frac{1}{2}$, $\\cos\\frac{\\pi}{2} = 0$, $\\cos\\frac{2\\pi}{3} = -\\frac{1}{2}$, $\\cos\\pi = -1$.", "Tính $\\pi t/T$ trước: nó là bội hữu tỉ của $\\pi$."]
        tag = "cosine"
    elif level == 4:  # khởi động lại (SGDR): t lấy theo chu kỳ
        cycles = rng.choice([2, 3])
        t_cur = int(T * ratio)
        t = t_cur + T * rng.randint(1, cycles)
        val = _cos_lr(eta_max, eta_min, ratio)
        text = (f"Cosine với khởi động lại (warm restart) mỗi $T = {T}$ bước: $\\eta_t = \\eta_{{\\min}} + \\dfrac{{1}}{{2}}(\\eta_{{\\max}} - \\eta_{{\\min}})\\left(1 + \\cos\\dfrac{{\\pi\\,(t \\bmod T)}}{{T}}\\right)$ "
                f"với $\\eta_{{\\max}} = {ft(eta_max)}$, $\\eta_{{\\min}} = {ft(eta_min)}$. Tính $\\eta_{{{t}}}$. ")
        flt = float(eta_min) + 0.5 * float(eta_max - eta_min) * (1 + math.cos(math.pi * (t % T) / T))
        check(t % T == t_cur and abs(flt - float(val)) < 1e-12, "restart: kiểm chứng sai")
        steps = [Step(text=f"${t} \\bmod {T} = {t_cur}$ nên $\\dfrac{{\\pi t_{{\\text{{cur}}}}}}{{T}} = {ft(ratio)}\\pi$ và $\\cos = {ft(COS_OF[ratio])}$:",
                      latex=f"\\eta_{{{t}}} = {ft(eta_min)} + \\dfrac{{1}}{{2}}\\left({ft(eta_max)} - {ft(eta_min)}\\right)\\left(1 + {fp(COS_OF[ratio])}\\right) = {ft(val)}")]
        hints = ["Chỉ dùng phần dư $t \\bmod T$: cứ sau $T$ bước, learning rate quay lại $\\eta_{\\max}$.", "Sau đó áp dụng công thức cosine thường với $t$ thay bằng phần dư."]
        tag = "restart"
    else:  # warmup tuyến tính W bước rồi cosine trên T - W bước còn lại
        wup = rng.choice([2, 4, 6, 10])
        span = rng.choice([6, 12, 18, 24, 30])
        total = wup + span
        in_warm = rng.random() < 0.4
        if in_warm:
            t = rng.randint(1, wup - 1) if wup > 1 else 1
            val = eta_max * F(t, wup)
            ratio_used = None
        else:
            ratio_used = rng.choice(list(COS_OF))
            t = wup + int(span * ratio_used)
            val = _cos_lr(eta_max, eta_min, ratio_used)
            flt = float(eta_min) + 0.5 * float(eta_max - eta_min) * (1 + math.cos(math.pi * (t - wup) / span))
            check(abs(flt - float(val)) < 1e-12, "warmup+cosine: kiểm chứng sai")
        text = (f"Lịch phổ biến khi huấn luyện Transformer: warmup tuyến tính trong $W$ bước, rồi cosine giảm về $\\eta_{{\\min}}$ ở bước $T$. $\\eta_t = \\eta_{{\\max}}\\,t/W$ khi $t < W$; "
                f"khi $W \\le t \\le T$: $\\eta_t = \\eta_{{\\min}} + \\dfrac{{1}}{{2}}(\\eta_{{\\max}} - \\eta_{{\\min}})\\left(1 + \\cos\\dfrac{{\\pi (t - W)}}{{T - W}}\\right)$. "
                f"Với $\\eta_{{\\max}} = {ft(eta_max)}$, $\\eta_{{\\min}} = {ft(eta_min)}$, $W = {wup}$, $T = {total}$, tính $\\eta_{{{t}}}$. ")
        if in_warm:
            steps = [Step(text=f"$t = {t} < W = {wup}$ nên đang warmup:", latex=f"\\eta_{{{t}}} = {ft(eta_max)}\\cdot\\dfrac{{{t}}}{{{wup}}} = {ft(val)}")]
        else:
            steps = [Step(text=f"$t = {t} \\ge W$ nên dùng cosine với $\\dfrac{{t - W}}{{T - W}} = \\dfrac{{{t - wup}}}{{{span}}} = {ft(ratio_used)}$ và $\\cos(\\pi\\cdot {ft(ratio_used)}) = {ft(COS_OF[ratio_used])}$:",
                          latex=f"\\eta_{{{t}}} = {ft(eta_min)} + \\dfrac{{1}}{{2}}\\left({ft(eta_max)} - {ft(eta_min)}\\right)\\left(1 + {fp(COS_OF[ratio_used])}\\right) = {ft(val)}")]
        hints = ["Xét $t$ thuộc giai đoạn nào: $t < W$ (warmup) hay $t \\ge W$ (cosine)?", "Trong giai đoạn cosine, thay $t$ bằng $t - W$ và $T$ bằng $T - W$.", "$\\cos 0 = 1$, $\\cos\\frac{\\pi}{3} = \\frac{1}{2}$, $\\cos\\frac{\\pi}{2} = 0$, $\\cos\\frac{2\\pi}{3} = -\\frac{1}{2}$, $\\cos\\pi = -1$."]
        tag = "warmup-cosine"
    return ob(type_="lr-cosine", lesson=L56, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(val), hints=hints, steps=steps,
              verified=["float-cos", "special-angles"], tags=[tag])


# -------------------------------------------------------------------- wd-step
w1_, w2_ = symbols("w1 w2")


def gen_wd_step(rng: random.Random, level: int, seed: int):
    d = 1 if level == 1 else 2
    k = 2 if level == 3 else 1
    for _ in range(500):
        x = [F(pick(rng, -4, 4)) for _ in range(d)]
        y = F(rng.randint(-9, 9))
        w = [F(rng.randint(-5, 5)) for _ in range(d)]
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 20), F(1, 8)])
        lam = rng.choice([F(1, 2), F(1), F(2), F(1, 5), F(1, 4), F(1, 10), F(3, 10)])
        path, rows = [list(w)], []
        for _t in range(k):
            cur = path[-1]
            e = sum(a * b for a, b in zip(cur, x)) - y
            grad = [e * xi + lam * wi for xi, wi in zip(x, cur)]
            # SymPy: J = 1/2 (w.x - y)^2 + lam/2 |w|^2
            ws = [w1_, w2_][:d]
            J = (sum(wi * _q(xi) for wi, xi in zip(ws, x)) - _q(y)) ** 2 / 2 + _q(lam) / 2 * sum(wi**2 for wi in ws)
            sg = [J.diff(wi).subs(dict(zip(ws, [_q(c) for c in cur]))) for wi in ws]
            check([_q(g) for g in grad] == sg, "gradient của J sai")
            nxt = [wi - eta * g for wi, g in zip(cur, grad)]
            # cách viết thứ hai (weight decay): (1 - eta lam) w - eta e x
            alt = [(1 - eta * lam) * wi - eta * e * xi for wi, xi in zip(cur, x)]
            check(alt == nxt, "dạng weight decay ≠ dạng gradient của J")
            rows.append((e, grad, nxt))
            path.append(nxt)
        if nice([v for p in path for v in p], 300, 150):
            break
    fm = (lambda a: ft(a[0])) if d == 1 else vt
    steps = []
    for t, (e, grad, nxt) in enumerate(rows, 1):
        cur = path[t - 1]
        steps.append(Step(text=f"Bước {t}: sai số $e = \\mathbf{{w}}\\T\\mathbf{{x}} - y = {ft(e)}$; gradient của $J$ là $e\\,\\mathbf{{x}} + \\lambda\\mathbf{{w}}$.",
                          latex=f"\\nabla J = {fm(grad)},\\qquad \\mathbf{{w}} \\leftarrow {fm(cur)} - {fp(eta)}\\cdot {('(' + fm(grad) + ')') if d > 1 else fp(grad[0])} = {fm(nxt)}"))
    return ob(type_="wd-step", lesson=L57, level=level, seed=seed,
              prompt=Prompt(text=f"Một mẫu dữ liệu $\\mathbf{{x}} = {fm(x)}$, $y = {ft(y)}$. Mô hình $\\hat{{y}} = \\mathbf{{w}}\\T\\mathbf{{x}}$, hàm mục tiêu có regularization L2 "
                                 f"$J(\\mathbf{{w}}) = \\dfrac{{1}}{{2}}(\\hat{{y}} - y)^2 + \\dfrac{{\\lambda}}{{2}}\\|\\mathbf{{w}}\\|_2^2$ với $\\lambda = {ft(lam)}$. "
                                 f"Chạy gradient descent ${k}$ bước với $\\eta = {ft(eta)}$ từ $\\mathbf{{w}}_0 = {fm(w)}$. Tính $\\mathbf{{w}}_{k}$. " + (NUM_HINT if d == 1 else VEC_HINT)),
              answer=num_ans(path[-1][0]) if d == 1 else vec_ans(path[-1]),
              hints=["$\\nabla_{\\mathbf{w}} \\dfrac{\\lambda}{2}\\|\\mathbf{w}\\|^2 = \\lambda\\mathbf{w}$.", "Gradient của $\\frac{1}{2}(\\mathbf{w}\\T\\mathbf{x} - y)^2$ là $(\\mathbf{w}\\T\\mathbf{x} - y)\\,\\mathbf{x}$.",
                     "Có thể viết gọn: $\\mathbf{w} \\leftarrow (1 - \\eta\\lambda)\\mathbf{w} - \\eta\\,e\\,\\mathbf{x}$ — thành phần $(1-\\eta\\lambda)$ là weight decay (co trọng số về $0$)."],
              steps=steps, verified=["sympy-gradient", "weight-decay-identity"], tags=[f"{d}d", f"{k}step"])


# ---------------------------------------------------------------- adamw-step
def gen_adamw_step(rng: random.Random, level: int, seed: int):
    p = 1 if level <= 2 else 2
    mode = "l2" if level == 1 else "decoupled" if level == 2 else rng.choice(["l2", "decoupled"])
    for _ in range(500):
        theta = [F(pick(rng, -9, 9)) for _ in range(p)]
        g = [F(pick(rng, -9, 9)) for _ in range(p)]
        eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 10), F(1, 8), F(1)])
        lam = rng.choice([F(1, 2), F(1, 4), F(1, 10), F(1, 5), F(1), F(2)])
        if mode == "l2":
            gp = [gi + lam * ti for gi, ti in zip(g, theta)]
            if any(v == 0 for v in gp):
                continue
            new = [ti - eta * ((gv > 0) - (gv < 0)) for ti, gv in zip(theta, gp)]
        else:
            new = [ti - eta * (((gi > 0) - (gi < 0)) + lam * ti) for ti, gi in zip(theta, g)]
        if nice(new, 100, 200):
            break
    # kiểm chứng độc lập: bước đầu của Adam có m_hat/sqrt(v_hat) = g/|g| (tính lại từ m, v với beta ngẫu nhiên)
    b1, b2 = F(9, 10), F(99, 100)
    src = [gi + lam * ti for gi, ti in zip(g, theta)] if mode == "l2" else g
    for sv in src:
        m, v = (1 - b1) * sv, (1 - b2) * sv * sv
        mh, vh = m / (1 - b1), v / (1 - b2)
        check(vh == sv * sv and mh / abs(sv) == F((sv > 0) - (sv < 0)), "bước đầu Adam ≠ sign(g)")
    other = "Adam + L2 (cộng $\\lambda\\theta$ vào gradient trước khi tính $m, v$)" if mode == "l2" else "AdamW (weight decay tách rời, áp dụng trực tiếp lên $\\theta$)"
    sgn = lambda vals: ", ".join(ft((v > 0) - (v < 0)) for v in vals)
    fm = (lambda a: ft(a[0])) if p == 1 else vt
    if mode == "l2":
        steps = [Step(text="Gradient hiệu dụng $g' = g + \\lambda\\theta_0$ (L2 đi vào cả $m$ lẫn $v$):", latex=f"g' = {fm(gp)}"),
                 Step(text="Bước đầu tiên của Adam (đặt $\\epsilon = 0$) có $\\hat{m}_1/\\sqrt{\\hat{v}_1} = g'/|g'| = \\mathrm{sign}(g')$ ở mỗi thành phần:", latex=f"\\theta_1 = {fm(theta)} - {fp(eta)}\\cdot \\mathrm{{sign}}(g') = {fm(new)}")]
    else:
        steps = [Step(text="Bước đầu tiên của Adam (đặt $\\epsilon = 0$): $\\hat{m}_1/\\sqrt{\\hat{v}_1} = \\mathrm{sign}(g)$ ở mỗi thành phần (không phụ thuộc $\\lambda$).", latex=f"\\mathrm{{sign}}(g) = {fm([F((v > 0) - (v < 0)) for v in g])}"),
                 Step(text="Weight decay tách rời: $\\theta_1 = \\theta_0 - \\eta\\left(\\mathrm{sign}(g) + \\lambda\\theta_0\\right)$:", latex=f"\\theta_1 = {fm(theta)} - {fp(eta)}\\left(\\mathrm{{sign}}(g) + {fp(lam)}\\cdot {fm(theta)}\\right) = {fm(new)}")]
    upd = ("\\theta_t = \\theta_{t-1} - \\eta\\,\\dfrac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}},\\quad \\text{với } m_t, v_t \\text{ tính từ } g'_t = g_t + \\lambda\\theta_{t-1}" if mode == "l2"
           else "\\theta_t = \\theta_{t-1} - \\eta\\left(\\dfrac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}} + \\lambda\\,\\theta_{t-1}\\right),\\quad \\text{với } m_t, v_t \\text{ tính từ } g_t")
    upd = upd.replace("\\text{với }", "\\text{ with }").replace("\\text{ tính từ }", "\\text{ from }") if False else upd
    return ob(type_="adamw-step", lesson=L57, level=level, seed=seed,
              prompt=Prompt(text=f"Bước **đầu tiên** của {other}, đặt $\\epsilon = 0$ và $m_0 = v_0 = 0$. $\\theta_0 = {fm(theta)}$, gradient của mất mát $g_1 = {fm(g)}$, $\\eta = {ft(eta)}$, $\\lambda = {ft(lam)}$. "
                                 f"Tính $\\theta_1$. " + (NUM_HINT if p == 1 else VEC_HINT)),
              answer=num_ans(new[0]) if p == 1 else vec_ans(new),
              hints=["Ở bước $t = 1$ với $m_0 = v_0 = 0$: sau hiệu chỉnh độ chệch, $\\hat{m}_1 = g_1$ và $\\hat{v}_1 = g_1^2$, nên $\\hat{m}_1 / \\sqrt{\\hat{v}_1} = \\mathrm{sign}(g_1)$ (khi $\\epsilon = 0$).",
                     "Adam + L2: dùng $g' = g + \\lambda\\theta$ thay cho $g$. AdamW: giữ nguyên $g$, thêm số hạng $\\lambda\\theta$ **ngoài** phần chia cho $\\sqrt{\\hat{v}}$.",
                     "Khác biệt then chốt: trong Adam + L2, độ lớn decay bị chia cho $\\sqrt{\\hat{v}}$; trong AdamW thì không."],
              steps=steps, verified=["first-step-sign-identity", "fraction-simulation"], tags=[mode, f"{p}param"])


# ----------------------------------------------------------------- ridge-solve
def gen_ridge_solve(rng: random.Random, level: int, seed: int):
    n, d = {1: (3, 1), 2: (3, 2), 3: (4, 2), 4: (4, 2), 5: (4, 3)}[level]
    for _ in range(1000):
        x = [[F(rng.randint(-3, 3)) for _ in range(d)] for _ in range(n)]
        y = [F(rng.randint(-6, 6)) for _ in range(n)]
        lam = F(rng.choice([1, 2, 3, 4, 5, 6, 8, 10]) if level >= 3 else rng.choice([1, 2, 3, 4]))
        xm = Matrix([[_q(v) for v in r] for r in x])
        ym = Matrix([_q(v) for v in y])
        a = xm.T * xm + _q(lam) * Matrix.eye(d)
        if a.det() == 0 or xm.rank() < d:
            continue
        w = a.LUsolve(xm.T * ym)
        wf = [F(int(v.p), int(v.q)) for v in w]
        if nice(wf, 200, 50):
            break
    # kiểm chứng độc lập: NumPy, gradient triệt tiêu, và ridge co nghiệm về 0 so với OLS
    xn, yn = np.array([[float(v) for v in r] for r in x]), np.array([float(v) for v in y])
    wn = np.linalg.solve(xn.T @ xn + float(lam) * np.eye(d), xn.T @ yn)
    check(np.allclose(wn, [float(v) for v in wf], atol=1e-9), "NumPy ≠ SymPy")
    grad = xm.T * (xm * w - ym) + _q(lam) * w
    check(grad.is_zero_matrix, "gradient của J tại w* ≠ 0")
    wo = np.linalg.lstsq(xn, yn, rcond=None)[0]
    check(np.linalg.norm(wn) <= np.linalg.norm(wo) + 1e-12, "ridge phải có chuẩn ≤ OLS")
    gram = xm.T * xm
    rhs = xm.T * ym
    cols = "c" * d
    xtab = "\\mathbf{X} = " + L_mat(x) + ",\\qquad \\mathbf{y} = " + L_mat([[v] for v in y])
    steps = [Step(text="Tính $\\mathbf{X}\\T\\mathbf{X}$ và $\\mathbf{X}\\T\\mathbf{y}$:", latex="\\mathbf{X}\\T\\mathbf{X} = " + _mt(gram) + ",\\qquad \\mathbf{X}\\T\\mathbf{y} = " + _mt(rhs)),
             Step(text=f"Cộng $\\lambda\\mathbf{{I}}$ với $\\lambda = {ft(lam)}$ và giải hệ $(\\mathbf{{X}}\\T\\mathbf{{X}} + \\lambda\\mathbf{{I}})\\mathbf{{w}} = \\mathbf{{X}}\\T\\mathbf{{y}}$:", latex="\\mathbf{w}^\\star = " + vt(wf))]
    return ob(type_="ridge-solve", lesson=L57, level=level, seed=seed,
              prompt=Prompt(text=f"Hồi quy ridge (L2): $J(\\mathbf{{w}}) = \\dfrac{{1}}{{2}}\\|\\mathbf{{X}}\\mathbf{{w}} - \\mathbf{{y}}\\|^2 + \\dfrac{{\\lambda}}{{2}}\\|\\mathbf{{w}}\\|^2$ với $\\lambda = {ft(lam)}$. Tìm nghiệm tối ưu $\\mathbf{{w}}^\\star$. "
                                 + (NUM_HINT if d == 1 else VEC_HINT), display=[xtab]),
              answer=num_ans(wf[0]) if d == 1 else vec_ans(wf),
              hints=["Đặt $\\nabla J = \\mathbf{X}\\T(\\mathbf{X}\\mathbf{w} - \\mathbf{y}) + \\lambda\\mathbf{w} = \\mathbf{0}$, suy ra $(\\mathbf{X}\\T\\mathbf{X} + \\lambda\\mathbf{I})\\mathbf{w} = \\mathbf{X}\\T\\mathbf{y}$.",
                     "Ma trận $\\mathbf{X}\\T\\mathbf{X} + \\lambda\\mathbf{I}$ luôn khả nghịch khi $\\lambda > 0$ — đây là một lợi ích của ridge.", "Với một đặc trưng: $w = \\dfrac{\\sum x_i y_i}{\\sum x_i^2 + \\lambda}$."],
              steps=steps, verified=["sympy-solve", "numpy-solve", "gradient-zero", "norm-shrinks"], tags=[f"d{d}", f"n{n}"])


def L_mat(rows) -> str:
    body = " \\\\ ".join(" & ".join(ft(v) for v in r) for r in rows)
    return f"\\begin{{bmatrix}} {body} \\end{{bmatrix}}"


def _mt(m: Matrix) -> str:
    return L_mat([[F(int(m[i, j].p), int(m[i, j].q)) for j in range(m.cols)] for i in range(m.rows)])


# ------------------------------------------------------------------- l1-prox
def _soft(z: F, t: F) -> F:
    return F((z > 0) - (z < 0)) * max(abs(z) - t, F(0))


def _prox_ok(w: F, z: F, lam: F, a: F = F(1)) -> bool:
    """Điều kiện tối ưu (dưới vi phân): a(w - z) + lam * s = 0 với s = sign(w), hoặc |a z| <= lam khi w = 0."""
    if w == 0:
        return abs(a * z) <= lam
    return a * (w - z) + lam * ((w > 0) - (w < 0)) == 0


def gen_l1_prox(rng: random.Random, level: int, seed: int):
    if level in (1, 2):
        n = 1 if level == 1 else 3
        for _ in range(100):
            z = [F(rng.randint(-9, 9)) for _ in range(n)]
            lam = F(rng.choice([1, 2, 3, 4, 5])) / rng.choice([1, 2])
            w = [_soft(v, lam) for v in z]
            if n == 1 or (any(v == 0 for v in w) and any(v != 0 for v in w)):
                break
        for wi, zi in zip(w, z):
            check(_prox_ok(wi, zi, lam), "điều kiện tối ưu sai")
        fm = (lambda a: ft(a[0])) if n == 1 else vt
        text = (f"Giải $\\min_{{w}}\\ \\dfrac{{1}}{{2}}(w - z)^2 + \\lambda|w|$" if n == 1 else f"Với mỗi thành phần, giải $\\min_{{w_i}}\\ \\dfrac{{1}}{{2}}(w_i - z_i)^2 + \\lambda|w_i|$ (tức $\\mathrm{{prox}}_{{\\lambda\\|\\cdot\\|_1}}(\\mathbf{{z}})$)")
        text += f" với $z = {fm(z)}$ và $\\lambda = {ft(lam)}$. " if n == 1 else f" với $\\mathbf{{z}} = {fm(z)}$ và $\\lambda = {ft(lam)}$. "
        steps = [Step(text="Soft-thresholding: $S_\\lambda(z) = \\mathrm{sign}(z)\\,\\max(|z| - \\lambda,\\ 0)$ áp dụng cho từng thành phần.",
                      latex=("w^\\star = " + ft(w[0])) if n == 1 else "\\mathbf{w}^\\star = " + vt(w))]
        ans = num_ans(w[0]) if n == 1 else vec_ans(w)
        tag = f"{n}d"
        hints = ["Nếu $|z| \\le \\lambda$ thì nghiệm là $0$ (L1 tạo ra nghiệm thưa); nếu không thì kéo $z$ về phía $0$ một đoạn $\\lambda$.", "$S_\\lambda(z) = \\mathrm{sign}(z)\\max(|z| - \\lambda, 0)$."]
        text += (NUM_HINT if n == 1 else VEC_HINT)
        disp = []
    elif level == 3:  # có hệ số a: min a/2 (w - z)^2 + lam|w| -> S_{lam/a}(z)
        for _ in range(100):
            a = F(rng.choice([2, 3, 4, 5]))
            z = F(rng.randint(-9, 9))
            lam = F(rng.choice([1, 2, 3, 4, 6, 8, 9, 12]))
            w = _soft(z, lam / a)
            if nice([w], 60, 50):
                break
        check(_prox_ok(w, z, lam, a), "điều kiện tối ưu sai (có a)")
        text = f"Giải $\\min_{{w}}\\ \\dfrac{{a}}{{2}}(w - z)^2 + \\lambda|w|$ với $a = {a.numerator}$, $z = {ft(z)}$, $\\lambda = {ft(lam)}$. " + NUM_HINT
        steps = [Step(text="Chia hai vế cho $a$: bài toán tương đương $\\min \\frac{1}{2}(w - z)^2 + \\frac{\\lambda}{a}|w|$, nên ngưỡng là $\\lambda/a$.", latex=f"w^\\star = S_{{\\lambda/a}}(z) = S_{{{ft(lam / a)}}}({ft(z)}) = {ft(w)}")]
        ans, tag, disp = num_ans(w), "scaled", []
        hints = ["Chia cho $a$ để đưa về dạng chuẩn: ngưỡng trở thành $\\lambda/a$.", "$S_t(z) = \\mathrm{sign}(z)\\max(|z| - t, 0)$."]
    else:  # bước proximal gradient (ISTA) cho f(w) = a/2 (w - c)^2
        k = 1 if level == 4 else 2
        for _ in range(500):
            a = [F(rng.choice([1, 2, 3, 4])) for _ in range(k)]
            c = [F(rng.randint(-8, 8)) for _ in range(k)]
            w0 = [F(rng.randint(-6, 6)) for _ in range(k)]
            eta = rng.choice([F(1, 2), F(1, 4), F(1, 5), F(1, 8), F(1, 10)])
            lam = F(rng.choice([1, 2, 3, 4, 5, 6]))
            zs = [w0[i] - eta * a[i] * (w0[i] - c[i]) for i in range(k)]
            w = [_soft(zs[i], eta * lam) for i in range(k)]
            if nice(w + zs, 100, 80) and (k == 1 or any(v == 0 for v in w)):
                break
        for i in range(k):  # tối ưu: bài toán prox tương ứng thỏa điều kiện dưới vi phân
            check(_prox_ok(w[i], zs[i], eta * lam), "prox gradient: điều kiện tối ưu sai")
        fm = (lambda arr: ft(arr[0])) if k == 1 else vt
        text = (f"Một bước proximal gradient (ISTA) cho $\\min\\ f(w) + \\lambda\\|w\\|_1$ với $f(w) = \\sum_i \\dfrac{{a_i}}{{2}}(w_i - c_i)^2$: "
                f"$w^{{+}} = S_{{\\eta\\lambda}}\\!\\left(w - \\eta\\nabla f(w)\\right)$. Với $a = {fm(a)}$, $c = {fm(c)}$, $\\eta = {ft(eta)}$, $\\lambda = {ft(lam)}$, bắt đầu từ $w = {fm(w0)}$, tính $w^{{+}}$. ")
        text += NUM_HINT if k == 1 else VEC_HINT
        steps = [Step(text="Bước gradient thuần túy rồi soft-thresholding với ngưỡng $\\eta\\lambda$:", latex=f"z = w - \\eta\\nabla f(w) = {fm(zs)},\\qquad w^{{+}} = S_{{{ft(eta * lam)}}}(z) = {fm(w)}")]
        ans, tag, disp = (num_ans(w[0]) if k == 1 else vec_ans(w)), "ista", []
        hints = ["$\\nabla f(w)_i = a_i(w_i - c_i)$.", "Làm gradient step trước (không có L1), rồi áp dụng soft-thresholding với ngưỡng $\\eta\\lambda$ (không phải $\\lambda$)."]
    return ob(type_="l1-prox", lesson=L57, level=level, seed=seed, prompt=Prompt(text=text, display=disp), answer=ans, hints=hints, steps=steps,
              verified=["subdifferential-optimality"], tags=[tag])


SPECS = [
    TopicSpec("lr-decay", L56, (1, 2, 3, 4), gen_lr_decay),
    TopicSpec("lr-cosine", L56, (1, 2, 3, 4, 5), gen_lr_cosine),
    TopicSpec("wd-step", L57, (1, 2, 3), gen_wd_step),
    TopicSpec("adamw-step", L57, (1, 2, 3, 4, 5), gen_adamw_step),
    TopicSpec("ridge-solve", L57, (1, 2, 3, 4, 5), gen_ridge_solve),
    TopicSpec("l1-prox", L57, (1, 2, 3, 4, 5), gen_l1_prox),
]
