"""Bài 6.1 Hồi quy tuyến tính: ols-fit, ols-metrics. Bài 6.2 Hồi quy logistic và softmax: logreg-grad, softmax-grad, logreg-decision."""
from __future__ import annotations

import random
from math import isqrt

import numpy as np
from sympy import Matrix, Rational, exp, log, simplify, symbols

from ._m import (F, L, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, cb, check, choice_exercise, data_table, fp, ft, mat_ans, mat_tex, nice, num_ans, pick, points_table,
                 vec_ans, vt)

L61, L62 = "6.1", "6.2"


def _q(v: F) -> Rational:
    return Rational(v.numerator, v.denominator)


def _sq(v: F):
    a, b = isqrt(v.numerator), isqrt(v.denominator)
    return F(a, b) if v >= 0 and a * a == v.numerator and b * b == v.denominator else None


# --------------------------------------------------------------------- ols-fit
def gen_ols_fit(rng: random.Random, level: int, seed: int):
    n = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5}[level]
    cols = {1: ["x"], 2: ["x", "1"], 3: ["x1", "x2"], 4: ["x1", "x2", "1"], 5: ["x", "1"]}[level]
    for _ in range(1000):
        feats = []
        for c in cols:
            feats.append([F(1)] * n if c == "1" else [F(rng.randint(-4, 4)) for _ in range(n)])
        y = [F(rng.randint(-8, 8)) for _ in range(n)]
        xm = Matrix([[_q(feats[j][i]) for j in range(len(cols))] for i in range(n)])
        if xm.rank() < len(cols):
            continue
        ym = Matrix([_q(v) for v in y])
        gram, rhs = xm.T * xm, xm.T * ym
        w = gram.LUsolve(rhs)
        wf = [F(int(v.p), int(v.q)) for v in w]
        if nice(wf, 200, 60):
            break
    # kiểm chứng độc lập: NumPy lstsq; phần dư trực giao với các cột của X
    xn = np.array([[float(f[i]) for f in feats] for i in range(n)])
    wn = np.linalg.lstsq(xn, np.array([float(v) for v in y]), rcond=None)[0]
    check(np.allclose(wn, [float(v) for v in wf], atol=1e-9), "NumPy lstsq ≠ SymPy")
    check((xm.T * (xm * w - ym)).is_zero_matrix, "phần dư không trực giao với cột của X")
    names = [c if c != "1" else "1" for c in cols]
    tbl = data_table([c for c in cols if c != "1"] + ["y"], [f for c, f in zip(cols, feats) if c != "1"] + [y])
    has_b = "1" in cols
    model = {1: "\\hat{y} = wx", 2: "\\hat{y} = wx + b", 3: "\\hat{y} = w_1x_1 + w_2x_2", 4: "\\hat{y} = w_1x_1 + w_2x_2 + b", 5: "\\hat{y} = wx + b"}[level]
    order = {1: "$w$", 2: "$(w, b)$", 3: "$(w_1, w_2)$", 4: "$(w_1, w_2, b)$", 5: "$(w, b)$"}[level]
    steps = [Step(text="Phương trình chuẩn $\\mathbf{X}\\T\\mathbf{X}\\,\\mathbf{w} = \\mathbf{X}\\T\\mathbf{y}$ với $\\mathbf{X}$ có cột " + ", ".join(f"${c}$" if c != "1" else "$\\mathbf{1}$" for c in cols) + ":",
                  latex="\\mathbf{X}\\T\\mathbf{X} = " + L.mat(gram) + ",\\qquad \\mathbf{X}\\T\\mathbf{y} = " + L.mat(rhs)),
             Step(text="Giải hệ:", latex="\\mathbf{w}^\\star = " + vt(wf))]
    if level == 5:
        xn_val = F(rng.randint(-4, 4))
        pred = wf[0] * xn_val + wf[1]
        steps.append(Step(text=f"Dự đoán tại $x = {ft(xn_val)}$:", latex=f"\\hat{{y}} = {ft(wf[0])}\\cdot {fp(xn_val)} + {fp(wf[1])} = {ft(pred)}"))
        text = f"Khớp mô hình ${model}$ bằng bình phương tối thiểu (OLS) trên dữ liệu dưới đây, rồi dự đoán $\\hat{{y}}$ tại $x = {ft(xn_val)}$. " + NUM_HINT
        ans = num_ans(pred)
    else:
        text = f"Khớp mô hình ${model}$ bằng bình phương tối thiểu (OLS) trên dữ liệu dưới đây (nghiệm của phương trình chuẩn). Tìm {order}. " + (NUM_HINT if len(wf) == 1 else VEC_HINT)
        ans = num_ans(wf[0]) if len(wf) == 1 else vec_ans(wf)
    return cb(type_="ols-fit", lesson=L61, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=ans,
              hints=["OLS tối thiểu $\\sum_i (\\hat{y}_i - y_i)^2$; điều kiện cực tiểu là phương trình chuẩn $\\mathbf{X}\\T\\mathbf{X}\\mathbf{w} = \\mathbf{X}\\T\\mathbf{y}$.",
                     "Thêm một cột toàn số $1$ vào $\\mathbf{X}$ nếu mô hình có hệ số chặn $b$." if has_b else "Một đặc trưng, không hệ số chặn: $w = \\dfrac{\\sum x_iy_i}{\\sum x_i^2}$.",
                     "Với một đặc trưng có hệ số chặn: $w = \\dfrac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2}$ và $b = \\bar{y} - w\\bar{x}$."],
              steps=steps, verified=["sympy-normal-equations", "numpy-lstsq", "residual-orthogonality"], tags=["ml", f"{len(cols)}param"])


# ------------------------------------------------------------------ ols-metrics
def gen_ols_metrics(rng: random.Random, level: int, seed: int):
    n = rng.choice([4, 5, 6]) if level != 4 else 4
    for _ in range(20000):
        y = [F(rng.randint(-6, 9)) for _ in range(n)]
        yh = [yi + F(rng.randint(-4, 4)) for yi in y]
        if len(set(y)) < 2:
            continue
        err = [a - b for a, b in zip(yh, y)]
        sse = sum(e * e for e in err)
        sst = sum((v - sum(y) / n) ** 2 for v in y)
        if sse == 0:
            continue
        if level == 1:
            val = sse / n
        elif level == 2:
            val = sum(abs(e) for e in err) / n
        elif level == 3:
            val = 1 - sse / sst
        elif level == 4:
            val = _sq(sse / n)
            if val is None:
                continue
        else:
            p = rng.choice([1, 2])
            if n - p - 1 <= 0:
                continue
            val = 1 - (1 - (1 - sse / sst)) * F(n - 1, n - p - 1)
        if nice([val], 300, 200):
            break
    name = {1: "MSE", 2: "MAE", 3: "R^2", 4: "RMSE", 5: "R^2_{adj}"}[level]
    # kiểm chứng độc lập: NumPy
    yn, yhn = np.array([float(v) for v in y]), np.array([float(v) for v in yh])
    ref = {1: np.mean((yhn - yn) ** 2), 2: np.mean(np.abs(yhn - yn)), 3: 1 - np.sum((yhn - yn) ** 2) / np.sum((yn - yn.mean()) ** 2),
           4: np.sqrt(np.mean((yhn - yn) ** 2)), 5: None}[level]
    if level == 5:
        r2 = 1 - np.sum((yhn - yn) ** 2) / np.sum((yn - yn.mean()) ** 2)
        ref = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    check(abs(ref - float(val)) < 1e-9, "NumPy ≠ phân số")
    tbl = data_table(["y", "\\hat{y}"], [y, yh])
    defs = {1: "MSE $= \\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)^2$", 2: "MAE $= \\frac{1}{n}\\sum_i |\\hat{y}_i - y_i|$", 3: "$R^2 = 1 - \\dfrac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$",
            4: "RMSE $= \\sqrt{\\mathrm{MSE}}$", 5: f"$R^2_{{\\mathrm{{adj}}}} = 1 - (1 - R^2)\\dfrac{{n-1}}{{n-p-1}}$ với $p = {p if level == 5 else 0}$ đặc trưng"}
    steps = [Step(text="Sai số $e_i = \\hat{y}_i - y_i$:", latex="e = (" + ", ".join(ft(e) for e in err) + ")"),
             Step(text=(f"$\\sum |e_i| = {ft(sum(abs(e) for e in err))}$" if level == 2 else f"$\\sum e_i^2 = {ft(sse)}$") + (f" và $\\sum (y_i - \\bar{{y}})^2 = {ft(sst)}$" if level in (3, 5) else "") + ".", latex=f"{name} = {ft(val)}")]
    return cb(type_="ols-metrics", lesson=L61, level=level, seed=seed,
              prompt=Prompt(text=f"Cho giá trị thật $y$ và dự đoán $\\hat{{y}}$ của một mô hình hồi quy ($n = {n}$). Tính {defs[level]}. " + NUM_HINT, display=[tbl]), answer=num_ans(val),
              hints=["Tính sai số từng mẫu $e_i = \\hat{y}_i - y_i$ trước.", "$R^2 = 1$ là khớp hoàn hảo; $R^2 = 0$ nghĩa là không tốt hơn dự đoán trung bình $\\bar{y}$.", "RMSE cùng đơn vị với $y$ (MSE thì bình phương đơn vị)."],
              steps=steps, verified=["numpy"], tags=["ml", name])


# ------------------------------------------------------------------ logreg-grad
_PS = [F(1, 2), F(1, 4), F(3, 4), F(1, 5), F(4, 5), F(1, 10), F(9, 10), F(1, 3), F(2, 3)]

# kiểm chứng biểu tượng (một lần): d/dz của log-loss với p = sigmoid(z) bằng p - y
_z, _y = symbols("z y")
_p = 1 / (1 + exp(-_z))
_loss = -_y * log(_p) - (1 - _y) * log(1 - _p)
check(simplify((_loss.diff(_z) - (_p - _y)).rewrite(exp)) == 0, "đạo hàm log-loss theo logit ≠ p - y")


def gen_logreg_grad(rng: random.Random, level: int, seed: int):
    n = {1: 1, 2: 3, 3: 3, 4: 3, 5: 3}[level]
    d = 2 if level >= 4 else 1
    with_b = level >= 3
    xs = [[F(pick(rng, -3, 3)) for _ in range(d)] for _ in range(n)]
    ys = [rng.randint(0, 1) for _ in range(n)]
    ps = [rng.choice(_PS) for _ in range(n)]
    lam, w0 = (F(rng.choice([1, 2, 5])) / rng.choice([1, 2, 5, 10]), [F(rng.randint(-3, 3)) for _ in range(d)]) if level == 5 else (None, None)
    res = [p - yv for p, yv in zip(ps, ys)]
    gw = [sum(r * x[j] for r, x in zip(res, xs)) / n for j in range(d)]
    gb = sum(res) / n
    if level == 5:
        gw = [g + lam * w for g, w in zip(gw, w0)]
    out = gw + ([gb] if with_b else [])
    # kiểm chứng độc lập: sai phân hữu hạn của log-loss theo logit z_i (p_i = sigma(z_i)), dùng z_i = logit(p_i)
    for i in range(n):
        zi = float(np.log(float(ps[i]) / (1 - float(ps[i]))))
        h = 1e-6
        f_ = lambda z: -ys[i] * np.log(1 / (1 + np.exp(-z))) - (1 - ys[i]) * np.log(1 - 1 / (1 + np.exp(-z)))
        num = (f_(zi + h) - f_(zi - h)) / (2 * h)
        check(abs(num - float(res[i])) < 1e-6, "d loss / d z ≠ p - y (sai phân hữu hạn)")
    cols = [[x[j] for x in xs] for j in range(d)] + [[F(v) for v in ys], ps]
    nms = ([f"x_{j + 1}" for j in range(d)] if d > 1 else ["x"]) + ["y", "p"]
    steps = [Step(text="Sai số $p_i - y_i$:", latex="(" + ", ".join(ft(r) for r in res) + ")")]
    fmt = (lambda v: ft(v[0])) if len(out) == 1 else vt
    steps.append(Step(text="Gradient của log-loss trung bình (với $p = \\sigma(z)$ thì $\\partial\\ell/\\partial z = p - y$):", latex="\\nabla = " + fmt(out)))
    reg = ""
    if level == 5:
        reg = f" Hàm mục tiêu có thêm $\\dfrac{{\\lambda}}{{2}}\\|\\mathbf{{w}}\\|^2$ với $\\lambda = {ft(lam)}$ tại $\\mathbf{{w}} = {vt(w0) if d > 1 else ft(w0[0])}$ (gradient thêm $\\lambda\\mathbf{{w}}$)."
    target = ("$\\dfrac{\\partial J}{\\partial w}$" if len(out) == 1 else "$\\nabla J$ theo " + ("$(w_1, w_2, b)$" if d > 1 else "$(w, b)$"))
    return cb(type_="logreg-grad", lesson=L62, level=level, seed=seed,
              prompt=Prompt(text=f"Hồi quy logistic với log-loss trung bình $J = -\\dfrac{{1}}{{n}}\\sum_i\\big[y_i\\ln p_i + (1-y_i)\\ln(1-p_i)\\big]$, $p_i = \\sigma(\\mathbf{{w}}\\T\\mathbf{{x}}_i" + (" + b" if with_b else "") +
                                 f")$. Xác suất dự đoán $p_i$ cho trong bảng (coi như đã tính từ tham số hiện tại; không cần tính lại từ $\\sigma$). Tính {target} tại điểm hiện tại.{reg} " + (NUM_HINT if len(out) == 1 else VEC_HINT),
                            display=[data_table(nms, cols)]),
              answer=num_ans(out[0]) if len(out) == 1 else vec_ans(out),
              hints=["Với sigmoid và log-loss, đạo hàm theo logit gọn đến mức: $\\dfrac{\\partial \\ell_i}{\\partial z_i} = p_i - y_i$.", "$\\dfrac{\\partial J}{\\partial w_j} = \\dfrac{1}{n}\\sum_i (p_i - y_i)\\,x_{ij}$ và $\\dfrac{\\partial J}{\\partial b} = \\dfrac{1}{n}\\sum_i (p_i - y_i)$.",
                     "Gradient của $\\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2$ là $\\lambda\\mathbf{w}$ (không cộng vào $b$)." if level == 5 else "Không cần tính $\\sigma$ hay $\\ln$: các $p_i$ đã cho."],
              steps=steps, verified=["sympy-identity-p-minus-y", "finite-difference"], tags=["ml", f"{d}feat"])


# ----------------------------------------------------------------- softmax-grad
def _dist(rng, c):
    base = [F(rng.randint(1, 6)) for _ in range(c)]
    tot = sum(base)
    return [b / tot for b in base]


def gen_softmax_grad(rng: random.Random, level: int, seed: int):
    c = 3 if level < 4 else rng.choice([3, 4])
    for _ in range(100):
        p = _dist(rng, c)
        if len({v.denominator for v in p}) >= 1 and all(v.denominator <= 20 for v in p):
            break
    t = rng.randrange(c)
    onehot = [F(1) if i == t else F(0) for i in range(c)]
    g = [pi - oi for pi, oi in zip(p, onehot)]
    check(sum(g) == 0 and sum(p) == 1, "tổng gradient logit phải bằng 0")
    # kiểm chứng độc lập: sai phân hữu hạn của cross-entropy theo logit z (p = softmax(z), z = log p cho tổng 1)
    z = np.log([float(v) for v in p])
    sm = lambda zz: np.exp(zz - zz.max()) / np.exp(zz - zz.max()).sum()
    for j in range(c):
        hh = 1e-6
        zp, zm = z.copy(), z.copy()
        zp[j] += hh
        zm[j] -= hh
        num = (-np.log(sm(zp)[t]) + np.log(sm(zm)[t])) / (2 * hh)
        check(abs(num - float(g[j])) < 1e-6, "dL/dz ≠ p - y (sai phân hữu hạn)")
    pt = "(" + ", ".join(ft(v) for v in p) + ")\\T"
    if level <= 2:
        text = f"Softmax cho xác suất $\\mathbf{{p}} = {pt}$ trên ${c}$ lớp; nhãn thật là lớp ${t + 1}$ (one-hot $\\mathbf{{y}}$ có $1$ ở vị trí ${t + 1}$). Tính gradient của cross-entropy theo logit $\\mathbf{{z}}$. " + VEC_HINT
        ans = vec_ans(g)
        steps = [Step(text="Với softmax và cross-entropy: $\\nabla_{\\mathbf{z}} \\ell = \\mathbf{p} - \\mathbf{y}$.", latex="\\mathbf{p} - \\mathbf{y} = " + vt(g))]
        tag = "logit"
    else:
        d = 2
        x = [F(pick(rng, -3, 3)) for _ in range(d)]
        gw = [[gi * xj for xj in x] for gi in g]
        if level == 3:
            ans = mat_ans(gw)
            text = (f"Mô hình softmax $\\mathbf{{z}} = \\mathbf{{W}}\\mathbf{{x}}$ với $\\mathbf{{W}} \\in \\R^{{{c}\\times 2}}$, mẫu $\\mathbf{{x}} = {vt(x)}$, xác suất $\\mathbf{{p}} = {pt}$, nhãn thật là lớp ${t + 1}$. "
                    "Tính gradient của cross-entropy theo $\\mathbf{W}$ (ma trận cùng kích thước với $\\mathbf{W}$). Nhập từng phần tử.")
            steps = [Step(text="$\\nabla_{\\mathbf{W}}\\ell = (\\mathbf{p} - \\mathbf{y})\\,\\mathbf{x}\\T$ (tích ngoài của vector sai số và $\\mathbf{x}$):", latex="(\\mathbf{p} - \\mathbf{y})\\mathbf{x}\\T = " + vt(g) + vt(x) + " = " + mat_tex(gw))]
            tag = "weights"
        else:
            # trung bình hai mẫu
            x2 = [F(pick(rng, -3, 3)) for _ in range(d)]
            p2 = _dist(rng, c)
            t2 = rng.randrange(c)
            g2 = [pi - (1 if i == t2 else 0) for i, pi in enumerate(p2)]
            gw2 = [[(g[i] * x[j] + g2[i] * x2[j]) / 2 for j in range(d)] for i in range(c)]
            ans = mat_ans(gw2)
            text = (f"Softmax $\\mathbf{{z}} = \\mathbf{{W}}\\mathbf{{x}}$, $\\mathbf{{W}} \\in \\R^{{{c}\\times 2}}$. Hai mẫu: (1) $\\mathbf{{x}}_1 = {vt(x)}$, $\\mathbf{{p}}_1 = {pt}$, lớp thật ${t + 1}$; "
                    f"(2) $\\mathbf{{x}}_2 = {vt(x2)}$, $\\mathbf{{p}}_2 = {'(' + ', '.join(ft(v) for v in p2) + ')' + chr(92) + 'T'}$, lớp thật ${t2 + 1}$. Tính gradient của cross-entropy **trung bình** theo $\\mathbf{{W}}$. Nhập từng phần tử.")
            steps = [Step(text="Gradient từng mẫu $(\\mathbf{p}_k - \\mathbf{y}_k)\\mathbf{x}_k\\T$ rồi lấy trung bình:", latex="\\nabla_{\\mathbf{W}} J = \\dfrac{1}{2}\\sum_{k=1}^{2}(\\mathbf{p}_k - \\mathbf{y}_k)\\mathbf{x}_k\\T = " + mat_tex(gw2))]
            tag = "batch"
            check(all(sum(gw2[i][j] for i in range(c)) == 0 for j in range(d)), "tổng các hàng gradient theo từng cột phải bằng 0")
    return cb(type_="softmax-grad", lesson=L62, level=level, seed=seed, prompt=Prompt(text=text), answer=ans,
              hints=["Gradient của cross-entropy theo logit là $\\mathbf{p} - \\mathbf{y}$ ($\\mathbf{y}$ là vector one-hot).", "Theo $\\mathbf{W}$: nhân thêm $\\mathbf{x}\\T$ (quy tắc chuỗi: $\\mathbf{z} = \\mathbf{W}\\mathbf{x}$).",
                     "Kiểm tra nhanh: tổng các thành phần của $\\mathbf{p} - \\mathbf{y}$ luôn bằng $0$."],
              steps=steps, verified=["finite-difference", "sum-to-zero"], tags=["ml", tag])


# -------------------------------------------------------------- logreg-decision
_PYTH = [(3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6), (8, 15), (15, 8)]


def gen_logreg_decision(rng: random.Random, level: int, seed: int):
    if level in (4,):
        w = [F(v) * rng.choice([1, -1]) for v in rng.choice(_PYTH)]
    else:
        w = [F(pick(rng, -4, 4)), F(pick(rng, -4, 4))]
    b = F(rng.randint(-6, 6))
    x = [F(rng.randint(-5, 5)), F(rng.randint(-5, 5))]
    z = w[0] * x[0] + w[1] * x[1] + b
    wt = vt(w)
    if level == 1:
        text = f"Mô hình logistic có $z = \\mathbf{{w}}\\T\\mathbf{{x}} + b$ với $\\mathbf{{w}} = {wt}$, $b = {ft(b)}$. Tính $z$ tại $\\mathbf{{x}} = {vt(x)}$. " + NUM_HINT
        ans, steps = num_ans(z), [Step(text="Thế số:", latex=f"z = {ft(w[0])}\\cdot {fp(x[0])} + {fp(w[1])}\\cdot {fp(x[1])} + {fp(b)} = {ft(z)}")]
        ch = None
    elif level == 2:
        while z == 0:
            x[0] += 1
            z = w[0] * x[0] + w[1] * x[1] + b
        text = f"Mô hình logistic có $z = \\mathbf{{w}}\\T\\mathbf{{x}} + b$ với $\\mathbf{{w}} = {wt}$, $b = {ft(b)}$, dự đoán lớp $1$ khi $\\sigma(z) \\ge 1/2$. Điểm $\\mathbf{{x}} = {vt(x)}$ được dự đoán thuộc lớp nào?"
        label = "Lớp 1" if z > 0 else "Lớp 0"
        ch, ans = choice_exercise(rng, label, ["Lớp 1" if label == "Lớp 0" else "Lớp 0"])
        steps = [Step(text=f"$z = {ft(z)}$ " + ("dương" if z > 0 else "âm") + f", và $\\sigma(z) \\ge 1/2 \\Leftrightarrow z \\ge 0$, nên dự đoán **{label}**.")]
    elif level == 3:
        while w[0] == 0:
            w[0] = F(pick(rng, -4, 4))
        x2 = F(rng.randint(-4, 4))
        x1 = -(w[1] * x2 + b) / w[0]
        text = f"Biên quyết định của mô hình logistic là $z = 0$ với $\\mathbf{{w}} = {vt(w)}$, $b = {ft(b)}$. Tại $x_2 = {ft(x2)}$, biên cắt ở $x_1$ bằng bao nhiêu? " + NUM_HINT
        check(w[0] * x1 + w[1] * x2 + b == 0, "điểm nằm trên biên")
        ans, steps = num_ans(x1), [Step(text="Giải $w_1x_1 + w_2x_2 + b = 0$ theo $x_1$:", latex=f"x_1 = -\\dfrac{{w_2x_2 + b}}{{w_1}} = {ft(x1)}")]
        ch = None
    elif level == 4:
        norm = F(isqrt(int(w[0] ** 2 + w[1] ** 2)))
        check(norm * norm == w[0] ** 2 + w[1] ** 2, "||w|| nguyên")
        dist = abs(z) / norm
        text = f"Biên quyết định $z = \\mathbf{{w}}\\T\\mathbf{{x}} + b = 0$ với $\\mathbf{{w}} = {wt}$, $b = {ft(b)}$. Tính khoảng cách Euclid từ $\\mathbf{{x}} = {vt(x)}$ đến biên. " + NUM_HINT
        ans, steps = num_ans(dist), [Step(text="Khoảng cách từ điểm đến siêu phẳng: $|z|/\\|\\mathbf{w}\\|_2$.", latex=f"\\dfrac{{|{ft(z)}|}}{{\\sqrt{{{ft(w[0] ** 2)} + {ft(w[1] ** 2)}}}}} = \\dfrac{{{ft(abs(z))}}}{{{ft(norm)}}} = {ft(dist)}")]
        ch = None
        yn = abs(float(z)) / np.hypot(float(w[0]), float(w[1]))
        check(abs(yn - float(dist)) < 1e-12, "khoảng cách NumPy ≠ phân số")
    else:
        pts = [[F(rng.randint(-4, 4)), F(rng.randint(-4, 4))] for _ in range(5)]
        zs = [w[0] * p[0] + w[1] * p[1] + b for p in pts]
        if any(v == 0 for v in zs):
            pts = [[p[0] + F(1, 2), p[1]] for p in pts]
            zs = [w[0] * p[0] + w[1] * p[1] + b for p in pts]
        labs = [rng.randint(0, 1) for _ in pts]
        pred = [1 if v > 0 else 0 for v in zs]
        acc = F(sum(1 for a, c in zip(pred, labs) if a == c), len(pts))
        text = (f"Mô hình logistic $z = \\mathbf{{w}}\\T\\mathbf{{x}} + b$ với $\\mathbf{{w}} = {wt}$, $b = {ft(b)}$, dự đoán lớp $1$ khi $z > 0$. Tính độ chính xác (accuracy) trên $5$ điểm dưới đây. " + NUM_HINT)
        ans = num_ans(acc)
        steps = [Step(text="Tính $z$ từng điểm rồi so dự đoán với nhãn:", latex="z = (" + ", ".join(ft(v) for v in zs) + "),\\quad \\text{pred} = (" + ", ".join(str(v) for v in pred) + ")"),
                 Step(text=f"Đúng ${sum(1 for a, c in zip(pred, labs) if a == c)}/{len(pts)}$:", latex=f"\\mathrm{{accuracy}} = {ft(acc)}")]
        ch = None
        return cb(type_="logreg-decision", lesson=L62, level=level, seed=seed,
                  prompt=Prompt(text=text, display=[points_table(pts, [F(v) for v in labs])]), answer=ans,
                  hints=["Dự đoán lớp $1$ khi $z > 0$ (vì $\\sigma(z) > 1/2 \\Leftrightarrow z > 0$).", "Accuracy = số dự đoán đúng / tổng số điểm."], steps=steps, verified=["recompute"], tags=["ml", "accuracy"])
    return cb(type_="logreg-decision", lesson=L62, level=level, seed=seed, prompt=Prompt(text=text), answer=ans, choices=ch,
              hints=["$z = w_1x_1 + w_2x_2 + b$; biên quyết định (ngưỡng xác suất $1/2$) là tập $z = 0$.", "$\\sigma(z) \\ge 1/2 \\Leftrightarrow z \\ge 0$, nên chỉ cần xét dấu của $z$."],
              steps=steps, verified=["recompute"], tags=["ml", f"level{level}"])


SPECS = [
    TopicSpec("ols-fit", L61, (1, 2, 3, 4, 5), gen_ols_fit),
    TopicSpec("ols-metrics", L61, (1, 2, 3, 4, 5), gen_ols_metrics),
    TopicSpec("logreg-grad", L62, (1, 2, 3, 4, 5), gen_logreg_grad),
    TopicSpec("softmax-grad", L62, (1, 2, 3, 4, 5), gen_softmax_grad),
    TopicSpec("logreg-decision", L62, (1, 2, 3, 4, 5), gen_logreg_decision),
]
