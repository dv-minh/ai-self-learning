"""Bài 7.4 Khởi tạo, BatchNorm, LayerNorm: init-var, norm-compute. Bài 7.5 Dropout, residual, embedding: dropout-calc, resid-grad, embed-lookup."""
from __future__ import annotations

import random

import numpy as np

from ._d import (F, L, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, choice_exercise, dl, fp, ft, mat_ans, mat_tex, mm, mv, nice, num_ans, pick, rand_mat, rand_vec,
                 sqrt_q, vec_ans, vt)

L74, L75 = "7.4", "7.5"
SQUARES = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 196, 256, 400, 576, 784]
SUMS_6 = [6, 24, 54, 96, 150, 216, 294, 384]  # 6/s là bình phương hữu tỉ


# ---------------------------------------------------------------------- init-var
def gen_init_var(rng: random.Random, level: int, seed: int):
    nin = rng.randint(2, 2048)
    nout = rng.randint(2, 1024)
    if level == 1:
        val = F(2, nin)
        text = f"Khởi tạo He (Kaiming) cho lớp có $n_{{\\mathrm{{in}}}} = {nin}$ đầu vào dùng ReLU: $\\mathrm{{Var}}(w) = 2/n_{{\\mathrm{{in}}}}$. Tính $\\mathrm{{Var}}(w)$. "
        st = f"\\dfrac{{2}}{{{nin}}} = {ft(val)}"
    elif level == 2:
        val = F(2, nin + nout)
        text = f"Khởi tạo Xavier (Glorot) với phân phối chuẩn: $\\mathrm{{Var}}(w) = \\dfrac{{2}}{{n_{{\\mathrm{{in}}}} + n_{{\\mathrm{{out}}}}}}$, với $n_{{\\mathrm{{in}}}} = {nin}$, $n_{{\\mathrm{{out}}}} = {nout}$. Tính $\\mathrm{{Var}}(w)$. "
        st = f"\\dfrac{{2}}{{{nin} + {nout}}} = {ft(val)}"
    elif level == 3:
        nin = rng.randint(1, 45) ** 2
        kind = rng.choice(["lecun", "he"])
        var = F(1, nin) if kind == "lecun" else F(2, nin)
        if kind == "lecun":
            val = sqrt_q(var)
            text = f"Khởi tạo LeCun: $\\mathrm{{Var}}(w) = 1/n_{{\\mathrm{{in}}}}$ với $n_{{\\mathrm{{in}}}} = {nin}$. Tính **độ lệch chuẩn** $\\sigma = \\sqrt{{\\mathrm{{Var}}(w)}}$. "
            st = f"\\sigma = \\sqrt{{\\dfrac{{1}}{{{nin}}}}} = {ft(val)}"
        else:
            nin = 2 * rng.randint(1, 40) ** 2
            var = F(2, nin)
            val = sqrt_q(var)
            text = f"Khởi tạo He: $\\mathrm{{Var}}(w) = 2/n_{{\\mathrm{{in}}}}$ với $n_{{\\mathrm{{in}}}} = {nin}$. Tính **độ lệch chuẩn** $\\sigma$. "
            st = f"\\sigma = \\sqrt{{\\dfrac{{2}}{{{nin}}}}} = {ft(val)}"
        check(val is not None, "σ hữu tỉ")
    elif level == 4:
        s = rng.choice(SUMS_6)
        nin = rng.randint(1, s - 1) * 1
        nout = s - nin
        a = sqrt_q(F(6, s))
        check(a is not None, "bound hữu tỉ")
        val = a
        text = f"Xavier phân phối đều $U(-a, a)$ có $a = \\sqrt{{\\dfrac{{6}}{{n_{{\\mathrm{{in}}}} + n_{{\\mathrm{{out}}}}}}}}$ (để $\\mathrm{{Var}}(w) = a^2/3 = 2/(n_{{\\mathrm{{in}}}} + n_{{\\mathrm{{out}}}})$). Với $n_{{\\mathrm{{in}}}} = {nin}$, $n_{{\\mathrm{{out}}}} = {nout}$, tính $a$. "
        st = f"a = \\sqrt{{\\dfrac{{6}}{{{s}}}}} = {ft(a)}"
    else:
        n = rng.choice([4, 8, 16, 32, 64, 100, 256])
        c = rng.choice([F(1, 2), F(1), F(2), F(1, 4), F(3, 2), F(3)])
        layers = rng.randint(2, 6)
        # mỗi lớp: Var(z) = n * Var(w) * E[a^2]; ReLU giữ lại một nửa moment bậc hai -> nhân hệ số (n Var(w)) / 2
        factor = c / 2
        val = factor**layers
        # kiểm chứng độc lập: mô phỏng lặp bằng Fraction
        m = F(1)
        for _ in range(layers):
            m = n * (c / n) * m / 2
        check(m == val, "mô phỏng lan truyền")
        text = (f"Mạng gồm ${layers}$ lớp ReLU, mỗi lớp có $n = {n}$ đầu vào và trọng số độc lập $\\mathrm{{Var}}(w) = {ft(c)}/n$. Đầu vào có $\\mathbb{{E}}[a^2] = 1$. "
                "Với ReLU, $\\mathbb{E}[a^2] = \\tfrac{1}{2}\\mathrm{Var}(z)$ và $\\mathrm{Var}(z) = n\\,\\mathrm{Var}(w)\\,\\mathbb{E}[a_{\\mathrm{prev}}^2]$. Tính $\\mathbb{E}[a^2]$ sau lớp cuối. ")
        st = f"\\text{{factor}} = \\dfrac{{n\\,\\mathrm{{Var}}(w)}}{{2}} = {ft(factor)},\\quad \\mathbb{{E}}[a^2] = {fp(factor)}^{{{layers}}} = {ft(val)}"
    return dl(type_="init-var", lesson=L74, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(val),
              hints=["Khởi tạo đúng giữ cho phương sai của kích hoạt không co về $0$ cũng không nổ qua các lớp.", "He: $2/n_{\\mathrm{in}}$ (cho ReLU). Xavier: $2/(n_{\\mathrm{in}} + n_{\\mathrm{out}})$. LeCun: $1/n_{\\mathrm{in}}$.", "Độ lệch chuẩn là căn bậc hai của phương sai."],
              steps=[Step(text="Áp dụng công thức:", latex=st)], verified=["recompute"], tags=["init", f"level{level}"])


# ------------------------------------------------------------------- norm-compute
def _good_vec(rng, n=4):
    for _ in range(5000):
        x = [F(rng.randint(-8, 12)) for _ in range(n)]
        mu = sum(x) / n
        var = sum((v - mu) ** 2 for v in x) / n
        s = sqrt_q(var)
        if s is not None and s > 0 and nice([mu], 4, 20):
            return x, mu, var, s
    raise RuntimeError("không tìm được vector")


def gen_norm_compute(rng: random.Random, level: int, seed: int):
    n = 4
    x, mu, var, s = _good_vec(rng, n)
    kind = rng.choice(["BatchNorm", "LayerNorm"])
    if level == 5:
        kind = "LayerNorm"  # γ, β theo từng đặc trưng chỉ có nghĩa với LayerNorm (BatchNorm: γ, β chung cho cả đặc trưng)
    where = ("giá trị của **một đặc trưng** trên $4$ mẫu trong batch (BatchNorm chuẩn hóa theo batch)" if kind == "BatchNorm"
             else "bốn đặc trưng của **một mẫu** (LayerNorm chuẩn hóa theo đặc trưng)")
    xh = [(v - mu) / s for v in x]
    gamma = [F(rng.choice([1, 2, 3, -1, 1])) for _ in range(n)] if level == 5 else [F(rng.choice([1, 2, 3, -1])) for _ in range(1)] * n
    beta = [F(rng.randint(-3, 3)) for _ in range(n)] if level == 5 else [F(rng.randint(-3, 3))] * n
    y = [g * h + b for g, h, b in zip(gamma, xh, beta)]
    # kiểm chứng độc lập (NumPy, $\epsilon = 0$)
    xn = np.array([float(v) for v in x])
    ref = (xn - xn.mean()) / xn.std()
    check(np.allclose(ref, [float(v) for v in xh]), "chuẩn hóa NumPy")
    base = f"Cho {where}: $\\mathbf{{x}} = {vt(x)}$ (đặt $\\epsilon = 0$; phương sai chia cho $n$). "
    if level == 1:
        val, text, ans = mu, base + f"Tính trung bình $\\mu$. " + NUM_HINT, num_ans(mu)
        steps = [Step(text="Trung bình:", latex=f"\\mu = \\dfrac{{1}}{{4}}\\sum x_i = {ft(mu)}")]
    elif level == 2:
        text, ans = base + "Tính phương sai $\\sigma^2 = \\frac{1}{n}\\sum (x_i - \\mu)^2$. " + NUM_HINT, num_ans(var)
        steps = [Step(text=f"$\\mu = {ft(mu)}$:", latex=f"\\sigma^2 = {ft(var)}")]
    elif level == 3:
        text, ans = base + "Tính vector chuẩn hóa $\\hat{x}_i = (x_i - \\mu)/\\sigma$. " + VEC_HINT, vec_ans(xh)
        steps = [Step(text=f"$\\mu = {ft(mu)}$, $\\sigma^2 = {ft(var)}$, $\\sigma = {ft(s)}$:", latex="\\hat{\\mathbf{x}} = " + vt(xh))]
    elif level == 4:
        text = base + f"Sau chuẩn hóa, áp dụng biến đổi học được $y_i = \\gamma\\hat{{x}}_i + \\beta$ với $\\gamma = {ft(gamma[0])}$, $\\beta = {ft(beta[0])}$. Tính $\\mathbf{{y}}$. " + VEC_HINT
        ans = vec_ans(y)
        steps = [Step(text=f"$\\hat{{\\mathbf{{x}}}} = {vt(xh)}$ (với $\\mu = {ft(mu)}$, $\\sigma = {ft(s)}$):", latex="\\mathbf{y} = " + vt(y))]
    else:
        text = base + f"Sau chuẩn hóa, áp dụng $y_i = \\gamma_i\\hat{{x}}_i + \\beta_i$ với $\\boldsymbol{{\\gamma}} = {vt(gamma)}$, $\\boldsymbol{{\\beta}} = {vt(beta)}$ (tham số riêng cho từng đặc trưng). Tính $\\mathbf{{y}}$. " + VEC_HINT
        ans = vec_ans(y)
        steps = [Step(text=f"$\\mu = {ft(mu)}$, $\\sigma = {ft(s)}$, $\\hat{{\\mathbf{{x}}}} = {vt(xh)}$:", latex="\\mathbf{y} = " + vt(y))]
    return dl(type_="norm-compute", lesson=L74, level=level, seed=seed, prompt=Prompt(text=text), answer=ans,
              hints=["$\\mu = \\frac{1}{n}\\sum x_i$; $\\sigma^2 = \\frac{1}{n}\\sum (x_i - \\mu)^2$ (chia $n$, không phải $n-1$).", "$\\hat{x}_i = (x_i - \\mu)/\\sqrt{\\sigma^2 + \\epsilon}$; ở đây $\\epsilon = 0$.", "Sau chuẩn hóa $\\hat{\\mathbf{x}}$ có trung bình $0$ và phương sai $1$ (kiểm tra nhanh)."],
              steps=steps, verified=["numpy"], tags=[kind, f"level{level}"])


# ------------------------------------------------------------------ dropout-calc
def gen_dropout_calc(rng: random.Random, level: int, seed: int):
    b_ = rng.randint(2, 20)
    p = F(rng.randint(1, b_ - 1), b_)
    scale = 1 / (1 - p)
    if level == 1:
        val = scale
        text = f"Inverted dropout với xác suất bỏ (drop) $p = {ft(p)}$: các đơn vị giữ lại được nhân với $1/(1-p)$. Hệ số nhân là bao nhiêu? "
        steps = [Step(text="$1/(1 - p)$:", latex=f"\\dfrac{{1}}{{1 - {ft(p)}}} = {ft(val)}")]
        ans = num_ans(val)
    elif level == 2:
        x = [F(rng.randint(-6, 6)) for _ in range(4)]
        mask = [rng.randint(0, 1) for _ in range(4)]
        if not any(mask):
            mask[0] = 1
        out = [m * v * scale for m, v in zip(mask, x)]
        text = f"Inverted dropout ($p = {ft(p)}$) lên $\\mathbf{{x}} = {vt(x)}$ với mặt nạ giữ (1 = giữ, 0 = bỏ) $\\mathbf{{m}} = {vt([F(m) for m in mask])}$. Tính đầu ra $\\mathbf{{m}} \\odot \\mathbf{{x}}/(1-p)$. " + VEC_HINT
        steps = [Step(text=f"Hệ số $1/(1-p) = {ft(scale)}$; phần tử bị bỏ thành $0$:", latex="\\mathbf{y} = " + vt(out))]
        ans = vec_ans(out)
    elif level == 3:
        x = F(rng.randint(-9, 9))
        # kiểm chứng độc lập: liệt kê hai khả năng (giữ với xác suất 1-p, bỏ với xác suất p)
        e = (1 - p) * (x * scale) + p * 0
        check(e == x, "kỳ vọng bằng x")
        n = rng.randint(3, 8)
        xs = [F(rng.randint(-5, 5)) for _ in range(n)]
        val = sum(xs)
        text = f"Inverted dropout ($p = {ft(p)}$) được áp dụng độc lập lên từng phần tử của $\\mathbf{{x}} = {vt(xs)}$. Tính **kỳ vọng của tổng** các phần tử đầu ra $\\mathbb{{E}}\\left[\\sum_i y_i\\right]$. "
        steps = [Step(text="Mỗi phần tử: $\\mathbb{E}[y_i] = (1-p)\\cdot\\dfrac{x_i}{1-p} + p\\cdot 0 = x_i$, nên kỳ vọng của tổng bằng tổng của $\\mathbf{x}$:", latex=f"\\sum x_i = {ft(val)}")]
        ans = num_ans(val)
    elif level == 4:
        x = F(rng.randint(-6, 6))
        while x == 0:
            x = F(rng.randint(-6, 6))
        var = x * x * p / (1 - p)
        # kiểm chứng: liệt kê hai khả năng
        m1 = (1 - p) * (x * scale) + p * 0
        v2 = (1 - p) * (x * scale - m1) ** 2 + p * (0 - m1) ** 2
        check(v2 == var, "phương sai dropout")
        text = f"Inverted dropout ($p = {ft(p)}$) lên một đơn vị có giá trị $x = {ft(x)}$: đầu ra là $x/(1-p)$ với xác suất $1-p$ và $0$ với xác suất $p$. Tính phương sai của đầu ra. " + NUM_HINT
        steps = [Step(text="Kỳ vọng là $x$; phương sai:", latex=f"\\mathrm{{Var}} = x^2\\dfrac{{p}}{{1-p}} = {ft(var)}")]
        ans = num_ans(var)
    else:
        n = rng.choice([4, 8, 10, 16, 20, 50, 100])
        k = rng.randint(2, 5)
        keep = (1 - p) * n
        if keep.denominator != 1:
            n = int(n * p.denominator)
            keep = (1 - p) * n
        val = keep
        text = f"Một lớp ẩn có ${n}$ đơn vị với dropout $p = {ft(p)}$ khi huấn luyện (mỗi đơn vị bị bỏ độc lập). Số đơn vị được giữ lại **kỳ vọng** là bao nhiêu? "
        steps = [Step(text="Mỗi đơn vị được giữ với xác suất $1-p$:", latex=f"n(1-p) = {n}\\cdot {ft(1 - p)} = {ft(val)}")]
        ans = num_ans(val)
    return dl(type_="dropout-calc", lesson=L75, level=level, seed=seed, prompt=Prompt(text=text + (VEC_HINT if level == 2 else NUM_HINT if level != 1 else NUM_HINT)), answer=ans,
              hints=["Inverted dropout nhân phần tử được giữ với $1/(1-p)$ ngay khi huấn luyện để kỳ vọng không đổi; lúc suy luận không dropout, không cần thêm hệ số.", "$p$ là xác suất **bỏ**; xác suất giữ là $1 - p$.", "Kỳ vọng: $(1-p)\\cdot\\dfrac{x}{1-p} + p\\cdot 0 = x$."],
              steps=steps, verified=["enumerate-two-outcomes"], tags=["dropout", f"level{level}"])


# --------------------------------------------------------------------- resid-grad
def gen_resid_grad(rng: random.Random, level: int, seed: int):
    if level == 1:
        a, x = F(pick(rng, -3, 3)), F(rng.randint(-4, 4))
        val = 1 + 2 * a * x
        text = f"Khối residual $y = x + F(x)$ với $F(x) = {ft(a)}x^2$. Tính $\\dfrac{{dy}}{{dx}}$ tại $x = {ft(x)}$. "
        steps = [Step(text="$\\dfrac{dy}{dx} = 1 + F'(x) = 1 + 2ax$:", latex=f"1 + 2\\cdot {fp(a)}\\cdot {fp(x)} = {ft(val)}")]
        f = lambda t: float(t) + float(a) * float(t) ** 2
        hh = 1e-6
        check(abs((f(float(x) + hh) - f(float(x) - hh)) / (2 * hh) - float(val)) < 1e-5, "sai phân hữu hạn")
    elif level == 2:
        w, x = F(pick(rng, -3, 3)), F(pick(rng, -4, 4))
        z = w * x
        val = 1 + (w if z > 0 else 0)
        text = f"Khối residual $y = x + \\mathrm{{ReLU}}(wx)$ với $w = {ft(w)}$. Tính $dy/dx$ tại $x = {ft(x)}$ (quy ước $\\mathrm{{ReLU}}'(0) = 0$). "
        steps = [Step(text=f"$wx = {ft(z)}$, nên $\\mathrm{{ReLU}}'(wx) = {1 if z > 0 else 0}$ và $\\dfrac{{dy}}{{dx}} = 1 + \\mathrm{{ReLU}}'(wx)\\,w$:", latex=f"= {ft(val)}")]
    elif level == 3:
        w1, w2 = F(pick(rng, -3, 3), rng.choice([1, 2, 3])), F(pick(rng, -3, 3), rng.choice([1, 2, 3]))
        while w1 == -1 or w2 == -1:  # 1 + w = 0 làm gradient bằng 0, trái với ý nghĩa của residual
            w1, w2 = F(pick(rng, -3, 3), rng.choice([1, 2, 3])), F(pick(rng, -3, 3), rng.choice([1, 2, 3]))
        val = (1 + w1) * (1 + w2)
        text = f"Hai khối residual nối tiếp với $F_i(x) = w_ix$: $h = x + w_1x$, $y = h + w_2h$, $w_1 = {ft(w1)}$, $w_2 = {ft(w2)}$. Tính $dy/dx$. "
        steps = [Step(text="Chain rule: nhân đạo hàm của từng khối $(1 + w_i)$:", latex=f"(1 + {fp(w1)})(1 + {fp(w2)}) = {ft(val)}")]
        check(val == (1 + w1) * (1 + w2) and (1 + w1 + w2 + w1 * w2) == val, "khai triển")
    elif level == 4:
        w = F(pick(rng, -3, 3), rng.choice([2, 4, 5]))
        while w == -1:
            w = F(pick(rng, -3, 3), rng.choice([2, 4, 5]))
        nl = rng.randint(2, 5)
        val = (1 + w) ** nl
        text = f"Mạng gồm ${nl}$ khối residual giống nhau $y = x + wx$ nối tiếp với $w = {ft(w)}$. Tính $dy_{{\\mathrm{{out}}}}/dx_{{\\mathrm{{in}}}}$. "
        steps = [Step(text="Mỗi khối nhân gradient với $1 + w$:", latex=f"(1 + {fp(w)})^{{{nl}}} = {ft(val)}")]
    else:
        w = F(rng.randint(1, 9), rng.choice([10, 12, 16, 20, 25]))
        nl = rng.randint(3, 6)
        val = (1 + w) ** nl - w**nl
        text = (f"So sánh gradient qua ${nl}$ khối: mạng thường $y = wx$ mỗi khối (gradient $w^{{{nl}}}$) và mạng residual $y = x + wx$ mỗi khối (gradient $(1+w)^{{{nl}}}$), với $w = {ft(w)}$. "
                "Tính hiệu $(1 + w)^{n} - w^{n}$. ")
        steps = [Step(text="Mạng thường co gradient rất nhanh, mạng residual giữ gradient gần $1$:", latex=f"(1 + {fp(w)})^{{{nl}}} - {fp(w)}^{{{nl}}} = {ft((1 + w) ** nl)} - {ft(w ** nl)} = {ft(val)}")]
    return dl(type_="resid-grad", lesson=L75, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(val),
              hints=["Residual: $y = x + F(x)$ nên $dy/dx = 1 + F'(x)$: luôn có \"đường tắt\" với hệ số $1$.", "Nối tiếp nhiều khối: nhân các đạo hàm của từng khối (chain rule).", "Số hạng $1$ giữ cho gradient không biến mất khi mạng sâu."],
              steps=steps, verified=["recompute"], tags=["residual", f"level{level}"])


# ------------------------------------------------------------------- embed-lookup
def gen_embed_lookup(rng: random.Random, level: int, seed: int):
    v, d = rng.choice([(4, 3), (5, 3), (4, 4), (5, 2)])
    e = rand_mat(rng, v, d, -3, 3)
    toks = [rng.randrange(v) for _ in range(rng.randint(2, 3))]
    etx = "\\mathbf{E} = " + mat_tex(e) + f"\\quad({v}\\times {d};\\ \\text{{row }} i \\text{{ is the embedding of token }} i)"
    etx = "\\mathbf{E} = " + mat_tex(e)
    note = f"Bảng embedding $\\mathbf{{E}} \\in \\R^{{{v}\\times {d}}}$ (hàng $i$ là vector của token $i$, đánh số từ $1$). "
    if level == 1:
        t = toks[0]
        ans, steps = vec_ans(e[t]), [Step(text=f"Tra hàng ${t + 1}$ (tương đương nhân vector one-hot với $\\mathbf{{E}}$):", latex=vt(e[t]))]
        text = note + f"Tìm embedding của token ${t + 1}$. " + VEC_HINT
    elif level == 2:
        rows = [e[t] for t in toks]
        ans = mat_ans(rows)
        steps = [Step(text="Tra từng token rồi xếp thành các hàng:", latex=mat_tex(rows))]
        text = note + f"Chuỗi token là $({', '.join(str(t + 1) for t in toks)})$. Tìm ma trận embedding của chuỗi (mỗi hàng là một token). Nhập từng phần tử."
    elif level == 3:
        rows = [e[t] for t in toks]
        mean = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
        ans = vec_ans(mean)
        steps = [Step(text="Embedding các token, rồi trung bình theo token (mean pooling):", latex="\\bar{\\mathbf{e}} = " + vt(mean))]
        text = note + f"Chuỗi token $({', '.join(str(t + 1) for t in toks)})$. Tính vector trung bình (mean pooling) của các embedding. " + VEC_HINT
    elif level == 4:
        pos = rand_mat(rng, len(toks), d, -2, 2)
        out = [[e[t][j] + pos[i][j] for j in range(d)] for i, t in enumerate(toks)]
        ans = mat_ans(out)
        steps = [Step(text="Cộng embedding token với embedding vị trí (cùng hàng):", latex=mat_tex(out))]
        text = note + f"Chuỗi token $({', '.join(str(t + 1) for t in toks)})$ và embedding vị trí $\\mathbf{{P}} = {mat_tex(pos)}$ (hàng $i$ cho vị trí $i$). Tính đầu vào của Transformer $\\mathbf{{E}}_{{\\mathrm{{tok}}}} + \\mathbf{{P}}$. Nhập từng phần tử."
    else:
        h = rand_vec(rng, d, -2, 2)
        logits = mv(e, h)
        ans = vec_ans(logits)
        # kiểm chứng NumPy
        check(np.allclose(np.array([[float(x) for x in r] for r in e]) @ np.array([float(x) for x in h]), [float(x) for x in logits]), "NumPy")
        steps = [Step(text="Weight tying: logit của từ vựng $= \\mathbf{E}\\mathbf{h}$ (cùng ma trận embedding):", latex="\\mathbf{z} = " + vt(logits))]
        text = note + f"Mô hình ngôn ngữ dùng weight tying: logit trên toàn từ vựng là $\\mathbf{{z}} = \\mathbf{{E}}\\mathbf{{h}}$. Với $\\mathbf{{h}} = {vt(h)}$, tính $\\mathbf{{z}}$. " + VEC_HINT
    return dl(type_="embed-lookup", lesson=L75, level=level, seed=seed, prompt=Prompt(text=text, display=["\\mathbf{E} = " + mat_tex(e)]), answer=ans,
              hints=["Embedding là tra bảng: lấy hàng của token (tương đương $\\mathbf{e}_i\\T\\mathbf{E}$ với $\\mathbf{e}_i$ là vector one-hot).", "Token đánh số từ $1$ trong đề (hàng $1$ là hàng đầu).", "Embedding vị trí được cộng (không nối) vào embedding token."],
              steps=steps, verified=["recompute"], tags=["embedding", f"level{level}"])


SPECS = [
    TopicSpec("init-var", L74, (1, 2, 3, 4, 5), gen_init_var),
    TopicSpec("norm-compute", L74, (1, 2, 3, 4, 5), gen_norm_compute),
    TopicSpec("dropout-calc", L75, (1, 2, 3, 4, 5), gen_dropout_calc),
    TopicSpec("resid-grad", L75, (1, 2, 3, 4, 5), gen_resid_grad),
    TopicSpec("embed-lookup", L75, (1, 2, 3, 4, 5), gen_embed_lookup),
]
