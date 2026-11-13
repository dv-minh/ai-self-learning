"""Bài 7.1 Perceptron và MLP: mlp-forward, mlp-params. Bài 7.2: activation-eval, softmax-ln2. Bài 7.3: backprop-net."""
from __future__ import annotations

import random
from math import log2

import numpy as np

from ._d import (F, L, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, choice_exercise, dl, fp, ft, mat_ans, mat_tex, mm, mv, nice, num_ans, pick, rand_mat, rand_vec,
                 relu, vec_ans, vt)

L71, L72, L73 = "7.1", "7.2", "7.3"


def _np(m):
    return np.array([[float(v) for v in r] for r in m]) if isinstance(m[0], list) else np.array([float(v) for v in m])


# ------------------------------------------------------------------ mlp-forward
def gen_mlp_forward(rng: random.Random, level: int, seed: int):
    sizes = {1: (2, 1), 2: (3, 1), 3: (2, 2), 4: (2, 2, 1), 5: (3, 3, 2)}[level]
    for _ in range(500):
        ws = [rand_mat(rng, sizes[i + 1], sizes[i]) for i in range(len(sizes) - 1)]
        bs = [rand_vec(rng, sizes[i + 1], -2, 2) if level in (1, 2, 5) else [F(0)] * sizes[i + 1] for i in range(len(sizes) - 1)]
        x = rand_vec(rng, sizes[0], -3, 3)
        acts, a = [], x
        zs = []
        for i, (w, b) in enumerate(zip(ws, bs)):
            z = [p + q for p, q in zip(mv(w, a), b)]
            zs.append(z)
            last = i == len(ws) - 1
            if level == 1:
                a = [F(1) if v > 0 else F(0) for v in z]  # perceptron bước
            elif last and level in (4, 5):
                a = z
            else:
                a = [relu(v) for v in z]
            acts.append(a)
        hidden_zs = zs[0] if len(ws) > 1 or level >= 2 else zs[0]
        if level >= 2 and any(v == 0 for v in zs[0]):
            continue
        if level >= 3 and not (any(v <= 0 for v in zs[0]) and any(v > 0 for v in zs[0])):
            continue
        break
    # kiểm chứng độc lập (NumPy)
    an = _np(x)
    for i, (w, b) in enumerate(zip(ws, bs)):
        zn = _np(w) @ an + _np(b)
        last = i == len(ws) - 1
        an = (zn > 0).astype(float) if level == 1 else (zn if last and level in (4, 5) else np.maximum(zn, 0))
    check(np.allclose(an, [float(v) for v in acts[-1]]), "NumPy forward ≠ Fraction")
    kind = {1: "perceptron bước ($\\mathbb{1}[z > 0]$)", 2: "nơ-ron ReLU", 3: "một lớp ReLU", 4: "mạng $2 \\to 2 \\to 1$ với ReLU ở lớp ẩn", 5: "mạng $3 \\to 3 \\to 2$ (ReLU ở lớp ẩn, tuyến tính ở đầu ra)"}[level]
    disp = []
    for i, (w, b) in enumerate(zip(ws, bs)):
        disp.append(f"\\mathbf{{W}}_{i + 1} = {mat_tex(w)}" + (f",\\quad \\mathbf{{b}}_{i + 1} = {vt(b)}" if any(v != 0 for v in b) else ""))
    steps = []
    a = x
    for i, z in enumerate(zs):
        steps.append(Step(text=f"Lớp {i + 1}: $\\mathbf{{z}}_{i + 1} = \\mathbf{{W}}_{i + 1}\\mathbf{{a}}_{i} + \\mathbf{{b}}_{i + 1}$, rồi áp dụng hàm kích hoạt.",
                          latex=f"\\mathbf{{z}}_{i + 1} = {vt(z)},\\quad \\mathbf{{a}}_{i + 1} = {vt(acts[i])}"))
    out = acts[-1]
    return dl(type_="mlp-forward", lesson=L71, level=level, seed=seed,
              prompt=Prompt(text=f"Tính đầu ra của {kind} với đầu vào $\\mathbf{{x}} = {vt(x)}$ và các tham số dưới đây. " + (NUM_HINT if len(out) == 1 else VEC_HINT), display=disp),
              answer=num_ans(out[0]) if len(out) == 1 else vec_ans(out),
              hints=["Mỗi lớp: $\\mathbf{z} = \\mathbf{W}\\mathbf{a} + \\mathbf{b}$ (nhân ma trận–vector), rồi áp dụng hàm kích hoạt từng phần tử.", "ReLU: $\\max(0, z)$. Perceptron bước: $1$ nếu $z > 0$, ngược lại $0$.",
                     "Đầu ra của lớp trước là đầu vào của lớp sau."], steps=steps, verified=["numpy-forward"], tags=["mlp", f"{len(ws)}layer"])


# ------------------------------------------------------------------- mlp-params
def gen_mlp_params(rng: random.Random, level: int, seed: int):
    for _ in range(100):
        nl = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3}[level]
        sizes = [rng.choice([2, 3, 4, 5, 8, 10, 16, 32, 64, 100, 128, 256, 784]) for _ in range(nl + 1)]
        if len(set(sizes)) > 1:
            break
    bias = rng.random() < 0.8 or level >= 3
    bn = level == 5
    cnt = 0
    parts = []
    for i in range(nl):
        w = sizes[i] * sizes[i + 1]
        b = sizes[i + 1] if bias else 0
        cnt += w + b
        parts.append(f"{sizes[i]}\\cdot {sizes[i + 1]}" + (f" + {sizes[i + 1]}" if bias else ""))
        if bn and i < nl - 1:
            cnt += 2 * sizes[i + 1]
            parts[-1] += f" + 2\\cdot {sizes[i + 1]}"
    # kiểm chứng độc lập: đếm bằng công thức đóng
    closed = sum(sizes[i] * sizes[i + 1] for i in range(nl)) + (sum(sizes[1:]) if bias else 0) + (2 * sum(sizes[1:-1]) if bn else 0)
    check(closed == cnt, "đếm tham số")
    macs = sum(sizes[i] * sizes[i + 1] for i in range(nl))
    ask_macs = level == 4
    val = macs if ask_macs else cnt
    arch = " \\to ".join(str(s) for s in sizes)
    extra = (" có hệ số chặn $\\mathbf{b}$ ở mọi lớp" if bias else " **không** có hệ số chặn") + (", và BatchNorm sau mỗi lớp ẩn (mỗi đặc trưng có $\\gamma, \\beta$ học được; bỏ qua running mean/var)" if bn else "")
    q = ("số phép nhân–cộng (multiply–accumulate, MAC) của một lần forward cho **một mẫu** (chỉ tính phép nhân ma trận–vector)" if ask_macs else "tổng số tham số học được")
    steps = [Step(text="Lớp $i$ có ma trận $n_{i+1}\\times n_i$ và (nếu có) vector chặn $n_{i+1}$:", latex=("\\text{MAC} = " if ask_macs else "\\text{params} = ") + (" + ".join(parts) if not ask_macs else " + ".join(f"{sizes[i]}\\cdot {sizes[i + 1]}" for i in range(nl))) + f" = {val}")]
    return dl(type_="mlp-params", lesson=L71, level=level, seed=seed,
              prompt=Prompt(text=f"MLP có các lớp kích thước ${arch}$ (đầu vào, các lớp ẩn, đầu ra),{extra}. Tính {q}. " + NUM_HINT), answer=num_ans(F(val)),
              hints=["Lớp kết nối đầy đủ từ $n_i$ đến $n_{i+1}$ có $n_in_{i+1}$ trọng số và $n_{i+1}$ hệ số chặn.", "Cộng theo từng lớp; đầu vào không có tham số.", "MAC của một lớp bằng đúng số trọng số $n_in_{i+1}$." if ask_macs else "BatchNorm có $2$ tham số ($\\gamma$, $\\beta$) cho mỗi đặc trưng."],
              steps=steps, verified=["closed-form-recount"], tags=["params", f"{nl}layer"])


# ---------------------------------------------------------------- activation-eval
def gen_activation_eval(rng: random.Random, level: int, seed: int):
    xs = [F(rng.randint(-40, 40), rng.choice([1, 2, 4, 5])) for _ in range(1 if level != 5 else 4)]
    alpha = F(rng.choice([1, 1, 1, 2, 3]), rng.choice([10, 5, 4]))
    if level == 1:
        x = xs[0]
        val, text, st = relu(x), f"Tính $\\mathrm{{ReLU}}({ft(x)})$ với $\\mathrm{{ReLU}}(z) = \\max(0, z)$. ", f"\\max(0, {ft(x)}) = {ft(relu(x))}"
    elif level == 2:
        x = xs[0]
        f = x if x > 0 else alpha * x
        val, text, st = f, f"Tính $\\mathrm{{LeakyReLU}}({ft(x)})$ với $\\mathrm{{LeakyReLU}}(z) = z$ nếu $z > 0$, ngược lại $\\alpha z$; $\\alpha = {ft(alpha)}$. ", ft(f)
    elif level == 3:
        x = xs[0]
        kind = rng.choice(["relu", "leaky"])
        d = F(1) if x > 0 else (F(0) if kind == "relu" else alpha)
        val = d
        text = (f"Tính đạo hàm của $\\mathrm{{ReLU}}$ tại $z = {ft(x)}$ (quy ước $\\mathrm{{ReLU}}'(0) = 0$). " if kind == "relu"
                else f"Tính đạo hàm của $\\mathrm{{LeakyReLU}}$ ($\\alpha = {ft(alpha)}$) tại $z = {ft(x)}$ (quy ước đạo hàm tại $0$ bằng $\\alpha$). ")
        st = ft(d)
    elif level == 4:
        x = xs[0]
        kind = rng.choice(["relu6", "hardtanh"])
        f = min(max(x, F(0)), F(6)) if kind == "relu6" else min(max(x, F(-1)), F(1))
        val = f
        text = (f"Tính $\\mathrm{{ReLU6}}({ft(x)})$ với $\\mathrm{{ReLU6}}(z) = \\min(\\max(0, z), 6)$. " if kind == "relu6" else f"Tính $\\mathrm{{HardTanh}}({ft(x)})$ với $\\mathrm{{HardTanh}}(z) = \\min(\\max(-1, z), 1)$. ")
        st = ft(f)
    else:
        out = [x if x > 0 else alpha * x for x in xs]
        steps = [Step(text="Áp dụng LeakyReLU từng phần tử:", latex="\\mathbf{a} = " + vt(out))]
        return dl(type_="activation-eval", lesson=L72, level=level, seed=seed,
                  prompt=Prompt(text=f"Áp dụng $\\mathrm{{LeakyReLU}}$ ($\\alpha = {ft(alpha)}$) từng phần tử cho $\\mathbf{{z}} = {vt(xs)}$. " + VEC_HINT), answer=vec_ans(out),
                  hints=["Phần tử dương giữ nguyên; phần tử âm nhân với $\\alpha$."], steps=steps, verified=["recompute"], tags=["vector"])
    return dl(type_="activation-eval", lesson=L72, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(val),
              hints=["Xét dấu của $z$ trước.", "LeakyReLU giữ một độ dốc nhỏ $\\alpha$ ở phía âm để gradient không bằng $0$."], steps=[Step(text="Tính trực tiếp:", latex=st)], verified=["recompute"], tags=[f"level{level}"])


# ------------------------------------------------------------------- softmax-ln2
def _softmax_set(rng, c, need_pow2):
    for _ in range(2000):
        ks = [rng.randint(0, 7) for _ in range(c)]
        s = sum(2**k for k in ks)
        if need_pow2 and s & (s - 1):
            continue
        if len(set(ks)) > 1:
            return ks
    return [1, 0] + [0] * (c - 2)


def gen_softmax_ln2(rng: random.Random, level: int, seed: int):
    c = {1: 2, 2: 3, 3: 3, 4: 3, 5: 3}[level]
    need = level == 4
    ks = _softmax_set(rng, c, need)
    es = [F(2) ** k for k in ks]
    tot = sum(es)
    p = [e / tot for e in es]
    # kiểm chứng độc lập: softmax bằng số thực với z = k ln 2
    z = np.array(ks) * np.log(2.0)
    sm = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    check(np.allclose(sm, [float(v) for v in p]), "softmax NumPy ≠ phân số")
    t = rng.randrange(c)
    zt = "(" + ", ".join(f"{k}\\ln 2" if k else "0" for k in ks) + ")\\T"
    steps = [Step(text="Với $z_i = k_i\\ln 2$ thì $e^{z_i} = 2^{k_i}$ (số hữu tỉ):", latex="e^{\\mathbf{z}} = " + vt(es) + f",\\quad \\sum = {ft(tot)}")]
    if level in (1, 2, 3):
        i = rng.randrange(c)
        val = p[i]
        steps.append(Step(text=f"$p_{i + 1} = e^{{z_{i + 1}}} / \\sum_j e^{{z_j}}$:", latex=f"p_{i + 1} = \\dfrac{{{ft(es[i])}}}{{{ft(tot)}}} = {ft(val)}"))
        text = f"Softmax của logit $\\mathbf{{z}} = {zt}$. Tính xác suất $p_{i + 1}$. "
        ans = num_ans(val)
    elif level == 4:
        bits = log2(float(tot)) - ks[t]
        check(abs(bits - round(bits)) < 1e-12, "bits nguyên")
        val = F(round(bits))
        steps.append(Step(text=f"Cross-entropy cho lớp thật ${t + 1}$: $\\ell = -\\ln p_{t + 1} = \\ln\\sum_j 2^{{k_j}} - k_{t + 1}\\ln 2$; chia cho $\\ln 2$ (đơn vị bit):", latex=f"\\dfrac{{\\ell}}{{\\ln 2}} = \\log_2({ft(tot)}) - {ks[t]} = {val}"))
        text = f"Softmax của logit $\\mathbf{{z}} = {zt}$, nhãn thật là lớp ${t + 1}$. Tính cross-entropy $\\ell = -\\ln p_{t + 1}$ **chia cho $\\ln 2$** (tức tính bằng bit). "
        ans = num_ans(val)
    else:
        g = [pi - (1 if i == t else 0) for i, pi in enumerate(p)]
        steps.append(Step(text="Gradient của cross-entropy theo logit là $\\mathbf{p} - \\mathbf{y}$:", latex="\\mathbf{p} = " + vt(p) + ",\\quad \\mathbf{p} - \\mathbf{y} = " + vt(g)))
        text = f"Softmax của logit $\\mathbf{{z}} = {zt}$, nhãn thật là lớp ${t + 1}$. Tính gradient $\\partial\\ell/\\partial\\mathbf{{z}}$ của cross-entropy. "
        ans = vec_ans(g)
    return dl(type_="softmax-ln2", lesson=L72, level=level, seed=seed, prompt=Prompt(text=text + (NUM_HINT if level < 5 else VEC_HINT)), answer=ans,
              hints=["$\\mathrm{softmax}(\\mathbf{z})_i = e^{z_i}/\\sum_j e^{z_j}$; vì $z_i = k_i\\ln 2$ nên $e^{z_i} = 2^{k_i}$.", "Các xác suất cộng bằng $1$.", "Cross-entropy $= -\\ln p_t$; gradient theo logit $= \\mathbf{p} - \\mathbf{y}$ (one-hot $\\mathbf{y}$)."],
              steps=steps, verified=["numpy-softmax"], tags=["softmax", f"{c}class"])


# -------------------------------------------------------------------- backprop-net
def gen_backprop_net(rng: random.Random, level: int, seed: int):
    if level <= 3:
        for _ in range(500):
            x, y = F(pick(rng, -3, 3)), F(rng.randint(-4, 4))
            w, b, w2 = F(pick(rng, -3, 3)), F(rng.randint(-2, 2)), F(pick(rng, -3, 3))
            if level == 1:
                z = w * x + b
                loss = (z - y) ** 2 / 2
                g = (z - y) * x
                break
            if level == 2:
                z = w * x + b
                if z == 0:
                    continue
                a = relu(z)
                g = (a - y) * (1 if z > 0 else 0) * x
                break
            h0 = w * x
            if h0 == 0:
                continue
            h = relu(h0)
            yh = w2 * h
            g = (yh - y) * w2 * (1 if h0 > 0 else 0) * x
            break
        # kiểm chứng độc lập: sai phân hữu hạn
        def fwd(wv):
            if level == 1:
                return (float(wv) * float(x) + float(b) - float(y)) ** 2 / 2
            if level == 2:
                return (max(0.0, float(wv) * float(x) + float(b)) - float(y)) ** 2 / 2
            return (float(w2) * max(0.0, float(wv) * float(x)) - float(y)) ** 2 / 2
        hh = 1e-6
        check(abs((fwd(float(w) + hh) - fwd(float(w) - hh)) / (2 * hh) - float(g)) < 1e-5, "sai phân hữu hạn")
        if level == 1:
            text = f"Nơ-ron tuyến tính $z = wx + b$ với $x = {ft(x)}$, $b = {ft(b)}$, $w = {ft(w)}$, mất mát $L = \\frac{{1}}{{2}}(z - y)^2$ với $y = {ft(y)}$. Tính $\\partial L/\\partial w$. "
            steps = [Step(text="Chain rule: $\\dfrac{\\partial L}{\\partial w} = \\dfrac{\\partial L}{\\partial z}\\cdot\\dfrac{\\partial z}{\\partial w} = (z - y)\\,x$.", latex=f"z = {ft(z)},\\quad \\dfrac{{\\partial L}}{{\\partial w}} = ({ft(z)} - {fp(y)})\\cdot {fp(x)} = {ft(g)}")]
        elif level == 2:
            text = f"$z = wx + b$, $a = \\mathrm{{ReLU}}(z)$, $L = \\frac{{1}}{{2}}(a - y)^2$ với $x = {ft(x)}$, $b = {ft(b)}$, $w = {ft(w)}$, $y = {ft(y)}$. Tính $\\partial L/\\partial w$ (quy ước $\\mathrm{{ReLU}}'(0) = 0$). "
            steps = [Step(text="Forward: $z$ và $a$. Backward: $\\dfrac{\\partial L}{\\partial w} = (a - y)\\cdot\\mathrm{ReLU}'(z)\\cdot x$.", latex=f"z = {ft(z)},\\ a = {ft(a)},\\ \\mathrm{{ReLU}}'(z) = {1 if z > 0 else 0},\\ \\dfrac{{\\partial L}}{{\\partial w}} = {ft(g)}")]
        else:
            text = f"Mạng $h = \\mathrm{{ReLU}}(w_1x)$, $\\hat{{y}} = w_2h$, $L = \\frac{{1}}{{2}}(\\hat{{y}} - y)^2$ với $x = {ft(x)}$, $y = {ft(y)}$, $w_1 = {ft(w)}$, $w_2 = {ft(w2)}$. Tính $\\partial L/\\partial w_1$. "
            steps = [Step(text="Forward rồi backward theo chuỗi $L \\leftarrow \\hat{y} \\leftarrow h \\leftarrow w_1$:", latex=f"h = {ft(h)},\\ \\hat{{y}} = {ft(yh)};\\quad \\dfrac{{\\partial L}}{{\\partial w_1}} = (\\hat{{y}} - y)\\,w_2\\,\\mathrm{{ReLU}}'(w_1x)\\,x = {ft(g)}")]
        ans = num_ans(g)
        hints = ["Tính forward trước và lưu các giá trị trung gian.", "Backward: nhân các đạo hàm cục bộ dọc theo đường đi từ $L$ về tham số.", "ReLU' bằng $1$ nếu đầu vào $> 0$, ngược lại $0$."]
        ver = ["finite-difference"]
    else:
        nb = 2 if level == 5 else 1
        for _ in range(1000):
            w1 = rand_mat(rng, 2, 2)
            w2 = rand_vec(rng, 2, -2, 2)
            xs = [rand_vec(rng, 2, -3, 3) for _ in range(nb)]
            ys = [F(rng.randint(-5, 5)) for _ in range(nb)]
            fw = []
            ok = True
            for x in xs:
                z = mv(w1, x)
                if any(v == 0 for v in z):
                    ok = False
                h = [relu(v) for v in z]
                yh = sum(a * b for a, b in zip(w2, h))
                fw.append((z, h, yh))
            if not ok or not any(any(v > 0 for v in f[0]) for f in fw) or all(v == 0 for v in w2):
                continue
            g = [[sum((yh - y) * w2[i] * (1 if z[i] > 0 else 0) * x[j] for (z, h, yh), x, y in zip(fw, xs, ys)) / nb for j in range(2)] for i in range(2)]
            if any(v != 0 for r in g for v in r):
                break
        # kiểm chứng: sai phân hữu hạn theo từng phần tử W1
        def loss(wm):
            tot = 0.0
            for x, y in zip(xs, ys):
                zz = np.array(wm) @ np.array([float(v) for v in x])
                tot += (np.array([float(v) for v in w2]) @ np.maximum(zz, 0) - float(y)) ** 2 / 2
            return tot / nb
        base = [[float(v) for v in r] for r in w1]
        for i in range(2):
            for j in range(2):
                hh = 1e-6
                a_, b_ = [r[:] for r in base], [r[:] for r in base]
                a_[i][j] += hh
                b_[i][j] -= hh
                check(abs((loss(a_) - loss(b_)) / (2 * hh) - float(g[i][j])) < 1e-5, "sai phân hữu hạn W1")
        smp = "; ".join(f"\\mathbf{{x}} = {vt(x)},\\ y = {ft(y)}" for x, y in zip(xs, ys))
        text = (f"Mạng $\\mathbf{{h}} = \\mathrm{{ReLU}}(\\mathbf{{W}}_1\\mathbf{{x}})$, $\\hat{{y}} = \\mathbf{{w}}_2\\T\\mathbf{{h}}$ (không có hệ số chặn), mất mát " + ("trung bình trên hai mẫu của " if nb == 2 else "") +
                f"$\\frac{{1}}{{2}}(\\hat{{y}} - y)^2$. Với $\\mathbf{{W}}_1 = {mat_tex(w1)}$, $\\mathbf{{w}}_2 = {vt(w2)}$ và mẫu: ${smp}$. Tính gradient $\\partial L/\\partial\\mathbf{{W}}_1$ (ma trận $2\\times 2$; quy ước $\\mathrm{{ReLU}}'(0) = 0$). Nhập từng phần tử.")
        z, h, yh = fw[0]
        steps = [Step(text="Với mỗi mẫu: forward $\\mathbf{z} = \\mathbf{W}_1\\mathbf{x}$, $\\mathbf{h}$, $\\hat{y}$; sai số $\\delta = \\hat{y} - y$.", latex="; ".join(f"\\hat{{y}} = {ft(f[2])}" for f in fw)),
                 Step(text="Backward: $\\dfrac{\\partial L}{\\partial\\mathbf{z}} = \\delta\\,(\\mathbf{w}_2 \\odot \\mathrm{ReLU}'(\\mathbf{z}))$ và $\\dfrac{\\partial L}{\\partial\\mathbf{W}_1} = \\dfrac{\\partial L}{\\partial\\mathbf{z}}\\,\\mathbf{x}\\T$ (trung bình trên các mẫu):", latex="\\dfrac{\\partial L}{\\partial\\mathbf{W}_1} = " + mat_tex(g))]
        ans = mat_ans(g)
        hints = ["Backward cho lớp ẩn: $\\partial L/\\partial \\mathbf{z} = \\delta\\,\\mathbf{w}_2 \\odot \\mathbb{1}[\\mathbf{z} > 0]$ (nhân từng phần tử).", "Gradient trọng số $= (\\partial L/\\partial\\mathbf{z})\\,\\mathbf{x}\\T$ (tích ngoài).", "Với hai mẫu, lấy trung bình gradient của hai mẫu." if nb == 2 else "Nơ-ron ReLU không hoạt động ($z \\le 0$) cho gradient $0$ ở hàng tương ứng."]
        ver = ["finite-difference"]
    return dl(type_="backprop-net", lesson=L73, level=level, seed=seed, prompt=Prompt(text=text + (NUM_HINT if level <= 3 else "")), answer=ans, hints=hints, steps=steps, verified=ver, tags=["backprop", f"level{level}"])


SPECS = [
    TopicSpec("mlp-forward", L71, (1, 2, 3, 4, 5), gen_mlp_forward),
    TopicSpec("mlp-params", L71, (1, 2, 3, 4, 5), gen_mlp_params),
    TopicSpec("activation-eval", L72, (1, 2, 3, 4, 5), gen_activation_eval),
    TopicSpec("softmax-ln2", L72, (1, 2, 3, 4, 5), gen_softmax_ln2),
    TopicSpec("backprop-net", L73, (1, 2, 3, 4, 5), gen_backprop_net),
]
