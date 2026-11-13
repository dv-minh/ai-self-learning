"""Bài 7.6 Attention và Transformer: attn-scores, attn-output, transformer-params. Bài 7.7 Fine-tuning, LoRA, PyTorch: lora-calc, torch-shape."""
from __future__ import annotations

import random

import numpy as np

from ._d import (F, L, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, choice_exercise, dl, fp, ft, mat_ans, mat_tex, mm, mv, nice, num_ans, pick, rand_mat, rand_vec,
                 vec_ans, vt)

L76, L77 = "7.6", "7.7"


def _T(m):
    return [list(r) for r in zip(*m)]


def _arr(m):
    return np.array([[float(v) for v in r] for r in m])


# ------------------------------------------------------------------ attn-scores
def gen_attn_scores(rng: random.Random, level: int, seed: int):
    d = 4
    if level == 5:  # nhiều đầu: d_model = 8, 2 đầu, d_k = 4
        n, heads = 2, 2
        q, k = rand_mat(rng, n, 8, -2, 2), rand_mat(rng, n, 8, -2, 2)
        h = rng.randrange(heads)
        qh = [r[h * 4:(h + 1) * 4] for r in q]
        kh = [r[h * 4:(h + 1) * 4] for r in k]
        s = [[x / 2 for x in r] for r in mm(qh, _T(kh))]
        check(np.allclose(_arr(s), (_arr(qh) @ _arr(kh).T) / np.sqrt(4)), "NumPy")
        text = (f"Multi-head attention với $d_{{\\mathrm{{model}}}} = 8$ và $2$ đầu, mỗi đầu có $d_k = 4$ (đầu $1$ dùng cột $1$–$4$, đầu $2$ dùng cột $5$–$8$). "
                f"$\\mathbf{{Q}} = {mat_tex(q)}$, $\\mathbf{{K}} = {mat_tex(k)}$. Tính ma trận điểm $\\mathbf{{S}} = \\mathbf{{Q}}_h\\mathbf{{K}}_h\\T/\\sqrt{{d_k}}$ của **đầu {h + 1}**. Nhập từng phần tử.")
        steps = [Step(text=f"Lấy các cột của đầu {h + 1}: $\\mathbf{{Q}}_h = {mat_tex(qh)}$, $\\mathbf{{K}}_h = {mat_tex(kh)}$; $\\sqrt{{d_k}} = 2$:", latex="\\mathbf{S} = \\dfrac{\\mathbf{Q}_h\\mathbf{K}_h\\T}{2} = " + mat_tex(s))]
        ans = mat_ans(s)
    else:
        n = {1: 2, 2: 2, 3: 3, 4: 3}[level]
        q, k = rand_mat(rng, n, d, -2, 2), rand_mat(rng, n, d, -2, 2)
        raw = mm(q, _T(k))
        s = [[x / 2 for x in r] for r in raw]
        check(np.allclose(_arr(s), (_arr(q) @ _arr(k).T) / np.sqrt(d)), "NumPy")
        base = f"$\\mathbf{{Q}} = {mat_tex(q)}$, $\\mathbf{{K}} = {mat_tex(k)}$ ($d_k = 4$). "
        if level == 1:
            i, j = rng.randrange(n), rng.randrange(n)
            ans = num_ans(s[i][j])
            text = base + f"Tính điểm attention $s_{{{i + 1}{j + 1}}} = \\mathbf{{q}}_{i + 1}\\T\\mathbf{{k}}_{j + 1}/\\sqrt{{d_k}}$. " + NUM_HINT
            steps = [Step(text="Tích vô hướng rồi chia $\\sqrt{d_k} = 2$:", latex=f"s_{{{i + 1}{j + 1}}} = \\dfrac{{{ft(raw[i][j])}}}{{2}} = {ft(s[i][j])}")]
        elif level in (2, 3):
            ans = mat_ans(s)
            text = base + "Tính ma trận điểm $\\mathbf{S} = \\mathbf{Q}\\mathbf{K}\\T/\\sqrt{d_k}$. Nhập từng phần tử."
            steps = [Step(text="$\\mathbf{Q}\\mathbf{K}\\T$ rồi chia $\\sqrt{d_k} = 2$:", latex="\\mathbf{Q}\\mathbf{K}\\T = " + mat_tex(raw) + ",\\quad \\mathbf{S} = " + mat_tex(s))]
        else:
            i = rng.randrange(n)
            val = sum(s[i][j] for j in range(i + 1))
            ans = num_ans(val)
            text = base + f"Attention nhân quả (causal): hàng ${i + 1}$ chỉ được nhìn các vị trí $j \\le {i + 1}$ (các vị trí sau bị che bằng $-\\infty$ trước softmax). Tính **tổng các điểm không bị che** của hàng ${i + 1}$ trong $\\mathbf{{S}} = \\mathbf{{Q}}\\mathbf{{K}}\\T/\\sqrt{{d_k}}$. " + NUM_HINT
            steps = [Step(text=f"Hàng ${i + 1}$ của $\\mathbf{{S}}$ là ${vt(s[i])}$; giữ $j \\le {i + 1}$:", latex=f"\\sum_{{j \\le {i + 1}}} s_{{{i + 1}j}} = {ft(val)}")]
    return dl(type_="attn-scores", lesson=L76, level=level, seed=seed, prompt=Prompt(text=text), answer=ans,
              hints=["Điểm attention: $s_{ij} = \\mathbf{q}_i\\T\\mathbf{k}_j/\\sqrt{d_k}$; ma trận $\\mathbf{S} = \\mathbf{Q}\\mathbf{K}\\T/\\sqrt{d_k}$.", "$\\sqrt{d_k}$ giữ phương sai của điểm không phình theo $d_k$ (ở đây $\\sqrt{4} = 2$).", "Mask nhân quả: $s_{ij} \\to -\\infty$ với $j > i$, sau softmax trọng số bằng $0$."],
              steps=steps, verified=["numpy"], tags=["attention", f"level{level}"])


# ----------------------------------------------------------------- attn-output
def _sm_row(ks):
    es = [F(2) ** k for k in ks]
    t = sum(es)
    return [e / t for e in es]


def gen_attn_output(rng: random.Random, level: int, seed: int):
    dv = 2
    if level <= 3:
        n = {1: 3, 2: 2, 3: 3}[level]
        a = []
        for _ in range(n):
            w = [rng.randint(1, 6) for _ in range(n)]
            t = sum(w)
            a.append([F(x, t) for x in w])
        v = rand_mat(rng, n, dv, -4, 4)
        out = mm(a, v)
        check(np.allclose(_arr(out), _arr(a) @ _arr(v)), "NumPy")
        if level == 1:
            i = rng.randrange(n)
            ans = vec_ans(out[i])
            text = f"Trọng số attention (softmax đã tính, mỗi hàng cộng bằng $1$) $\\mathbf{{A}} = {mat_tex(a)}$ và $\\mathbf{{V}} = {mat_tex(v)}$. Tính đầu ra của vị trí ${i + 1}$: $\\mathbf{{o}}_{i + 1} = \\sum_j a_{{{i + 1}j}}\\mathbf{{v}}_j$. " + VEC_HINT
            steps = [Step(text="Tổ hợp tuyến tính các hàng của $\\mathbf{V}$ với trọng số của hàng " + str(i + 1) + ":", latex="\\mathbf{o} = " + vt(out[i]))]
        else:
            ans = mat_ans(out)
            text = f"Trọng số attention $\\mathbf{{A}} = {mat_tex(a)}$ (mỗi hàng cộng bằng $1$) và $\\mathbf{{V}} = {mat_tex(v)}$. Tính đầu ra $\\mathbf{{O}} = \\mathbf{{A}}\\mathbf{{V}}$. Nhập từng phần tử."
            steps = [Step(text="$\\mathbf{O} = \\mathbf{A}\\mathbf{V}$ (mỗi hàng là trung bình có trọng số các hàng của $\\mathbf{V}$):", latex="\\mathbf{O} = " + mat_tex(out))]
    else:
        n = 2 if level == 4 else 3
        causal = level == 5
        ks = [[rng.randint(0, 3) for _ in range(n)] for _ in range(n)]
        a = []
        for i in range(n):
            if causal:
                row = _sm_row(ks[i][:i + 1]) + [F(0)] * (n - i - 1)
            else:
                row = _sm_row(ks[i])
            a.append(row)
        v = rand_mat(rng, n, dv, -4, 4)
        out = mm(a, v)
        # kiểm chứng độc lập: softmax số thực với điểm k ln 2 và mặt nạ -inf
        sc = np.array(ks, dtype=float) * np.log(2.0)
        if causal:
            sc = np.where(np.tril(np.ones((n, n))) > 0, sc, -np.inf)
        e = np.exp(sc - sc.max(axis=1, keepdims=True))
        sm = e / e.sum(axis=1, keepdims=True)
        check(np.allclose(sm @ _arr(v), _arr(out)), "attention NumPy")
        sh = "\\mathbf{S} = " + L.mat(__import__("sympy").Matrix([[__import__("sympy").Integer(x) for x in r] for r in ks])).replace("\\begin{bmatrix}", "\\ln 2\\cdot\\begin{bmatrix}")
        if level == 4:
            i = rng.randrange(n)
            ans = vec_ans(out[i])
            text = (f"Điểm attention $\\mathbf{{S}} = \\ln 2 \\cdot {mat_tex([[F(x) for x in r] for r in ks])}$ (mỗi $s_{{ij}} = k_{{ij}}\\ln 2$ nên $e^{{s_{{ij}}}} = 2^{{k_{{ij}}}}$), $\\mathbf{{V}} = {mat_tex(v)}$. "
                    f"Tính softmax theo hàng rồi đầu ra $\\mathbf{{o}}_{i + 1}$ của vị trí ${i + 1}$. " + VEC_HINT)
            steps = [Step(text=f"Hàng ${i + 1}$ của $e^{{\\mathbf{{S}}}}$ là ${vt([F(2) ** k for k in ks[i]])}$; chia cho tổng:", latex="\\mathbf{a}_" + str(i + 1) + " = " + vt(a[i]) + ",\\quad \\mathbf{o}_" + str(i + 1) + " = " + vt(out[i]))]
        else:
            ans = mat_ans(out)
            text = (f"Attention nhân quả (hàng $i$ chỉ nhìn các cột $j \\le i$). Điểm $\\mathbf{{S}} = \\ln 2 \\cdot {mat_tex([[F(x) for x in r] for r in ks])}$ (mỗi $s_{{ij}} = k_{{ij}}\\ln 2$, nên $e^{{s_{{ij}}}} = 2^{{k_{{ij}}}}$) và $\\mathbf{{V}} = {mat_tex(v)}$. "
                    "Tính $\\mathbf{O} = \\mathrm{softmax}(\\mathbf{S} + \\mathbf{M})\\mathbf{V}$ với $\\mathbf{M}$ là mặt nạ nhân quả. Nhập từng phần tử.")
            steps = [Step(text="Che các cột $j > i$ (trọng số $0$), softmax trên các cột còn lại:", latex="\\mathbf{A} = " + mat_tex(a) + ",\\quad \\mathbf{O} = " + mat_tex(out))]
    return dl(type_="attn-output", lesson=L76, level=level, seed=seed, prompt=Prompt(text=text), answer=ans,
              hints=["Đầu ra của attention: $\\mathbf{O} = \\mathbf{A}\\mathbf{V}$ với $\\mathbf{A} = \\mathrm{softmax}(\\mathbf{S})$ theo hàng.", "Mỗi hàng của $\\mathbf{A}$ cộng bằng $1$, nên mỗi vector ra là trung bình có trọng số của các hàng $\\mathbf{V}$.", "Với $s = k\\ln 2$: $e^{s} = 2^{k}$, các trọng số là số hữu tỉ. Mặt nạ nhân quả cho trọng số $0$ ở cột $j > i$."],
              steps=steps, verified=["numpy-softmax-attention"], tags=["attention", f"level{level}"])


# ------------------------------------------------------------- transformer-params
def gen_transformer_params(rng: random.Random, level: int, seed: int):
    d = 8 * rng.randint(8, 160)
    f = rng.choice([2, 4, 4, 4]) * d
    nl = rng.choice([2, 4, 6, 8, 12, 24])
    vocab = 1000 * rng.randint(1, 64)
    T = 64 * rng.randint(1, 32)

    def mha(dm):
        return {"Wq": dm * dm + dm, "Wk": dm * dm + dm, "Wv": dm * dm + dm, "Wo": dm * dm + dm}

    def ffn(dm, ff):
        return {"W1": dm * ff + ff, "W2": ff * dm + dm}

    def ln(dm):
        return {"gamma": dm, "beta": dm}

    if level == 1:
        val, what = sum(mha(d).values()), "một khối multi-head attention (bốn phép chiếu $\\mathbf{W}_Q, \\mathbf{W}_K, \\mathbf{W}_V, \\mathbf{W}_O$ đều $d\\times d$, có hệ số chặn)"
        form = f"4(d^2 + d) = 4({d}^2 + {d})"
    elif level == 2:
        val, what = sum(ffn(d, f).values()), f"một khối feed-forward $d \\to d_{{\\mathrm{{ff}}}} \\to d$ với $d_{{\\mathrm{{ff}}}} = {f}$ (hai lớp tuyến tính, có hệ số chặn)"
        form = f"(d\\,d_{{\\mathrm{{ff}}}} + d_{{\\mathrm{{ff}}}}) + (d_{{\\mathrm{{ff}}}}\\,d + d)"
    elif level == 3:
        val = sum(mha(d).values()) + sum(ffn(d, f).values()) + 2 * sum(ln(d).values())
        what = f"**một lớp** encoder (multi-head attention + feed-forward $d_{{\\mathrm{{ff}}}} = {f}$ + hai LayerNorm; mọi lớp tuyến tính có hệ số chặn, mỗi LayerNorm có $\\gamma, \\beta$)"
        form = "\\text{attn} + \\text{ffn} + 2\\cdot 2d"
    elif level == 4:
        layer = sum(mha(d).values()) + sum(ffn(d, f).values()) + 2 * sum(ln(d).values())
        val = nl * layer + vocab * d
        what = f"mô hình gồm ${nl}$ lớp encoder như trên ($d_{{\\mathrm{{ff}}}} = {f}$) cộng bảng embedding token $V\\times d$ với $V = {vocab}$ (chưa tính embedding vị trí và lớp đầu ra)"
        form = f"{nl}\\cdot\\text{{layer}} + Vd"
    else:
        layer = sum(mha(d).values()) + sum(ffn(d, f).values()) + 2 * sum(ln(d).values())
        val = nl * layer + vocab * d + T * d + sum(ln(d).values())
        what = (f"mô hình ${nl}$ lớp ($d_{{\\mathrm{{ff}}}} = {f}$) với embedding token $V = {vocab}$, embedding vị trí học được cho $T = {T}$ vị trí, một LayerNorm cuối, và lớp đầu ra dùng **weight tying** (dùng lại bảng embedding, không thêm tham số)")
        form = f"{nl}\\cdot\\text{{layer}} + Vd + Td + 2d"
    # kiểm chứng độc lập: dựng danh sách hình dạng tensor rồi cộng tích
    shapes = []
    if level >= 1:
        shapes += [(d, d), (d,)] * 4 if level in (1, 3, 4, 5) else []
    if level in (2, 3, 4, 5):
        shapes += [(d, f), (f,), (f, d), (d,)]
    if level in (3, 4, 5):
        shapes += [(d,)] * 4
    base_shapes = [int(np.prod(s)) for s in shapes]
    tot = sum(base_shapes)
    if level == 4:
        tot = nl * tot + vocab * d
    if level == 5:
        tot = nl * tot + vocab * d + T * d + 2 * d
    check(tot == val, "đếm theo hình dạng tensor ≠ công thức")
    return dl(type_="transformer-params", lesson=L76, level=level, seed=seed,
              prompt=Prompt(text=f"Với $d = d_{{\\mathrm{{model}}}} = {d}$: tính số tham số học được của {what}. " + NUM_HINT), answer=num_ans(F(val)),
              hints=["Lớp tuyến tính $n_{\\mathrm{in}}\\to n_{\\mathrm{out}}$ có $n_{\\mathrm{in}}n_{\\mathrm{out}} + n_{\\mathrm{out}}$ tham số (trọng số + chặn).", "LayerNorm có $2d$ tham số ($\\gamma$ và $\\beta$). Attention và softmax không có tham số học thêm.", "Weight tying: lớp đầu ra dùng chung ma trận embedding nên không đếm thêm."],
              steps=[Step(text="Cộng theo từng thành phần:", latex=f"{form} = {val}")], verified=["tensor-shape-recount"], tags=["params", f"level{level}"])


# --------------------------------------------------------------------- lora-calc
def gen_lora_calc(rng: random.Random, level: int, seed: int):
    if level <= 3:
        din = rng.choice([64, 128, 256, 512, 768, 1024, 2048, 4096])
        dout = rng.choice([64, 128, 256, 512, 768, 1024, 2048, 4096])
        r = rng.choice([1, 2, 4, 8, 16, 32])
        full, lora = din * dout, r * (din + dout)
        if level == 1:
            val, text = F(full), f"Ma trận trọng số $\\mathbf{{W}} \\in \\R^{{{dout}\\times {din}}}$ (không chặn). Fine-tuning đầy đủ cập nhật bao nhiêu tham số của $\\mathbf{{W}}$? "
            st = f"{dout}\\cdot {din} = {full}"
        elif level == 2:
            val = F(lora)
            text = f"LoRA thay cập nhật $\\Delta\\mathbf{{W}} = \\mathbf{{B}}\\mathbf{{A}}$ với $\\mathbf{{B}} \\in \\R^{{{dout}\\times {r}}}$, $\\mathbf{{A}} \\in \\R^{{{r}\\times {din}}}$ (hạng $r = {r}$) và đóng băng $\\mathbf{{W}}$. Có bao nhiêu tham số **huấn luyện được**? "
            st = f"r(d_{{\\mathrm{{in}}}} + d_{{\\mathrm{{out}}}}) = {r}({din} + {dout}) = {lora}"
        else:
            val = F(lora, full)
            text = f"LoRA hạng $r = {r}$ cho $\\mathbf{{W}} \\in \\R^{{{dout}\\times {din}}}$. Tỉ số (số tham số LoRA) / (số tham số của $\\mathbf{{W}}$) là bao nhiêu (phân số rút gọn)? "
            st = f"\\dfrac{{{lora}}}{{{full}}} = {ft(val)}"
        return dl(type_="lora-calc", lesson=L77, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(val),
                  hints=["LoRA: $\\mathbf{W}' = \\mathbf{W} + \\dfrac{\\alpha}{r}\\mathbf{B}\\mathbf{A}$, chỉ $\\mathbf{A}, \\mathbf{B}$ được huấn luyện.", "$\\mathbf{B}$ có $d_{\\mathrm{out}}r$ phần tử, $\\mathbf{A}$ có $r\\,d_{\\mathrm{in}}$ phần tử.", "Khi $r \\ll d$, số tham số LoRA nhỏ hơn rất nhiều."],
                  steps=[Step(text="Đếm số phần tử:", latex=st)], verified=["recount"], tags=["lora", f"level{level}"])
    if level == 4:
        r = rng.choice([1, 2])
        alpha = F(rng.choice([1, 2, 4, 8]))
        w = rand_mat(rng, 2, 2, -3, 3)
        b = rand_mat(rng, 2, r, -2, 2)
        a = rand_mat(rng, r, 2, -2, 2)
        delta = mm(b, a)
        out = [[w[i][j] + alpha / r * delta[i][j] for j in range(2)] for i in range(2)]
        check(np.allclose(_arr(out), _arr(w) + float(alpha) / r * (_arr(b) @ _arr(a))), "NumPy")
        text = (f"LoRA với hạng $r = {r}$, hệ số $\\alpha = {ft(alpha)}$: $\\mathbf{{W}}' = \\mathbf{{W}} + \\dfrac{{\\alpha}}{{r}}\\mathbf{{B}}\\mathbf{{A}}$. Cho $\\mathbf{{W}} = {mat_tex(w)}$, $\\mathbf{{B}} = {mat_tex(b)}$, $\\mathbf{{A}} = {mat_tex(a)}$. Tính $\\mathbf{{W}}'$. Nhập từng phần tử.")
        steps = [Step(text=f"$\\mathbf{{B}}\\mathbf{{A}} = {mat_tex(delta)}$ và hệ số $\\alpha/r = {ft(alpha / r)}$:", latex="\\mathbf{W}' = " + mat_tex(out))]
        return dl(type_="lora-calc", lesson=L77, level=level, seed=seed, prompt=Prompt(text=text), answer=mat_ans(out),
                  hints=["Tính tích $\\mathbf{B}\\mathbf{A}$ ($2\\times r$ nhân $r\\times 2$) trước.", "Nhân với hệ số $\\alpha/r$ rồi cộng vào $\\mathbf{W}$.", "Sau khi huấn luyện có thể gộp (merge) $\\mathbf{W}'$ để suy luận không tốn thêm độ trễ."],
                  steps=steps, verified=["numpy"], tags=["lora", "merge"])
    d = rng.choice([512, 768, 1024, 2048, 4096])
    nl = rng.choice([6, 12, 24, 32])
    r = rng.choice([4, 8, 16, 32])
    targets = rng.choice([("$\\mathbf{W}_Q$ và $\\mathbf{W}_V$", 2), ("$\\mathbf{W}_Q, \\mathbf{W}_K, \\mathbf{W}_V, \\mathbf{W}_O$", 4)])
    val = nl * targets[1] * r * (d + d)
    check(val == sum(r * (d + d) for _ in range(nl * targets[1])), "đếm tổng")
    text = f"Một Transformer ${nl}$ lớp có $d_{{\\mathrm{{model}}}} = {d}$. Áp dụng LoRA hạng $r = {r}$ lên {targets[0]} (mỗi ma trận $d\\times d$) ở **mọi** lớp. Tổng số tham số LoRA huấn luyện được? "
    return dl(type_="lora-calc", lesson=L77, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(F(val)),
              hints=["Mỗi ma trận được thích ứng có $r(d_{\\mathrm{in}} + d_{\\mathrm{out}}) = 2rd$ tham số LoRA.", "Nhân với số ma trận được thích ứng mỗi lớp và số lớp."],
              steps=[Step(text="Số ma trận $\\times$ tham số mỗi ma trận:", latex=f"{nl}\\cdot {targets[1]}\\cdot {r}\\cdot (2\\cdot {d}) = {val}")], verified=["recount"], tags=["lora", "model"])


# ------------------------------------------------------------------- torch-shape
def _fmt(shape):
    return "(" + ", ".join(str(v) for v in shape) + ("," if len(shape) == 1 else "") + ")"


def gen_torch_shape(rng: random.Random, level: int, seed: int):
    B, T = rng.choice([2, 3, 4, 8]), rng.choice([5, 6, 7, 10, 16])
    h = rng.choice([2, 4])
    dk = rng.choice([3, 4, 8, 16])
    D = h * dk
    H = rng.choice([D * 2, D * 4, 10, 32])
    kind = rng.choice(["a", "b"])
    if level == 1:
        code = f"x = torch.randn({B}, {T}, {D})\nlin = nn.Linear({D}, {H})\ny = lin(x)"
        shape = (B, T, H)
        sim = np.zeros((B, T, D)) @ np.zeros((D, H))
    elif level == 2:
        code = f"x = torch.randn({B}, {T}, {D})\ny = x.view({B}, {T}, {h}, {dk}).transpose(1, 2)"
        shape = (B, h, T, dk)
        sim = np.zeros((B, T, D)).reshape(B, T, h, dk).transpose(0, 2, 1, 3)
    elif level == 3:
        code = f"q = torch.randn({B}, {h}, {T}, {dk})\nk = torch.randn({B}, {h}, {T}, {dk})\nscores = q @ k.transpose(-2, -1)"
        shape = (B, h, T, T)
        sim = np.zeros((B, h, T, dk)) @ np.zeros((B, h, T, dk)).transpose(0, 1, 3, 2)
    elif level == 4:
        T2 = rng.choice([3, 4, 5])
        if kind == "a":
            code = f"x = torch.randn({B}, {T}, {D})\ny = x.mean(dim=1)"
            shape = (B, D)
            sim = np.zeros((B, T, D)).mean(axis=1)
        else:
            code = f"a = torch.randn({B}, {T}, {D})\nb = torch.randn({B}, {T2}, {D})\ny = torch.cat([a, b], dim=1)"
            shape = (B, T + T2, D)
            sim = np.concatenate([np.zeros((B, T, D)), np.zeros((B, T2, D))], axis=1)
    else:
        code = (f"x = torch.randn({B}, {T}, {D})\nq = x.view({B}, {T}, {h}, {dk}).transpose(1, 2)\nk = q.clone()\nv = q.clone()\n"
                f"att = torch.softmax(q @ k.transpose(-2, -1) / {dk} ** 0.5, dim=-1)\nout = (att @ v).transpose(1, 2).reshape({B}, {T}, {D})\ny = out.mean(dim=-1)")
        shape = (B, T)
        qn = np.zeros((B, T, D)).reshape(B, T, h, dk).transpose(0, 2, 1, 3)
        att = qn @ qn.transpose(0, 1, 3, 2)
        sim = (att @ qn).transpose(0, 2, 1, 3).reshape(B, T, D).mean(axis=-1)
    check(sim.shape == shape, "hình dạng NumPy ≠ đáp án")
    s = _fmt(shape)
    wrong = set()
    for _ in range(50):
        w = list(shape)
        rng.shuffle(w)
        if len(w) > 1 and tuple(w) != shape:
            wrong.add(_fmt(tuple(w)))
        w2 = list(shape) + [rng.choice([B, T, D, h])] if len(shape) < 4 else list(shape)[:-1]
        wrong.add(_fmt(tuple(w2)))
        if len(w) > 2:
            w3 = list(shape)
            w3[-1] = w3[-1] * 2
            wrong.add(_fmt(tuple(w3)))
    wrong.discard(s)
    wl = sorted(wrong)
    rng.shuffle(wl)
    choices, ans = choice_exercise(rng, s, wl[:3])
    return dl(type_="torch-shape", lesson=L77, level=level, seed=seed,
              prompt=Prompt(text="Đọc đoạn code PyTorch dưới đây. `y` (hoặc `scores`, `out` ở dòng cuối) có hình dạng (shape) nào?", code=code), answer=ans, choices=choices,
              hints=["Theo dõi shape sau từng dòng.", "`view`/`reshape` giữ tổng số phần tử; `transpose(a, b)` hoán đổi hai chiều; `@` nhân ma trận trên hai chiều cuối và giữ các chiều batch.", "`mean(dim=k)` bỏ chiều $k$; `torch.cat(..., dim=k)` cộng độ dài chiều $k$."],
              steps=[Step(text=f"Kết quả có shape **{s}**.")], verified=["numpy-shape-simulation"], tags=["pytorch", f"level{level}"])


SPECS = [
    TopicSpec("attn-scores", L76, (1, 2, 3, 4, 5), gen_attn_scores),
    TopicSpec("attn-output", L76, (1, 2, 3, 4, 5), gen_attn_output),
    TopicSpec("transformer-params", L76, (1, 2, 3, 4, 5), gen_transformer_params),
    TopicSpec("lora-calc", L77, (1, 2, 3, 4, 5), gen_lora_calc),
    TopicSpec("torch-shape", L77, (1, 2, 3, 4, 5), gen_torch_shape),
]
