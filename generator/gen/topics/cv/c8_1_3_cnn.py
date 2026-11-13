"""Bài 8.1 conv2d-compute, conv-shape. Bài 8.2 pool2d, receptive-field. Bài 8.3 cnn-params, resnet-block."""
from __future__ import annotations

import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..dl._d import (F, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, cv, fp, ft, mat_ans, mat_tex, num_ans, vec_ans, vt)

L81, L82, L83 = "8.1", "8.2", "8.3"


def _img(rng, n, lo=0, hi=5):
    return [[F(rng.randint(lo, hi)) for _ in range(n)] for _ in range(n)]


def _arr(m):
    return np.array([[float(v) for v in r] for r in m])


def _pad(m, p):
    n = len(m)
    out = [[F(0)] * (n + 2 * p) for _ in range(n + 2 * p)]
    for i in range(n):
        for j in range(n):
            out[i + p][j + p] = m[i][j]
    return out


def _conv(img, ker, s=1):
    n, k = len(img), len(ker)
    m = (n - k) // s + 1
    return [[sum(img[i * s + a][j * s + b] * ker[a][b] for a in range(k) for b in range(k)) for j in range(m)] for i in range(m)]


# ---------------------------------------------------------------- conv2d-compute
def gen_conv2d_compute(rng: random.Random, level: int, seed: int):
    n, k, p, s, ch = {1: (4, 2, 0, 1, 1), 2: (4, 2, 0, 1, 1), 3: (4, 3, 0, 1, 1), 4: (3, 3, 1, 1, 1), 5: (3, 2, 0, 1, 2)}[level]
    imgs = [_img(rng, n) for _ in range(ch)]
    kers = [[[F(rng.randint(-2, 2)) for _ in range(k)] for _ in range(k)] for _ in range(ch)]
    if all(v == 0 for kk in kers for r in kk for v in r):
        kers[0][0][0] = F(1)
    outs = [_conv(_pad(im, p), ke, s) for im, ke in zip(imgs, kers)]
    m = len(outs[0])
    out = [[sum(o[i][j] for o in outs) for j in range(m)] for i in range(m)]
    # kiểm chứng độc lập: sliding_window_view của NumPy
    tot = np.zeros((m, m))
    for im, ke in zip(imgs, kers):
        a = np.pad(_arr(im), p)
        w = sliding_window_view(a, (k, k))[::s, ::s]
        tot += (w * _arr(ke)).sum(axis=(2, 3))
    check(np.allclose(tot, _arr(out)), "NumPy conv ≠ Fraction")
    disp = []
    for c in range(ch):
        tag = f"_{c + 1}" if ch > 1 else ""
        disp.append(f"\\mathbf{{X}}{tag} = {mat_tex(imgs[c])},\\quad \\mathbf{{K}}{tag} = {mat_tex(kers[c])}")
    base = "Tích chập 2D theo kiểu học sâu (thực chất là tương quan chéo: **không** lật kernel). "
    if level == 1:
        i, j = rng.randrange(m), rng.randrange(m)
        text = base + f"Ảnh $\\mathbf{{X}}$ cỡ ${n}\\times {n}$, kernel $\\mathbf{{K}}$ cỡ ${k}\\times {k}$, stride $1$, không padding. Tính phần tử đầu ra $Y_{{{i + 1}{j + 1}}}$ (hàng ${i + 1}$, cột ${j + 1}$). " + NUM_HINT
        ans = num_ans(out[i][j])
        win = [imgs[0][i + a][j + b] for a in range(k) for b in range(k)]
        steps = [Step(text=f"Nhân kernel với cửa sổ $\\mathbf{{X}}[{i + 1}:{i + k},\\ {j + 1}:{j + k}]$ rồi cộng:", latex=f"Y_{{{i + 1}{j + 1}}} = {ft(out[i][j])}")]
    else:
        extra = {2: "stride $1$, không padding", 3: "stride $1$, không padding", 4: "padding $1$ (viền số $0$) và stride $1$ (đầu ra cùng kích thước với đầu vào)", 5: "stride $1$, không padding, **hai kênh** (cộng kết quả của hai kênh)"}[level]
        text = base + f"Ảnh cỡ ${n}\\times {n}$ và kernel ${k}\\times {k}$ với {extra}. Tính toàn bộ ảnh đầu ra $\\mathbf{{Y}}$ cỡ ${m}\\times {m}$. Nhập từng phần tử."
        ans = mat_ans(out)
        steps = [Step(text="Trượt kernel trên ảnh (đã padding nếu có), mỗi vị trí lấy tổng tích từng phần tử" + (", rồi cộng hai kênh:" if ch > 1 else ":"), latex="\\mathbf{Y} = " + mat_tex(out))]
    return cv(type_="conv2d-compute", lesson=L81, level=level, seed=seed, prompt=Prompt(text=text, display=disp), answer=ans,
              hints=["Mỗi phần tử đầu ra $= \\sum_{a,b} K_{ab}\\,X_{i+a,\\,j+b}$ trên một cửa sổ cùng cỡ kernel.", "Kích thước đầu ra (stride $1$): $n + 2p - k + 1$.", "Padding thêm viền $0$ quanh ảnh; nhiều kênh: tính từng kênh rồi cộng."],
              steps=steps, verified=["numpy-sliding-window"], tags=["conv", f"k{k}", f"p{p}"])


# ---------------------------------------------------------------------- conv-shape
def _out(n, k, p, s, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def gen_conv_shape(rng: random.Random, level: int, seed: int):
    n = rng.randint(7, 256)
    k = rng.choice([1, 3, 5, 7, 11])
    if level == 1:
        p, s, d = 0, 1, 1
    elif level == 2:
        p, s, d = rng.randint(0, 3), 1, 1
    elif level == 3:
        p, s, d = rng.randint(0, 3), rng.choice([2, 3, 4]), 1
    elif level == 4:
        p, s, d = rng.randint(0, 2), rng.choice([1, 2]), 1
    else:
        p, s, d = rng.randint(0, 3), rng.choice([1, 2]), rng.choice([2, 3])
    if _out(n, k, p, s, d) < 1:
        n += d * (k - 1) + 3
    val = _out(n, k, p, s, d)
    # kiểm chứng độc lập: đếm số vị trí cửa sổ hợp lệ trên ảnh đã padding
    pos = sum(1 for st in range(0, n + 2 * p, s) if st + d * (k - 1) < n + 2 * p)
    check(pos == val, "đếm vị trí cửa sổ ≠ công thức")
    if level == 4:
        k2, s2, p2 = rng.choice([1, 3]), rng.choice([1, 2]), rng.randint(0, 1)
        mid = val
        val = _out(mid, k2, p2, s2)
        if val < 1:
            return gen_conv_shape(random.Random(rng.random()), level, seed)
        pos2 = sum(1 for st in range(0, mid + 2 * p2, s2) if st + (k2 - 1) < mid + 2 * p2)
        check(pos2 == val, "đếm lớp 2")
        text = (f"Ảnh vuông cạnh ${n}$ qua hai lớp tích chập liên tiếp: lớp 1 có kernel ${k}$, padding ${p}$, stride ${s}$; lớp 2 có kernel ${k2}$, padding ${p2}$, stride ${s2}$. "
                "Cạnh của ảnh đầu ra cuối là bao nhiêu? ")
        steps = [Step(text="Áp dụng công thức cho từng lớp: $\\lfloor (n + 2p - k)/s \\rfloor + 1$.", latex=f"{n} \\to {mid} \\to {val}")]
    else:
        dtxt = f", dilation ${d}$" if d > 1 else ""
        text = f"Ảnh vuông cạnh ${n}$ qua một lớp tích chập kernel ${k}$, padding ${p}$, stride ${s}${dtxt}. Cạnh của ảnh đầu ra là bao nhiêu? "
        steps = [Step(text="Công thức: $\\left\\lfloor \\dfrac{n + 2p - d(k-1) - 1}{s} \\right\\rfloor + 1$:", latex=f"\\left\\lfloor \\dfrac{{{n} + {2 * p} - {d * (k - 1)} - 1}}{{{s}}} \\right\\rfloor + 1 = {val}")]
    return cv(type_="conv-shape", lesson=L81, level=level, seed=seed, prompt=Prompt(text=text + NUM_HINT), answer=num_ans(F(val)),
              hints=["Cạnh ra $= \\lfloor (n + 2p - k)/s \\rfloor + 1$ (không dilation).", "Với dilation $d$, kernel hiệu dụng rộng $d(k-1) + 1$.", "Padding $p = (k-1)/2$ với stride $1$ giữ nguyên kích thước (\"same\")."],
              steps=steps, verified=["window-position-count"], tags=["conv-shape", f"level{level}"])


# ------------------------------------------------------------------------- pool2d
def gen_pool2d(rng: random.Random, level: int, seed: int):
    if level == 5:  # global average pooling trên 2 kênh
        chs = [_img(rng, 3, 0, 9) for _ in range(2)]
        out = [sum(v for r in c for v in r) / 9 for c in chs]
        check(np.allclose([float(v) for v in out], [_arr(c).mean() for c in chs]), "NumPy")
        return cv(type_="pool2d", lesson=L82, level=level, seed=seed,
                  prompt=Prompt(text="Global average pooling lấy trung bình toàn bộ mỗi kênh. Với tensor $2$ kênh $3\\times 3$ dưới đây, tính vector đầu ra (một số cho mỗi kênh). " + VEC_HINT,
                                display=[f"\\mathbf{{X}}_1 = {mat_tex(chs[0])},\\quad \\mathbf{{X}}_2 = {mat_tex(chs[1])}"]),
                  answer=vec_ans(out), hints=["Global average pooling: trung bình tất cả $H\\times W$ giá trị của từng kênh.", "Đầu ra có đúng $C$ số (số kênh)."],
                  steps=[Step(text="Trung bình từng kênh:", latex="\\mathbf{y} = " + vt(out))], verified=["numpy-mean"], tags=["global-avg"])
    n = 4
    img = _img(rng, n, 0, 9)
    kind = {1: "max", 2: "max", 3: "avg", 4: rng.choice(["max", "avg"])}[level]
    k, s = (3, 1) if level == 4 else (2, 2)
    m = (n - k) // s + 1
    f = (lambda w: max(w)) if kind == "max" else (lambda w: sum(w) / len(w))
    out = [[f([img[i * s + a][j * s + b] for a in range(k) for b in range(k)]) for j in range(m)] for i in range(m)]
    w = sliding_window_view(_arr(img), (k, k))[::s, ::s]
    ref = w.max(axis=(2, 3)) if kind == "max" else w.mean(axis=(2, 3))
    check(np.allclose(ref, _arr(out)), "NumPy pooling")
    name = "max pooling" if kind == "max" else "average pooling"
    if level == 1:
        i, j = rng.randrange(m), rng.randrange(m)
        text = f"Max pooling cửa sổ ${k}\\times {k}$, stride ${s}$ trên ảnh $\\mathbf{{X}}$ dưới đây. Tính phần tử đầu ra hàng ${i + 1}$, cột ${j + 1}$. " + NUM_HINT
        ans = num_ans(out[i][j])
        steps = [Step(text="Lấy giá trị lớn nhất trong cửa sổ tương ứng:", latex=f"Y_{{{i + 1}{j + 1}}} = {ft(out[i][j])}")]
    else:
        text = f"Áp dụng {name} cửa sổ ${k}\\times {k}$, stride ${s}$ lên $\\mathbf{{X}}$ ($4\\times 4$). Tính ảnh đầu ra ${m}\\times {m}$. Nhập từng phần tử."
        ans = mat_ans(out)
        steps = [Step(text=f"Mỗi cửa sổ lấy {'giá trị lớn nhất' if kind == 'max' else 'trung bình'}:", latex="\\mathbf{Y} = " + mat_tex(out))]
    return cv(type_="pool2d", lesson=L82, level=level, seed=seed, prompt=Prompt(text=text, display=["\\mathbf{X} = " + mat_tex(img)]), answer=ans,
              hints=["Pooling không có tham số học được; chỉ rút gọn mỗi cửa sổ thành một số.", "Cạnh ra $= \\lfloor (n - k)/s \\rfloor + 1$.", "Max pooling giữ phản hồi mạnh nhất; average pooling lấy trung bình."],
              steps=steps, verified=["numpy-sliding-window"], tags=["pool", kind])


# ---------------------------------------------------------------- receptive-field
def gen_receptive_field(rng: random.Random, level: int, seed: int):
    nl = {1: 1, 2: 2, 3: 3, 4: 3, 5: 4}[level]
    layers = []
    for i in range(nl):
        k = rng.randint(2, 45) if level == 1 else rng.randint(2, 11)
        s = 1 if level <= 2 else rng.choice([1, 1, 2, 3])
        d = 1 if level <= 3 else rng.choice([1, 1, 2, 3])
        layers.append((k, s, d))
    # công thức: r += d (k-1) jump; jump *= s
    r, j = 1, 1
    for k, s, d in layers:
        r += d * (k - 1) * j
        j *= s
    # kiểm chứng độc lập: lan truyền ngược khoảng vị trí đầu vào ảnh hưởng một đầu ra
    lo, hi = 0, 0
    for k, s, d in reversed(layers):
        lo, hi = lo * s, hi * s + d * (k - 1)
    check(hi - lo + 1 == r, "trường tiếp nhận theo lan truyền ngược ≠ công thức")
    desc = ", ".join(f"kernel ${k}$ stride ${s}$" + (f" dilation ${d}$" if d > 1 else "") for k, s, d in layers)
    steps = []
    rr, jj = 1, 1
    for i, (k, s, d) in enumerate(layers, 1):
        rr += d * (k - 1) * jj
        jj *= s
        steps.append(Step(text=f"Sau lớp {i} (kernel {k}, stride {s}" + (f", dilation {d}" if d > 1 else "") + f"): trường tiếp nhận ${rr}$, bước nhảy ${jj}$."))
    return cv(type_="receptive-field", lesson=L82, level=level, seed=seed,
              prompt=Prompt(text=f"Một chồng ${nl}$ lớp tích chập 1D/2D (theo mỗi chiều) liên tiếp: {desc}. Trường tiếp nhận (receptive field) của một nơ-ron ở lớp cuối, tính theo số vị trí của đầu vào, là bao nhiêu? " + NUM_HINT),
              answer=num_ans(F(r)), hints=["Trường tiếp nhận ban đầu là $1$ và bước nhảy (jump) là $1$.", "Mỗi lớp: $r \\leftarrow r + d(k - 1)\\cdot j$ ($d$ là dilation, mặc định $1$) rồi $j \\leftarrow j\\cdot s$.", "Hai lớp $3\\times 3$ stride $1$ có trường tiếp nhận $5$, giống một lớp $5\\times 5$."],
              steps=steps, verified=["backward-interval-propagation"], tags=["rf", f"{nl}layer"])


# --------------------------------------------------------------------- cnn-params
def gen_cnn_params(rng: random.Random, level: int, seed: int):
    cin = rng.choice([1, 3, 16, 32, 64, 128, 256])
    cout = rng.choice([8, 16, 32, 64, 128, 256, 512])
    k = rng.choice([1, 3, 5, 7])
    H = rng.choice([7, 14, 28, 32, 56, 112, 224])
    if level == 1:
        val, text = cout * (cin * k * k + 1), f"Lớp tích chập $C_{{\\mathrm{{in}}}} = {cin}$, $C_{{\\mathrm{{out}}}} = {cout}$, kernel ${k}\\times {k}$, **có** hệ số chặn. Số tham số?"
        st = f"{cout}({cin}\\cdot {k}\\cdot {k} + 1) = {val}"
        recount = sum(cin * k * k + 1 for _ in range(cout))
    elif level == 2:
        val, text = cout * cin * k * k, f"Lớp tích chập $C_{{\\mathrm{{in}}}} = {cin}$, $C_{{\\mathrm{{out}}}} = {cout}$, kernel ${k}\\times {k}$, **không** có hệ số chặn (thường khi đi kèm BatchNorm). Số tham số?"
        st = f"{cout}\\cdot {cin}\\cdot {k}^2 = {val}"
        recount = sum(cin * k * k for _ in range(cout))
    elif level == 3:
        std = cin * cout * k * k
        sep = cin * k * k + cin * cout
        val = sep
        text = f"Tích chập tách theo chiều sâu (depthwise separable): depthwise ${k}\\times {k}$ trên mỗi kênh ($C_{{\\mathrm{{in}}}} = {cin}$ bộ lọc) rồi pointwise $1\\times 1$ lên $C_{{\\mathrm{{out}}}} = {cout}$ kênh (không chặn). Số tham số? (Tích chập chuẩn tương ứng có ${std}$.)"
        st = f"{cin}\\cdot {k}^2 + {cin}\\cdot {cout} = {val}"
        recount = cin * k * k + cin * cout
    elif level == 4:
        Ho = H  # padding 'same', stride 1
        val = cout * cin * k * k * Ho * Ho
        text = f"Tích chập ${k}\\times {k}$ ($C_{{\\mathrm{{in}}}} = {cin}$, $C_{{\\mathrm{{out}}}} = {cout}$), padding \"same\", stride $1$, ảnh vào ${H}\\times {H}$ (đầu ra cũng ${H}\\times {H}$). Số phép nhân–cộng (MAC) cho một ảnh, bỏ qua hệ số chặn?"
        st = f"{cout}\\cdot {cin}\\cdot {k}^2\\cdot {Ho}^2 = {val}"
        recount = sum(cin * k * k for _ in range(cout)) * Ho * Ho
    else:
        c1, c2, c3 = rng.choice([8, 16, 32]), rng.choice([16, 32, 64]), rng.choice([32, 64, 128])
        ncls = rng.choice([10, 100, 1000])
        pool_cnt = 7
        val = (3 * 9 + 1) * c1 + (c1 * 9 + 1) * c2 + (c2 * 9 + 1) * c3 + (c3 + 1) * ncls
        text = (f"CNN: ảnh RGB $\\to$ conv $3\\times 3$ ($3\\to {c1}$) $\\to$ conv $3\\times 3$ (${c1}\\to {c2}$) $\\to$ conv $3\\times 3$ (${c2}\\to {c3}$) $\\to$ global average pooling $\\to$ lớp tuyến tính ${c3}\\to {ncls}$. "
                "Mọi lớp có hệ số chặn, padding \"same\". Tổng số tham số?")
        st = f"{3 * 9 + 1}\\cdot {c1} + ({c1}\\cdot 9 + 1)\\cdot {c2} + ({c2}\\cdot 9 + 1)\\cdot {c3} + ({c3} + 1)\\cdot {ncls} = {val}"
        recount = sum(a for a in [(3 * 9 + 1) * c1, (c1 * 9 + 1) * c2, (c2 * 9 + 1) * c3, (c3 + 1) * ncls])
    check(recount == val, "đếm lại")
    return cv(type_="cnn-params", lesson=L83, level=level, seed=seed, prompt=Prompt(text=text + " " + NUM_HINT), answer=num_ans(F(val)),
              hints=["Một bộ lọc có $C_{\\mathrm{in}}\\cdot k\\cdot k$ trọng số (+ $1$ chặn); có $C_{\\mathrm{out}}$ bộ lọc.", "Tham số không phụ thuộc kích thước ảnh; MAC thì có: nhân thêm số vị trí đầu ra $H_{\\mathrm{out}}W_{\\mathrm{out}}$.", "Global average pooling không có tham số."],
              steps=[Step(text="Áp dụng công thức:", latex=st)], verified=["recount"], tags=["cnn-params", f"level{level}"])


# ------------------------------------------------------------------- resnet-block
def gen_resnet_block(rng: random.Random, level: int, seed: int):
    c = 8 * rng.randint(1, 80)
    if level == 1:
        val = 2 * 9 * c * c
        text, st = f"Khối residual cơ bản (basic block) của ResNet: hai conv $3\\times 3$ giữ $C = {c}$ kênh, **không** hệ số chặn, bỏ qua BatchNorm. Số tham số?", f"2\\cdot 9\\cdot {c}^2 = {val}"
    elif level == 2:
        val = 2 * 9 * c * c + 2 * 2 * c
        text, st = f"Khối cơ bản như trên ($C = {c}$, hai conv $3\\times 3$ không chặn) **kèm** hai lớp BatchNorm (mỗi lớp $2C$ tham số $\\gamma, \\beta$). Số tham số?", f"2\\cdot 9\\cdot {c}^2 + 2\\cdot 2\\cdot {c} = {val}"
    elif level == 3:
        val = 4 * c * c + 9 * c * c + c * 4 * c
        text, st = f"Khối bottleneck: conv $1\\times 1$ ($4C\\to C$), conv $3\\times 3$ ($C\\to C$), conv $1\\times 1$ ($C\\to 4C$) với $C = {c}$, không chặn, bỏ qua BatchNorm. Số tham số?", f"4\\cdot {c}^2 + 9\\cdot {c}^2 + 4\\cdot {c}^2 = {val}"
    elif level == 4:
        val = 4 * c * c + 9 * c * c + 4 * c * c + 2 * (c + c + 4 * c)
        text, st = f"Khối bottleneck ($1\\times 1$: $4C\\to C$, $3\\times 3$: $C\\to C$, $1\\times 1$: $C\\to 4C$; $C = {c}$, conv không chặn) **kèm** BatchNorm sau mỗi conv. Số tham số?", f"17\\cdot {c}^2 + 2(C + C + 4C) = {val}"
    else:
        cin, cout = c, 2 * c
        main = 9 * cin * cout + 9 * cout * cout
        short = cin * cout
        val = main + short
        text = (f"Khối cơ bản **giảm mẫu** của ResNet: conv $3\\times 3$ stride $2$ ($C = {cin}\\to {cout}$), conv $3\\times 3$ ($ {cout}\\to {cout}$), và nhánh tắt chiếu bằng conv $1\\times 1$ stride $2$ (${cin}\\to {cout}$). "
                "Mọi conv không chặn, bỏ qua BatchNorm. Số tham số của cả khối?")
        st = f"9\\cdot {cin}\\cdot {cout} + 9\\cdot {cout}^2 + {cin}\\cdot {cout} = {val}"
    chk = {1: 2 * 9 * c * c, 2: 2 * 9 * c * c + 4 * c, 3: 17 * c * c, 4: 17 * c * c + 12 * c, 5: 9 * c * 2 * c + 9 * 4 * c * c + 2 * c * c}[level]
    check(chk == val, "đếm lại")
    return cv(type_="resnet-block", lesson=L83, level=level, seed=seed, prompt=Prompt(text=text + " " + NUM_HINT), answer=num_ans(F(val)),
              hints=["Mỗi conv $k\\times k$ từ $C_a$ đến $C_b$ kênh có $k^2C_aC_b$ trọng số.", "Nhánh tắt (identity) không có tham số; nhánh chiếu $1\\times 1$ có $C_{\\mathrm{in}}C_{\\mathrm{out}}$.", "BatchNorm có $2C$ tham số học được."],
              steps=[Step(text="Cộng tham số các lớp:", latex=st)], verified=["recount"], tags=["resnet", f"level{level}"])


SPECS = [
    TopicSpec("conv2d-compute", L81, (1, 2, 3, 4, 5), gen_conv2d_compute),
    TopicSpec("conv-shape", L81, (1, 2, 3, 4, 5), gen_conv_shape),
    TopicSpec("pool2d", L82, (1, 2, 3, 4, 5), gen_pool2d),
    TopicSpec("receptive-field", L82, (1, 2, 3, 4, 5), gen_receptive_field),
    TopicSpec("cnn-params", L83, (1, 2, 3, 4, 5), gen_cnn_params),
    TopicSpec("resnet-block", L83, (1, 2, 3, 4, 5), gen_resnet_block),
]
