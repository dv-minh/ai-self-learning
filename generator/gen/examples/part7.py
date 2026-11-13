r"""Ví dụ cho Phần 7 (Deep Learning). Mọi số do chương trình tính bằng Fraction và kiểm chứng; MDX chỉ chứa kết quả."""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np

from ..core.verify import check
from ..topics.dl._d import mm, mv, relu
from ..topics.opt._o import fp, ft, mat_tex, vt
from .common import display, write_examples

EX: dict[str, dict] = {}


def lesson(code: str):
    def deco(fn):
        EX.setdefault(code, {})[fn.__name__.replace("_", "-")] = fn
        return fn
    return deco


def table(header: list[str], rows: list[list[str]]) -> str:
    cols = "c" * len(header)
    body = " \\\\ ".join(" & ".join(r) for r in rows)
    return f"\\begin{{array}}{{{cols}}} {' & '.join(header)} \\\\ \\hline {body} \\end{{array}}"


def _arr(m):
    return np.array([[float(v) for v in r] for r in m])


# ------------------------------------------------------------------ 7.1
@lesson("7.1")
def mlp_forward_example():
    w1, b1 = [[F(1), F(-1)], [F(2), F(1)]], [F(0), F(-5)]
    w2, b2 = [[F(1), F(2)]], [F(1)]
    x = [F(2), F(1)]
    z1 = [a + b for a, b in zip(mv(w1, x), b1)]
    a1 = [relu(v) for v in z1]
    z2 = [a + b for a, b in zip(mv(w2, a1), b2)]
    ref = np.maximum(_arr(w1) @ np.array([2.0, 1.0]) + np.array([0.0, -5.0]), 0)
    check(np.allclose(ref, [float(v) for v in a1]), "NumPy")
    return display(f"\\mathbf{{x}} = {vt(x)},\\ \\mathbf{{W}}_1 = {mat_tex(w1)},\\ \\mathbf{{b}}_1 = {vt(b1)}", f"\\mathbf{{z}}_1 = \\mathbf{{W}}_1\\mathbf{{x}} + \\mathbf{{b}}_1 = {vt(z1)},\\quad \\mathbf{{a}}_1 = \\mathrm{{ReLU}}(\\mathbf{{z}}_1) = {vt(a1)}",
                   f"\\mathbf{{W}}_2 = {mat_tex(w2)},\\ \\mathbf{{b}}_2 = {vt(b2)}:\\quad \\hat{{y}} = \\mathbf{{W}}_2\\mathbf{{a}}_1 + \\mathbf{{b}}_2 = {ft(z2[0])}") + (f"\n\nNơ-ron thứ hai có $z = {ft(z1[1])} \\le 0$ nên bị ReLU cắt về $0$ và không đóng góp cho đầu ra." if z1[1] <= 0 else "")


@lesson("7.1")
def perceptron_gates():
    rows = []
    for x1 in (0, 1):
        for x2 in (0, 1):
            z_and = x1 + x2 - F(3, 2)
            z_or = x1 + x2 - F(1, 2)
            rows.append([str(x1), str(x2), "1" if z_and > 0 else "0", "1" if z_or > 0 else "0"])
    return display("\\hat{y} = \\mathbb{1}[w_1x_1 + w_2x_2 + b > 0],\\quad \\text{AND}: (1, 1, -\\tfrac{3}{2}),\\ \\text{OR}: (1, 1, -\\tfrac{1}{2})", table(["x_1", "x_2", "\\mathrm{AND}", "\\mathrm{OR}"], rows)) + \
        "\n\nMột perceptron là một đường thẳng chia mặt phẳng; XOR không thể tách bằng một đường thẳng nên cần ít nhất một lớp ẩn."


@lesson("7.1")
def param_count_example():
    sizes = [784, 128, 64, 10]
    parts = [(sizes[i] * sizes[i + 1], sizes[i + 1]) for i in range(3)]
    total = sum(a + b for a, b in parts)
    check(total == sum((sizes[i] + 1) * sizes[i + 1] for i in range(3)), "đếm theo (n_in + 1) n_out")
    lines = [f"{sizes[i]}\\cdot {sizes[i + 1]} + {sizes[i + 1]} = {parts[i][0] + parts[i][1]}" for i in range(3)]
    return display("784 \\to 128 \\to 64 \\to 10", *lines, f"\\text{{total}} = {total}")


@lesson("7.1")
def prop_linear_collapse():
    w1, w2 = [[F(1), F(2)], [F(0), F(1)]], [[F(2), F(0)], [F(1), F(1)]]
    x = [F(3), F(-1)]
    stacked = mv(w2, mv(w1, x))
    single = mv(mm(w2, w1), x)
    check(stacked == single, "hai lớp tuyến tính = một lớp tuyến tính")
    return display(f"\\mathbf{{W}}_2(\\mathbf{{W}}_1\\mathbf{{x}}) = {vt(stacked)},\\qquad (\\mathbf{{W}}_2\\mathbf{{W}}_1)\\mathbf{{x}} = {vt(single)}", f"\\mathbf{{W}}_2\\mathbf{{W}}_1 = {mat_tex(mm(w2, w1))}")


@lesson("7.1")
def mistake_no_activation():
    return "**Sai:** xếp nhiều lớp tuyến tính mà không có hàm kích hoạt phi tuyến và hy vọng mô hình mạnh hơn.\n\n**Đúng:** $\\mathbf{W}_2(\\mathbf{W}_1\\mathbf{x}) = (\\mathbf{W}_2\\mathbf{W}_1)\\mathbf{x}$ chỉ là một phép biến đổi tuyến tính. Cần hàm kích hoạt phi tuyến (ReLU...) giữa các lớp."


# ------------------------------------------------------------------ 7.2
@lesson("7.2")
def relu_family():
    xs = [F(-4), F(-1), F(0), F(2), F(7)]
    a = F(1, 10)
    rows = [[ft(x), ft(relu(x)), ft(x if x > 0 else a * x), ft(min(max(x, F(0)), F(6)))] for x in xs]
    return display("\\alpha = \\tfrac{1}{10}", table(["z", "\\mathrm{ReLU}", "\\mathrm{LeakyReLU}", "\\mathrm{ReLU6}"], rows))


@lesson("7.2")
def softmax_ln2_example():
    ks = [2, 1, 0]
    es = [F(2) ** k for k in ks]
    t = sum(es)
    p = [e / t for e in es]
    z = np.array(ks) * np.log(2.0)
    sm = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    check(np.allclose(sm, [float(v) for v in p]), "softmax NumPy")
    return display("\\mathbf{z} = (2\\ln 2,\\ 1\\ln 2,\\ 0)\\T", f"e^{{\\mathbf{{z}}}} = {vt(es)},\\quad \\sum = {ft(t)},\\quad \\mathbf{{p}} = {vt(p)}", f"\\text{{true class }} 1:\\ \\ell = -\\ln p_1 = \\ln\\tfrac{{{ft(t)}}}{{{ft(es[0])}}},\\quad \\nabla_{{\\mathbf{{z}}}}\\ell = \\mathbf{{p}} - \\mathbf{{y}} = {vt([p[0] - 1, p[1], p[2]])}")


@lesson("7.2")
def softmax_shift_invariance():
    ks = [3, 1, 0]
    p = [F(2) ** k for k in ks]
    s = sum(p)
    p = [v / s for v in p]
    shifted = [F(2) ** (k - 3) for k in ks]
    s2 = sum(shifted)
    shifted = [v / s2 for v in shifted]
    check(p == shifted, "softmax bất biến khi cộng hằng số")
    return display(f"\\mathrm{{softmax}}(\\mathbf{{z}}) = \\mathrm{{softmax}}(\\mathbf{{z}} - c\\mathbf{{1}}) = {vt(p)}") + "\n\nTrừ giá trị lớn nhất khỏi mọi logit không đổi kết quả nhưng tránh tràn số khi tính $e^{z}$ (thủ thuật ổn định số)."


@lesson("7.2")
def mistake_dead_relu():
    z = [F(-3), F(-1), F(-2)]
    grads = [F(1) if v > 0 else F(0) for v in z]
    return f"**Sai:** bỏ qua khả năng \"ReLU chết\". Nếu đầu vào $z$ của một nơ-ron luôn $\\le 0$ (ví dụ ${vt(z)}$) thì đạo hàm ReLU luôn bằng $0$ ({vt(grads)}) và trọng số của nơ-ron đó không bao giờ được cập nhật.\n\n**Đúng:** dùng LeakyReLU/GELU, khởi tạo hợp lý và learning rate không quá lớn để hạn chế nơ-ron chết."


# ------------------------------------------------------------------ 7.3
@lesson("7.3")
def backprop_scalar():
    w1, w2, x, y = F(2), F(-1), F(3), F(1)
    h0 = w1 * x
    h = relu(h0)
    yh = w2 * h
    loss = (yh - y) ** 2 / 2
    dL_dyh = yh - y
    dL_dw2 = dL_dyh * h
    dL_dh = dL_dyh * w2
    dL_dh0 = dL_dh * (1 if h0 > 0 else 0)
    dL_dw1 = dL_dh0 * x
    f = lambda a, b: (float(b) * max(0.0, float(a) * float(x)) - float(y)) ** 2 / 2
    hh = 1e-6
    check(abs((f(float(w1) + hh, w2) - f(float(w1) - hh, w2)) / (2 * hh) - float(dL_dw1)) < 1e-5, "sai phân hữu hạn dw1")
    check(abs((f(w1, float(w2) + hh) - f(w1, float(w2) - hh)) / (2 * hh) - float(dL_dw2)) < 1e-5, "sai phân hữu hạn dw2")
    return display(f"h = \\mathrm{{ReLU}}(w_1x) = {ft(h)},\\quad \\hat{{y}} = w_2h = {ft(yh)},\\quad L = \\tfrac{{1}}{{2}}(\\hat{{y}} - y)^2 = {ft(loss)}",
                   f"\\dfrac{{\\partial L}}{{\\partial\\hat{{y}}}} = \\hat{{y}} - y = {ft(dL_dyh)},\\quad \\dfrac{{\\partial L}}{{\\partial w_2}} = {ft(dL_dyh)}\\cdot h = {ft(dL_dw2)}",
                   f"\\dfrac{{\\partial L}}{{\\partial h}} = {ft(dL_dyh)}\\cdot w_2 = {ft(dL_dh)},\\quad \\dfrac{{\\partial L}}{{\\partial w_1}} = {ft(dL_dh)}\\cdot\\mathrm{{ReLU}}'(w_1x)\\cdot x = {ft(dL_dw1)}")


@lesson("7.3")
def backprop_matrix():
    w1 = [[F(1), F(-1)], [F(2), F(1)]]
    w2 = [F(1), F(2)]
    x, y = [F(1), F(2)], F(3)
    z = mv(w1, x)
    h = [relu(v) for v in z]
    yh = sum(a * b for a, b in zip(w2, h))
    delta = yh - y
    dz = [delta * w2[i] * (1 if z[i] > 0 else 0) for i in range(2)]
    gw1 = [[dz[i] * x[j] for j in range(2)] for i in range(2)]
    return display(f"\\mathbf{{z}} = {vt(z)},\\ \\mathbf{{h}} = {vt(h)},\\ \\hat{{y}} = {ft(yh)},\\ \\delta = \\hat{{y}} - y = {ft(delta)}", f"\\dfrac{{\\partial L}}{{\\partial\\mathbf{{z}}}} = \\delta\\,(\\mathbf{{w}}_2 \\odot \\mathbb{{1}}[\\mathbf{{z}} > 0]) = {vt(dz)}",
                   "\\dfrac{\\partial L}{\\partial\\mathbf{W}_1} = \\dfrac{\\partial L}{\\partial\\mathbf{z}}\\,\\mathbf{x}\\T = " + mat_tex(gw1))


@lesson("7.3")
def gradient_check():
    w, x, y = F(2), F(3), F(1)
    L = lambda ww: (relu(ww * x) - y) ** 2 / 2
    g = (relu(w * x) - y) * x
    h = F(1, 1000)
    num = (L(w + h) - L(w - h)) / (2 * h)
    check(num == g, "với hàm tuyến tính từng khúc và h nhỏ, sai phân trung tâm khớp chính xác")
    return display(f"L(w) = \\tfrac{{1}}{{2}}(\\mathrm{{ReLU}}(wx) - y)^2,\\ x = {ft(x)},\\ y = {ft(y)},\\ w = {ft(w)}", f"\\text{{backprop}}: {ft(g)},\\qquad \\dfrac{{L(w + h) - L(w - h)}}{{2h}} = {ft(num)}\\quad (h = \\tfrac{{1}}{{1000}})") + \
        "\n\nKiểm tra gradient bằng sai phân hữu hạn là cách chuẩn để phát hiện lỗi khi tự cài đặt backpropagation."


@lesson("7.3")
def prop_gradient_accumulates():
    a, b = F(2), F(3)
    # L = (a*b + a)^2 / 2: a xuất hiện ở hai nhánh
    u = a * b
    v = u + a
    L = v * v / 2
    dv = v
    da = dv * b + dv * 1
    f = lambda aa: ((float(aa) * float(b)) + float(aa)) ** 2 / 2
    hh = 1e-6
    check(abs((f(float(a) + hh) - f(float(a) - hh)) / (2 * hh) - float(da)) < 1e-4, "sai phân")
    return display(f"u = ab,\\ v = u + a,\\ L = \\tfrac{{1}}{{2}}v^2:\\quad \\dfrac{{\\partial L}}{{\\partial a}} = v\\cdot b + v\\cdot 1 = {ft(da)}") + "\n\nBiến xuất hiện ở nhiều nhánh nhận gradient là **tổng** các đóng góp."


@lesson("7.3")
def mistake_relu_backward():
    z, up = F(-2), F(5)
    return f"**Sai:** cho gradient đi qua ReLU nguyên vẹn. Với $z = {ft(z)}$ và gradient từ phía sau ${ft(up)}$, viết $\\partial L/\\partial z = {ft(up)}$.\n\n**Đúng:** $\\partial L/\\partial z = {ft(up)}\\cdot\\mathrm{{ReLU}}'(z) = {ft(up)}\\cdot 0 = 0$. ReLU chặn gradient của nơ-ron không hoạt động."


# ------------------------------------------------------------------ 7.4
@lesson("7.4")
def init_variance_demo():
    n = 100
    rows = []
    for c in (F(1, 2), F(2), F(4)):
        m = F(1)
        seq = [m]
        for _ in range(4):
            m = m * c / 2
            seq.append(m)
        rows.append([ft(c)] + [ft(v) for v in seq])
    return display(f"n = {n},\\ \\mathrm{{Var}}(w) = c/n,\\ \\text{{factor per ReLU layer}} = c/2", table(["c", "\\ell=0", "\\ell=1", "\\ell=2", "\\ell=3", "\\ell=4"], rows)) + \
        "\n\n$c = 2$ (khởi tạo He) giữ $\\mathbb{E}[a^2] = 1$ qua mọi lớp; $c < 2$ làm kích hoạt co về $0$ (gradient biến mất), $c > 2$ làm nó nổ."


@lesson("7.4")
def batchnorm_example():
    from itertools import combinations_with_replacement as combinations

    from ..topics.dl._d import sqrt_q

    for x in combinations(range(0, 12), 4):  # bộ 4 giá trị khác nhau đầu tiên có phương sai là bình phương hữu tỉ
        xs = [F(v) for v in x]
        mu = sum(xs) / 4
        var = sum((v - mu) ** 2 for v in xs) / 4
        s = sqrt_q(var)
        if s is not None and s > 0 and mu.denominator == 1 and len(set(x)) >= 3:
            break
    x = xs
    xh = [(v - mu) / s for v in x]
    gamma, beta = F(2), F(1)
    y = [gamma * h + beta for h in xh]
    xn = np.array([float(v) for v in x])
    check(np.allclose((xn - xn.mean()) / xn.std(), [float(v) for v in xh]), "NumPy")
    return display(f"\\mathbf{{x}} = {vt(x)},\\quad \\mu = {ft(mu)},\\quad \\sigma^2 = {ft(var)},\\quad \\sigma = {ft(s)}",
                   f"\\hat{{\\mathbf{{x}}}} = {vt(xh)},\\qquad \\mathbf{{y}} = \\gamma\\hat{{\\mathbf{{x}}}} + \\beta = {vt(y)}\\ \\ (\\gamma = {ft(gamma)},\\ \\beta = {ft(beta)})")


@lesson("7.4")
def layernorm_vs_batchnorm():
    x = [[F(1), F(3)], [F(5), F(7)]]  # 2 mẫu, 2 đặc trưng
    bn_mean = [sum(x[i][j] for i in range(2)) / 2 for j in range(2)]
    ln_mean = [sum(x[i]) / 2 for i in range(2)]
    return display("\\mathbf{X} = " + mat_tex(x) + "\\ (\\text{rows: samples, columns: features})", f"\\text{{BatchNorm: mean over samples (per column)}} = {vt(bn_mean)}", f"\\text{{LayerNorm: mean over features (per row)}} = {vt(ln_mean)}") + \
        "\n\nBatchNorm thống kê theo cột (qua batch) nên phụ thuộc kích thước batch; LayerNorm thống kê theo hàng (qua đặc trưng của một mẫu) nên không phụ thuộc batch, hợp với Transformer."


@lesson("7.4")
def mistake_epsilon_and_eval():
    return ("**Sai:** dùng thống kê của batch hiện tại khi suy luận (BatchNorm), khiến đầu ra của một mẫu phụ thuộc vào các mẫu khác trong batch.\n\n"
            "**Đúng:** khi huấn luyện dùng $\\mu, \\sigma^2$ của batch và cập nhật trung bình động (running mean, running variance); khi suy luận (`model.eval()`) dùng các giá trị đã lưu. LayerNorm không có vấn đề này.")


# ------------------------------------------------------------------ 7.5
@lesson("7.5")
def dropout_example():
    p = F(1, 4)
    x = [F(4), F(8), F(-12), F(16)]
    mask = [1, 0, 1, 1]
    scale = 1 / (1 - p)
    y = [m * v * scale for m, v in zip(mask, x)]
    e = [(1 - p) * v * scale for v in x]
    check(e == x, "kỳ vọng bằng x")
    return display(f"p = {ft(p)},\\ \\text{{scale}} = \\dfrac{{1}}{{1 - p}} = {ft(scale)}", f"\\mathbf{{x}} = {vt(x)},\\ \\mathbf{{m}} = {vt([F(m) for m in mask])},\\ \\mathbf{{y}} = \\mathbf{{m}}\\odot\\mathbf{{x}}/(1-p) = {vt(y)}", f"\\mathbb{{E}}[\\mathbf{{y}}] = {vt(e)} = \\mathbf{{x}}")


@lesson("7.5")
def residual_depth():
    w = F(1, 10)
    rows = []
    for n in (2, 4, 8):
        rows.append([str(n), ft(w**n), ft((1 + w) ** n)])
    return display(f"w = {ft(w)}", table(["n", "w^n\\ (\\text{plain})", "(1+w)^n\\ (\\text{residual})"], rows)) + "\n\nMạng thường nhân gradient với $w$ ở mỗi khối nên co về $0$; khối residual nhân với $1 + w \\approx 1$ nên gradient được giữ."


@lesson("7.5")
def embedding_example():
    e = [[F(1), F(0), F(2)], [F(-1), F(3), F(1)], [F(0), F(2), F(-2)], [F(4), F(1), F(0)]]
    toks = [2, 0, 3]
    rows = [e[t] for t in toks]
    onehot = [F(1) if i == toks[0] else F(0) for i in range(4)]
    via = [sum(onehot[i] * e[i][j] for i in range(4)) for j in range(3)]
    check(via == e[toks[0]], "one-hot nhân E = tra hàng")
    return display("\\mathbf{E} = " + mat_tex(e), f"\\text{{tokens}} = (3, 1, 4) \\Rightarrow \\ " + mat_tex(rows), f"\\mathbf{{e}}_3\\T\\mathbf{{E}} = {vt(via)}") + "\n\nTra bảng embedding tương đương nhân vector one-hot với $\\mathbf{E}$, nhưng chỉ cần đọc một hàng."


@lesson("7.5")
def mistake_dropout_eval():
    return "**Sai:** để dropout bật khi suy luận. Kết quả cho cùng một đầu vào thay đổi ngẫu nhiên giữa các lần chạy.\n\n**Đúng:** dropout chỉ áp dụng khi huấn luyện (`model.train()`); khi suy luận (`model.eval()`) tắt dropout. Với inverted dropout không cần nhân thêm hệ số lúc suy luận."


# ------------------------------------------------------------------ 7.6
@lesson("7.6")
def attention_scores_example():
    q = [[F(1), F(0), F(2), F(1)], [F(0), F(1), F(-1), F(2)]]
    k = [[F(2), F(1), F(0), F(0)], [F(1), F(-1), F(1), F(2)]]
    raw = mm(q, [list(r) for r in zip(*k)])
    s = [[v / 2 for v in r] for r in raw]
    check(np.allclose(_arr(s), _arr(q) @ _arr(k).T / 2), "NumPy")
    return display("\\mathbf{Q} = " + mat_tex(q) + ",\\ \\mathbf{K} = " + mat_tex(k) + ",\\ d_k = 4", "\\mathbf{Q}\\mathbf{K}\\T = " + mat_tex(raw) + ",\\quad \\mathbf{S} = \\dfrac{\\mathbf{Q}\\mathbf{K}\\T}{\\sqrt{d_k}} = " + mat_tex(s))


@lesson("7.6")
def attention_output_example():
    ks = [[1, 0], [2, 1]]
    a = []
    for r in ks:
        es = [F(2) ** k for k in r]
        t = sum(es)
        a.append([e / t for e in es])
    v = [[F(3), F(0)], [F(0), F(6)]]
    o = mm(a, v)
    return display("\\mathbf{S} = \\ln 2\\cdot " + mat_tex([[F(x) for x in r] for r in ks]), "\\mathbf{A} = \\mathrm{softmax}(\\mathbf{S}) = " + mat_tex(a), "\\mathbf{O} = \\mathbf{A}\\mathbf{V} = " + mat_tex(o)) + "\n\nMỗi hàng của $\\mathbf{O}$ là trung bình có trọng số các hàng của $\\mathbf{V}$; trọng số lớn hơn cho khóa có điểm cao hơn."


@lesson("7.6")
def causal_mask_example():
    ks = [[0, 3, 2], [1, 0, 3], [2, 1, 0]]
    a = []
    for i, r in enumerate(ks):
        es = [F(2) ** k for k in r[:i + 1]]
        t = sum(es)
        a.append([e / t for e in es] + [F(0)] * (2 - i))
    check(all(sum(r) == 1 for r in a), "hàng cộng bằng 1")
    return display("\\text{causal: } s_{ij} \\to -\\infty\\ (j > i)", "\\mathbf{A} = " + mat_tex(a)) + "\n\nHàng $1$ chỉ nhìn chính nó (trọng số $1$); mỗi vị trí không bao giờ \"nhìn thấy tương lai\", điều kiện cần để sinh văn bản từng token."


@lesson("7.6")
def params_example():
    d, f, nl, V = 512, 2048, 6, 32000
    attn = 4 * (d * d + d)
    ffn = d * f + f + f * d + d
    ln = 2 * (2 * d)
    layer = attn + ffn + ln
    total = nl * layer + V * d
    return display(f"d = {d},\\ d_{{\\mathrm{{ff}}}} = {f},\\ {nl}\\text{{ layers}},\\ V = {V}", f"\\text{{attn}} = 4(d^2 + d) = {attn},\\quad \\text{{ffn}} = {ffn},\\quad \\text{{LN}} = {ln}", f"\\text{{layer}} = {layer},\\qquad \\text{{total}} = {nl}\\cdot {layer} + Vd = {total}")


@lesson("7.6")
def mistake_no_scaling():
    d = 64
    return f"**Sai:** bỏ chia $\\sqrt{{d_k}}$. Nếu các thành phần của $\\mathbf{{q}}, \\mathbf{{k}}$ có phương sai $1$ thì $\\mathbf{{q}}\\T\\mathbf{{k}}$ có phương sai $d_k = {d}$ (độ lệch chuẩn $8$): điểm quá lớn làm softmax gần one-hot và gradient rất nhỏ.\n\n**Đúng:** chia $\\sqrt{{d_k}}$ để phương sai điểm về $1$."


# ------------------------------------------------------------------ 7.7
@lesson("7.7")
def lora_params_example():
    din = dout = 4096
    r = 8
    full, lora = din * dout, r * (din + dout)
    return display(f"d_{{\\mathrm{{in}}}} = d_{{\\mathrm{{out}}}} = {din},\\ r = {r}", f"\\text{{full}} = {full},\\quad \\text{{LoRA}} = r(d_{{\\mathrm{{in}}}} + d_{{\\mathrm{{out}}}}) = {lora},\\quad \\text{{ratio}} = {ft(F(lora, full))} \\approx 0{{,}}{round(lora / full * 10000):04d}") + "\n\nLoRA chỉ huấn luyện khoảng $0{,}4\\%$ số tham số của ma trận gốc."


@lesson("7.7")
def lora_merge_example():
    w = [[F(1), F(2)], [F(0), F(-1)]]
    b = [[F(1)], [F(2)]]
    a = [[F(2), F(-1)]]
    r, alpha = 1, F(4)
    delta = mm(b, a)
    out = [[w[i][j] + alpha / r * delta[i][j] for j in range(2)] for i in range(2)]
    x = [F(1), F(2)]
    lhs = mv(out, x)
    rhs = [p + alpha / r * q for p, q in zip(mv(w, x), mv(b, mv(a, x)))]
    check(lhs == rhs, "gộp LoRA không đổi kết quả")
    return display("\\mathbf{W} = " + mat_tex(w) + ",\\ \\mathbf{B} = " + mat_tex(b) + ",\\ \\mathbf{A} = " + mat_tex(a) + f",\\ \\alpha/r = {ft(alpha / r)}", "\\mathbf{W}' = \\mathbf{W} + \\dfrac{\\alpha}{r}\\mathbf{B}\\mathbf{A} = " + mat_tex(out),
                   f"\\mathbf{{W}}'\\mathbf{{x}} = {vt(lhs)} = \\mathbf{{W}}\\mathbf{{x}} + \\dfrac{{\\alpha}}{{r}}\\mathbf{{B}}(\\mathbf{{A}}\\mathbf{{x}})") + "\n\nSau huấn luyện có thể gộp $\\mathbf{W}'$ nên suy luận không chậm hơn."


@lesson("7.7")
def torch_shape_example():
    B, T, h, dk = 2, 5, 2, 4
    D = h * dk
    x = np.zeros((B, T, D))
    q = x.reshape(B, T, h, dk).transpose(0, 2, 1, 3)
    s = q @ q.transpose(0, 1, 3, 2)
    out = (s @ q).transpose(0, 2, 1, 3).reshape(B, T, D)
    code = (f"x = torch.randn({B}, {T}, {D})                                  # {x.shape}\n"
            f"q = x.view({B}, {T}, {h}, {dk}).transpose(1, 2)                   # {q.shape}\n"
            f"scores = q @ q.transpose(-2, -1)                          # {s.shape}\n"
            f"out = (scores @ q).transpose(1, 2).reshape({B}, {T}, {D})        # {out.shape}")
    return f"```python\n{code}\n```\n\nCác shape ở chú thích do NumPy tính trên cùng dãy phép biến đổi."


@lesson("7.7")
def mistake_freeze():
    return "**Sai:** fine-tune toàn bộ mô hình lớn với learning rate của huấn luyện từ đầu: dễ phá vỡ kiến thức đã học (catastrophic forgetting).\n\n**Đúng:** dùng learning rate nhỏ hơn nhiều, hoặc đóng băng phần lớn tham số và chỉ huấn luyện một số ít (LoRA, adapter, chỉ lớp cuối)."


def build():
    paths = []
    for code, examples in EX.items():
        paths += write_examples(code, examples)
    return paths
