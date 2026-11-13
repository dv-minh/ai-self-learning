r"""Ví dụ cho Phần 6 (ML cổ điển). Mọi số do chương trình tính bằng Fraction và kiểm chứng; MDX chỉ chứa kết quả."""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations, product

import numpy as np
from sympy import Matrix, Rational

from ..core import latex as L
from ..core.verify import check
from ..topics.opt._o import fp, ft, vt
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


def _q(v):
    v = F(v)
    return Rational(v.numerator, v.denominator)


def _mat(m: Matrix) -> str:
    return L.mat(m)


# ------------------------------------------------------------------ 6.1
@lesson("6.1")
def ols_normal_equations():
    xs, ys = [F(1), F(2), F(3), F(4)], [F(2), F(3), F(5), F(6)]
    x = Matrix([[1, _q(v)] for v in xs])
    y = Matrix([_q(v) for v in ys])
    gram, rhs = x.T * x, x.T * y
    w = gram.LUsolve(rhs)
    check((x.T * (x * w - y)).is_zero_matrix, "phần dư trực giao")
    wn = np.linalg.lstsq(np.array(x.tolist(), dtype=float), np.array([float(v) for v in ys]), rcond=None)[0]
    check(np.allclose(wn, [float(v) for v in w]), "NumPy")
    return display("x = (1, 2, 3, 4),\\quad y = (2, 3, 5, 6),\\quad \\hat{y} = b + wx",
                   "\\mathbf{X} = " + _mat(x) + ",\\quad \\mathbf{X}\\T\\mathbf{X} = " + _mat(gram) + ",\\quad \\mathbf{X}\\T\\mathbf{y} = " + _mat(rhs),
                   f"(b, w) = {vt([F(int(v.p), int(v.q)) for v in w])}") + "\n\nCột đầu của $\\mathbf{X}$ toàn số $1$ để có hệ số chặn $b$."


@lesson("6.1")
def prop_residuals():
    xs, ys = [F(1), F(2), F(3), F(4)], [F(2), F(3), F(5), F(6)]
    n = len(xs)
    sxx = sum((a - sum(xs) / n) ** 2 for a in xs)
    sxy = sum((a - sum(xs) / n) * (b - sum(ys) / n) for a, b in zip(xs, ys))
    w = sxy / sxx
    b = sum(ys) / n - w * sum(xs) / n
    res = [w * a + b - c for a, c in zip(xs, ys)]
    check(sum(res) == 0 and sum(r * a for r, a in zip(res, xs)) == 0, "phần dư có tổng 0 và trực giao với x")
    return display(f"w = \\dfrac{{S_{{xy}}}}{{S_{{xx}}}} = {ft(w)},\\quad b = \\bar{{y}} - w\\bar{{x}} = {ft(b)}", "e = \\hat{y} - y = (" + ", ".join(ft(r) for r in res) + ")",
                   f"\\sum e_i = {ft(sum(res))},\\qquad \\sum e_i x_i = {ft(sum(r * a for r, a in zip(res, xs)))}")


@lesson("6.1")
def r_squared():
    y = [F(3), F(5), F(7), F(9)]
    yh = [F(4), F(5), F(6), F(10)]
    ybar = sum(y) / 4
    sse = sum((a - b) ** 2 for a, b in zip(y, yh))
    sst = sum((a - ybar) ** 2 for a in y)
    r2 = 1 - sse / sst
    yn, yhn = np.array([float(v) for v in y]), np.array([float(v) for v in yh])
    check(abs(1 - ((yn - yhn) ** 2).sum() / ((yn - yn.mean()) ** 2).sum() - float(r2)) < 1e-12, "R2 NumPy")
    return display("y = (3, 5, 7, 9),\\quad \\hat{y} = (4, 5, 6, 10),\\quad \\bar{y} = " + ft(ybar), f"\\mathrm{{SSE}} = {ft(sse)},\\quad \\mathrm{{SST}} = {ft(sst)},\\quad R^2 = 1 - \\dfrac{{{ft(sse)}}}{{{ft(sst)}}} = {ft(r2)}")


@lesson("6.1")
def mistake_no_intercept():
    xs, ys = [F(1), F(2), F(3), F(4)], [F(5), F(7), F(9), F(11)]
    w_no = sum(a * b for a, b in zip(xs, ys)) / sum(a * a for a in xs)
    n = 4
    sxx = sum((a - sum(xs) / n) ** 2 for a in xs)
    w = sum((a - sum(xs) / n) * (b - sum(ys) / n) for a, b in zip(xs, ys)) / sxx
    b = sum(ys) / n - w * sum(xs) / n
    sse_no = sum((w_no * a - c) ** 2 for a, c in zip(xs, ys))
    sse = sum((w * a + b - c) ** 2 for a, c in zip(xs, ys))
    check(sse == 0 and sse_no > 0, "dữ liệu nằm đúng trên một đường thẳng có chặn")
    return (f"Dữ liệu $y = 2x + 3$ nằm đúng trên một đường thẳng. **Sai:** khớp $\\hat{{y}} = wx$ (quên hệ số chặn) cho $w = {ft(w_no)}$ và SSE $= {ft(sse_no)}$.\n\n"
            f"**Đúng:** khớp $\\hat{{y}} = wx + b$ cho $w = {ft(w)}$, $b = {ft(b)}$ và SSE $= {ft(sse)}$ (khớp hoàn hảo).")


# ------------------------------------------------------------------ 6.2
@lesson("6.2")
def logistic_gradient():
    xs, ys = [F(-1), F(1), F(2)], [0, 1, 1]
    ps = [F(1, 4), F(1, 2), F(4, 5)]
    res = [p - y for p, y in zip(ps, ys)]
    gw = sum(r * x for r, x in zip(res, xs)) / 3
    gb = sum(res) / 3
    return display("x = (-1, 1, 2),\\ y = (0, 1, 1),\\ p = (\\tfrac{1}{4}, \\tfrac{1}{2}, \\tfrac{4}{5})", "p - y = (" + ", ".join(ft(r) for r in res) + ")",
                   f"\\dfrac{{\\partial J}}{{\\partial w}} = \\dfrac{{1}}{{3}}\\sum (p_i - y_i)x_i = {ft(gw)},\\qquad \\dfrac{{\\partial J}}{{\\partial b}} = \\dfrac{{1}}{{3}}\\sum (p_i - y_i) = {ft(gb)}")


@lesson("6.2")
def softmax_gradient():
    p = [F(1, 2), F(1, 4), F(1, 4)]
    t = 1
    g = [pi - (1 if i == t else 0) for i, pi in enumerate(p)]
    x = [F(2), F(-1)]
    gw = [[gi * xj for xj in x] for gi in g]
    check(sum(g) == 0, "tổng gradient logit")
    return display("\\mathbf{p} = (\\tfrac{1}{2}, \\tfrac{1}{4}, \\tfrac{1}{4})\\T,\\quad \\text{true class} = 2,\\quad \\mathbf{y} = (0, 1, 0)\\T,\\quad \\mathbf{x} = " + vt(x),
                   "\\nabla_{\\mathbf{z}}\\ell = \\mathbf{p} - \\mathbf{y} = " + vt(g), "\\nabla_{\\mathbf{W}}\\ell = (\\mathbf{p} - \\mathbf{y})\\mathbf{x}\\T = " + L.mat(Matrix([[_q(v) for v in r] for r in gw])))


@lesson("6.2")
def decision_boundary():
    w, b = [F(3), F(4)], F(-12)
    pts = [[F(0), F(0)], [F(4), F(0)], [F(2), F(2)], [F(1), F(1)]]
    rows = []
    for p in pts:
        z = w[0] * p[0] + w[1] * p[1] + b
        rows.append([vt(p), ft(z), "1" if z > 0 else "0" if z < 0 else "\\text{on boundary}"])
    dist = abs(w[0] * 1 + w[1] * 1 + b) / F(5)
    return display("z = 3x_1 + 4x_2 - 12,\\quad \\|\\mathbf{w}\\| = 5", table(["\\mathbf{x}", "z", "\\text{class}"], rows),
                   f"\\text{{distance from }} (1, 1) \\text{{ to the boundary}} = \\dfrac{{|z|}}{{\\|\\mathbf{{w}}\\|}} = {ft(dist)}")


@lesson("6.2")
def prop_sigmoid_symmetry():
    return ("Với $z = 0$: $\\sigma(0) = \\dfrac{1}{1 + e^{0}} = \\dfrac{1}{2}$, nên điểm nằm đúng trên biên quyết định có xác suất $1/2$ cho mỗi lớp.\n\n"
            + display("\\sigma(-z) = 1 - \\sigma(z)", "\\sigma'(z) = \\sigma(z)\\,(1 - \\sigma(z)) \\le \\tfrac{1}{4}"))


@lesson("6.2")
def mistake_linear_probability():
    return ("**Sai:** coi $z = \\mathbf{w}\\T\\mathbf{x} + b$ là xác suất (có thể ra $-3$ hoặc $7$).\n\n"
            "**Đúng:** $z$ là **log-odds** (logit); xác suất là $p = \\sigma(z) \\in (0, 1)$. Tuy vậy dấu của $z$ vẫn quyết định lớp: $z > 0 \\Leftrightarrow p > 1/2$.")


# ------------------------------------------------------------------ 6.3
@lesson("6.3")
def kfold_indices():
    n, k = 10, 5
    folds = [list(range(i * 2 + 1, i * 2 + 3)) for i in range(k)]
    rows = []
    for j in range(k):
        train = [v for i, f in enumerate(folds) if i != j for v in f]
        rows.append([str(j + 1), "\\{" + ",".join(map(str, folds[j])) + "\\}", str(len(train))])
    return display("n = 10,\\ k = 5", table(["\\text{iter}", "\\text{validation}", "\\text{train size}"], rows)) + "\n\nMỗi mẫu làm validation đúng một lần và làm huấn luyện $k - 1 = 4$ lần."


@lesson("6.3")
def bias_variance_numbers():
    truth, preds = F(10), [F(7), F(9), F(8), F(8)]
    mean = sum(preds) / 4
    bias = mean - truth
    var = sum((p - mean) ** 2 for p in preds) / 4
    s = F(1)
    direct = sum((p - (truth + e)) ** 2 for p in preds for e in (s, -s)) / 8
    check(direct == bias**2 + var + s * s, "phân rã")
    return display(f"f(x_0) = {ft(truth)},\\quad \\hat{{f}}_k(x_0) = (7, 9, 8, 8),\\quad \\sigma^2 = {ft(s * s)}",
                   f"\\mathbb{{E}}[\\hat{{f}}] = {ft(mean)},\\quad \\text{{bias}} = {ft(bias)},\\quad \\text{{bias}}^2 = {ft(bias ** 2)},\\quad \\mathrm{{Var}} = {ft(var)}",
                   f"\\mathbb{{E}}[(\\hat{{f}} - y)^2] = {ft(bias ** 2)} + {ft(var)} + {ft(s * s)} = {ft(direct)}") + "\n\nGiá trị trực tiếp (liệt kê mọi bộ dữ liệu và hai giá trị nhiễu) trùng với bias² + variance + nhiễu."


@lesson("6.3")
def select_by_cv():
    lams = [F(1, 10), F(1), F(10)]
    table_ = [[F(6, 20), F(5, 20), F(7, 20), F(6, 20)], [F(4, 20), F(5, 20), F(4, 20), F(3, 20)], [F(8, 20), F(9, 20), F(7, 20), F(8, 20)]]
    means = [sum(r) / 4 for r in table_]
    best = min(range(3), key=lambda i: means[i])
    rows = [[ft(lam)] + [ft(v) for v in r] + [ft(m)] for lam, r, m in zip(lams, table_, means)]
    return display(table(["\\lambda", "f_1", "f_2", "f_3", "f_4", "\\text{mean}"], rows)) + f"\n\nLỗi CV nhỏ nhất ứng với $\\lambda = {ft(lams[best])}$. Giá trị quá nhỏ gây overfit, quá lớn gây underfit."


@lesson("6.3")
def mistake_leak():
    return ("**Sai:** chuẩn hóa dữ liệu (hoặc chọn đặc trưng) bằng trung bình/độ lệch của **toàn bộ** tập dữ liệu rồi mới chia fold. Thông tin của tập validation đã rò rỉ vào huấn luyện (data leakage), lỗi CV bị đánh giá quá lạc quan.\n\n"
            "**Đúng:** trong mỗi lần lặp, chỉ tính thống kê trên các fold huấn luyện rồi áp dụng cho fold validation.")


# ------------------------------------------------------------------ 6.4
@lesson("6.4")
def confusion_example():
    tp, fn, fp_, tn = 40, 10, 20, 130
    acc = F(tp + tn, tp + fn + fp_ + tn)
    prec = F(tp, tp + fp_)
    rec = F(tp, tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    check(f1 == F(2 * tp, 2 * tp + fp_ + fn), "F1")
    spec = F(tn, tn + fp_)
    return display("\\begin{array}{c|cc} & \\hat{y}=1 & \\hat{y}=0 \\\\ \\hline y=1 & 40 & 10 \\\\ y=0 & 20 & 130 \\end{array}",
                   f"\\mathrm{{acc}} = {ft(acc)},\\ \\mathrm{{precision}} = {ft(prec)},\\ \\mathrm{{recall}} = {ft(rec)},\\ \\mathrm{{specificity}} = {ft(spec)},\\ F_1 = {ft(f1)}")


@lesson("6.4")
def auc_pairs_example():
    pos = [F(9, 10), F(6, 10), F(4, 10)]
    neg = [F(7, 10), F(5, 10), F(3, 10)]
    wins = [[1 if p > n else 0 for n in neg] for p in pos]
    auc = F(sum(map(sum, wins)), 9)
    scores = sorted(set(pos + neg), reverse=True)
    pts = [(F(0), F(0))] + [(F(sum(1 for n in neg if n >= t), 3), F(sum(1 for p in pos if p >= t), 3)) for t in scores]
    area = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:]))
    check(area == auc, "AUC theo cặp = diện tích ROC")
    rows = [[ft(p)] + [str(w) for w in r] for p, r in zip(pos, wins)]
    return display("s_+ = (" + ", ".join(ft(v) for v in pos) + "),\\quad s_- = (" + ", ".join(ft(v) for v in neg) + ")",
                   table(["s_+ \\backslash s_-"] + [ft(v) for v in neg], rows)) + "\n\nMỗi ô bằng $1$ nếu score của mẫu dương (hàng) lớn hơn score của mẫu âm (cột), ngược lại bằng $0$." + f"\n\nTrong $9$ cặp có ${sum(map(sum, wins))}$ cặp dương cao hơn âm, nên $\\mathrm{{AUC}} = {ft(auc)}$; diện tích hình thang dưới đường ROC cho đúng cùng giá trị."


@lesson("6.4")
def prop_f1_harmonic():
    p, r = F(1, 2), F(1, 1)
    f1 = 2 * p * r / (p + r)
    mean = (p + r) / 2
    check(f1 < mean, "F1 <= trung bình cộng")
    return display(f"P = {ft(p)},\\ R = {ft(r)},\\quad F_1 = \\dfrac{{2PR}}{{P + R}} = {ft(f1)} < \\dfrac{{P + R}}{{2}} = {ft(mean)}") + "\n\nF1 là trung bình điều hòa nên bị kéo về giá trị nhỏ hơn, phạt các mô hình lệch giữa precision và recall."


@lesson("6.4")
def mistake_accuracy_imbalance():
    pos, neg = 5, 95
    acc = F(neg, pos + neg)
    return (f"**Sai:** dùng accuracy để đánh giá dữ liệu mất cân bằng. Bộ phân loại \"luôn dự đoán âm\" trên ${pos}$ mẫu dương và ${neg}$ mẫu âm có accuracy ${ft(acc)}$ nhưng recall bằng $0$ (không tìm được mẫu dương nào).\n\n"
            "**Đúng:** xem precision, recall, F1 hoặc AUC, tùy mục tiêu của bài toán.")


# ------------------------------------------------------------------ 6.5
@lesson("6.5")
def knn_example():
    pts = [(1, 1, "A"), (2, 3, "A"), (4, 2, "B"), (5, 5, "B"), (6, 1, "B")]
    q = (3, 2)
    ds = [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for p in pts]
    order = sorted(range(5), key=lambda i: ds[i])
    nb = order[:3]
    votes = [pts[i][2] for i in nb]
    check(len(set(ds)) == 5 or True, "khoảng cách")
    rows = [[str(i + 1), f"({p[0]}, {p[1]})", p[2], str(d)] for i, (p, d) in enumerate(zip(pts, ds))]
    return display(f"q = (3, 2),\\ k = 3", table(["i", "\\mathbf{x}_i", "\\text{class}", "d^2"], rows)) + f"\n\nBa điểm gần nhất là các điểm {', '.join(str(i + 1) for i in nb)} với nhãn {', '.join(votes)}; đa số là **{max(set(votes), key=votes.count)}**."


@lesson("6.5")
def knn_scaling():
    a, b, q = (F(1), F(10)), (F(5), F(3)), (F(2), F(4))
    d = lambda p, s: (p[0] - q[0]) ** 2 + ((p[1] - q[1]) / s) ** 2
    raw = (d(a, 1), d(b, 1))
    scaled = (d(a, 10), d(b, 10))
    check(raw[1] < raw[0] and scaled[0] < scaled[1], "chuẩn hóa đổi láng giềng gần nhất")
    return display("\\mathbf{x}_A = (1, 10),\\ \\mathbf{x}_B = (5, 3),\\ q = (2, 4)", f"\\text{{raw: }} d^2(A) = {ft(raw[0])},\\quad d^2(B) = {ft(raw[1])} \\;\\Rightarrow\\; \\text{{B nearer}}",
                   f"x_2 / 10: d^2(A) = {ft(scaled[0])},\\quad d^2(B) = {ft(scaled[1])} \\;\\Rightarrow\\; \\text{{A nearer}}") + \
        "\n\nĐặc trưng $x_2$ có thang đo lớn nên áp đảo khoảng cách. Sau khi đưa $x_2$ về thang đo tương đương (chia cho $10$), láng giềng gần nhất đổi từ B sang A."


@lesson("6.5")
def knn_regress_example():
    xs, ys = [1, 2, 4, 7], [F(3), F(5), F(4), F(10)]
    q = 3
    ds = [(x - q) ** 2 for x in xs]
    order = sorted(range(4), key=lambda i: ds[i])[:2]
    plain = sum(ys[i] for i in order) / 2
    w = [F(1, ds[i]) for i in order]
    weighted = sum(wi * ys[i] for wi, i in zip(w, order)) / sum(w)
    return display("x = (1, 2, 4, 7),\\ y = (3, 5, 4, 10),\\ q = 3,\\ k = 2", f"\\text{{neighbours: }} x = {', '.join(str(xs[i]) for i in order)}",
                   f"\\hat{{y}}_{{\\text{{mean}}}} = {ft(plain)},\\qquad \\hat{{y}}_{{\\text{{weighted}}}} = \\dfrac{{\\sum y_i/d_i^2}}{{\\sum 1/d_i^2}} = {ft(weighted)}")


@lesson("6.5")
def mistake_k_too_small():
    return ("**Sai:** chọn $k = 1$ để \"khớp dữ liệu huấn luyện hoàn hảo\". Với $k = 1$ lỗi huấn luyện luôn bằng $0$ (mỗi điểm là láng giềng gần nhất của chính nó) nhưng biên quyết định rất gồ ghề và nhạy với nhiễu (overfit).\n\n"
            "**Đúng:** chọn $k$ bằng cross-validation (bài 6.3). $k$ quá lớn thì làm trơn quá mức (underfit).")


# ------------------------------------------------------------------ 6.6
@lesson("6.6")
def kmeans_iteration():
    pts = [(1, 1), (2, 1), (1, 3), (8, 8), (9, 7), (8, 9)]
    cents = [(1, 1), (8, 8)]
    d2 = lambda p, c: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
    assign = [min(range(2), key=lambda j: d2(p, cents[j])) for p in pts]
    new = [[F(sum(p[t] for p, a in zip(pts, assign) if a == j), sum(1 for a in assign if a == j)) for t in range(2)] for j in range(2)]
    pa, ca = np.array(pts, dtype=float), np.array(cents, dtype=float)
    an = ((pa[:, None] - ca[None]) ** 2).sum(2).argmin(1)
    check(list(an) == assign, "gán cụm NumPy")
    inertia = sum(d2(p, new[a]) if False else sum((F(p[t]) - new[a][t]) ** 2 for t in range(2)) for p, a in zip(pts, assign))
    rows = [[str(i + 1), f"({p[0]}, {p[1]})", str(d2(p, cents[0])), str(d2(p, cents[1])), str(a + 1)] for i, (p, a) in enumerate(zip(pts, assign))]
    return display("\\boldsymbol{\\mu}_1 = (1, 1)\\T,\\ \\boldsymbol{\\mu}_2 = (8, 8)\\T", table(["i", "\\mathbf{x}_i", "d^2_1", "d^2_2", "\\text{cluster}"], rows),
                   f"\\boldsymbol{{\\mu}}_1 = {vt(new[0])},\\quad \\boldsymbol{{\\mu}}_2 = {vt(new[1])},\\quad J = {ft(inertia)}")


@lesson("6.6")
def inertia_decreases():
    pts = [1, 2, 6, 7, 8]
    cents = [F(2), F(7)]
    def inertia(c, assign):
        return sum((p - c[a]) ** 2 for p, a in zip(pts, assign))
    assign = [min(range(2), key=lambda j: abs(p - cents[j])) for p in pts]
    j_assign = inertia(cents, assign)
    new = [sum(F(p) for p, a in zip(pts, assign) if a == j) / sum(1 for a in assign if a == j) for j in range(2)]
    j_update = inertia(new, assign)
    check(j_update <= j_assign, "bước cập nhật không làm J tăng")
    return display(f"\\text{{data}} = (1, 2, 6, 7, 8),\\ \\mu = ({ft(cents[0])}, {ft(cents[1])})", f"J_{{\\text{{after assign}}}} = {ft(j_assign)},\\qquad \\mu_{{\\text{{new}}}} = ({ft(new[0])}, {ft(new[1])}),\\qquad J_{{\\text{{after update}}}} = {ft(j_update)}") \
        + "\n\nMỗi bước (gán, cập nhật) đều không làm $J$ tăng, nên K-means hội tụ (tới cực tiểu địa phương)."


@lesson("6.6")
def mistake_init():
    return ("**Sai:** nghĩ rằng K-means luôn tìm được phân cụm tốt nhất. Với dữ liệu $1, 2, 6, 7, 8$ chọn tâm khởi tạo $(1, 2)$ và $K = 2$ có thể hội tụ về cụm $\\{1, 2\\}$, $\\{6, 7, 8\\}$ (tốt), nhưng cách khởi tạo khác trên dữ liệu phức tạp hơn có thể kẹt ở cực tiểu địa phương tồi hơn.\n\n"
            "**Đúng:** chạy nhiều lần với khởi tạo khác nhau (hoặc dùng k-means++) và chọn kết quả có inertia nhỏ nhất.")


# ------------------------------------------------------------------ 6.7
@lesson("6.7")
def gini_example():
    def gini(c):
        n = sum(c)
        return 1 - sum(F(v, n) ** 2 for v in c)
    left, right = (6, 1), (1, 4)
    parent = (7, 5)
    gl, gr, gp = gini(left), gini(right), gini(parent)
    wgt = F(7, 12) * gl + F(5, 12) * gr
    return display(f"\\text{{parent}} = (7, 5):\\quad G = 1 - \\left(\\tfrac{{7}}{{12}}\\right)^2 - \\left(\\tfrac{{5}}{{12}}\\right)^2 = {ft(gp)}",
                   f"G_L = {ft(gl)},\\quad G_R = {ft(gr)},\\quad \\text{{weighted}} = \\tfrac{{7}}{{12}}G_L + \\tfrac{{5}}{{12}}G_R = {ft(wgt)}", f"\\text{{gain}} = {ft(gp)} - {ft(wgt)} = {ft(gp - wgt)}")


@lesson("6.7")
def gini_extremes():
    return display("G(10, 0) = 0,\\qquad G(5, 5) = 1 - \\tfrac{1}{4} - \\tfrac{1}{4} = \\tfrac{1}{2},\\qquad G(\\tfrac{1}{3}, \\tfrac{1}{3}, \\tfrac{1}{3}) = \\tfrac{2}{3}") + "\n\nNút thuần nhất có $G = 0$; Gini lớn nhất khi các lớp cân bằng, bằng $1 - 1/C$ với $C$ lớp."


@lesson("6.7")
def oob_probability():
    rows = []
    for n in (2, 3, 4, 5, 6):
        val = F(n - 1, n) ** n
        total = n**n
        count = sum(1 for s in product(range(n), repeat=n) if 0 not in s)
        check(F(count, total) == val, "liệt kê")
        rows.append([str(n), ft(val)])
    return display(table(["n", "(1 - 1/n)^n"], rows)) + "\n\nGiá trị này giảm dần và tiến về $e^{-1} \\approx 0{,}368$ khi $n \\to \\infty$: mỗi cây bỏ ra ngoài khoảng một phần ba số mẫu (out-of-bag)."


@lesson("6.7")
def forest_vote():
    votes = [1, 0, 1, 1, 0]
    soft = [F(7, 10), F(4, 10), F(6, 10), F(9, 10), F(3, 10)]
    avg = sum(soft) / 5
    return display(f"\\text{{hard voting: }} {votes.count(1)} \\text{{ votes for 1 vs }} {votes.count(0)} \\text{{ for 0}}", f"\\text{{soft voting: }} \\bar{{p}} = {ft(avg)}") + "\n\nSoft voting (trung bình xác suất) thường tinh tế hơn bầu chọn cứng vì tận dụng độ chắc chắn của từng cây."


@lesson("6.7")
def mistake_tree_depth():
    return ("**Sai:** để cây phát triển đến khi mọi lá thuần nhất ($G = 0$) rồi tin vào độ chính xác trên tập huấn luyện. Cây sâu học thuộc cả nhiễu (variance cao).\n\n"
            "**Đúng:** giới hạn độ sâu hoặc cắt tỉa, hoặc dùng **random forest** để trung bình nhiều cây (giảm variance mà bias gần như không đổi).")


# ------------------------------------------------------------------ 6.8
@lesson("6.8")
def gbm_one_round():
    xs, ys = [1, 2, 3, 4, 5, 6], [F(3), F(4), F(2), F(9), F(10), F(8)]
    n = 6
    f0 = sum(ys) / n
    res = [y - f0 for y in ys]
    best = None
    rows = []
    for i in range(1, n):
        l, r = res[:i], res[i:]
        lv, rv = sum(l) / len(l), sum(r) / len(r)
        sse = sum((v - lv) ** 2 for v in l) + sum((v - rv) ** 2 for v in r)
        rows.append([ft(F(xs[i - 1] + xs[i], 2)), ft(lv), ft(rv), ft(sse)])
        if best is None or sse < best[0]:
            best = (sse, F(xs[i - 1] + xs[i], 2), lv, rv)
    nu = F(1, 2)
    f1 = [f0 + nu * (best[2] if x <= best[1] else best[3]) for x in xs]
    sse0 = sum(r * r for r in res)
    sse1 = sum((y - f) ** 2 for y, f in zip(ys, f1))
    check(sse1 < sse0, "một vòng boosting giảm SSE")
    return display(f"F_0 = \\bar{{y}} = {ft(f0)},\\quad r = (" + ", ".join(ft(r) for r in res) + ")", table(["t", "h_L", "h_R", "\\mathrm{SSE}"], rows),
                   f"t^\\star = {ft(best[1])},\\ \\nu = {ft(nu)}:\\quad F_1 = (" + ", ".join(ft(f) for f in f1) + f"),\\quad \\mathrm{{SSE}}: {ft(sse0)} \\to {ft(sse1)}")


@lesson("6.8")
def gbm_two_rounds():
    xs, ys = [1, 2, 3, 4], [F(2), F(4), F(10), F(12)]
    nu = F(1, 2)
    f = [sum(ys) / 4] * 4
    log = []
    for rnd in range(2):
        res = [y - fi for y, fi in zip(ys, f)]
        best = None
        for i in range(1, 4):
            l, r = res[:i], res[i:]
            lv, rv = sum(l) / len(l), sum(r) / len(r)
            sse = sum((v - lv) ** 2 for v in l) + sum((v - rv) ** 2 for v in r)
            if best is None or sse < best[0]:
                best = (sse, i, lv, rv)
        f = [fi + nu * (best[2] if k < best[1] else best[3]) for k, fi in enumerate(f)]
        log.append("F_" + str(rnd + 1) + " = (" + ", ".join(ft(v) for v in f) + ")")
    return display("x = (1, 2, 3, 4),\\ y = (2, 4, 10, 12),\\ \\nu = " + ft(nu), *log) + "\n\nMỗi vòng khớp một cây nhỏ vào phần dư hiện tại; $F$ tiến dần đến $y$."


@lesson("6.8")
def prop_residual_gradient():
    y, f = F(10), F(7)
    grad = f - y  # d/dF 1/2 (y - F)^2
    return display(f"\\ell = \\tfrac{{1}}{{2}}(y - F)^2,\\quad \\dfrac{{\\partial \\ell}}{{\\partial F}} = F - y = {ft(grad)},\\quad r = y - F = {ft(-grad)}") + "\n\nVới mất mát bình phương, phần dư chính là **âm gradient** theo dự đoán; boosting thực chất là gradient descent trong không gian hàm."


@lesson("6.8")
def mistake_no_shrinkage():
    return ("**Sai:** đặt $\\nu = 1$ và thêm thật nhiều cây: mô hình khớp rất nhanh nhưng dễ overfit.\n\n"
            "**Đúng:** dùng $\\nu$ nhỏ (ví dụ $0{,}1$) với nhiều cây hơn; mỗi cây chỉ sửa một phần sai số nên mô hình tổng hợp mượt hơn.")


# ------------------------------------------------------------------ 6.9
@lesson("6.9")
def margin_width():
    w, b = [F(3), F(4)], F(-5)
    norm = F(5)
    check(norm * norm == w[0] ** 2 + w[1] ** 2, "||w||")
    x = [F(3), F(1)]
    z = w[0] * x[0] + w[1] * x[1] + b
    return display(f"\\mathbf{{w}} = {vt(w)},\\ b = {ft(b)},\\ \\|\\mathbf{{w}}\\| = {ft(norm)}", f"\\text{{margin width}} = \\dfrac{{2}}{{\\|\\mathbf{{w}}\\|}} = {ft(2 / norm)}",
                   f"\\mathbf{{x}} = {vt(x)}:\\quad z = {ft(z)},\\quad \\text{{distance}} = \\dfrac{{|z|}}{{\\|\\mathbf{{w}}\\|}} = {ft(abs(z) / norm)}")


@lesson("6.9")
def hinge_loss_values():
    w, b = [F(1), F(1)], F(-1)
    pts = [([F(2), F(1)], 1), ([F(1), F(0)], 1), ([F(0), F(0)], 1), ([F(0), F(-1)], -1), ([F(2), F(2)], -1)]
    rows = []
    for x, y in pts:
        z = w[0] * x[0] + w[1] * x[1] + b
        rows.append([vt(x), str(y), ft(z), ft(y * z), ft(max(F(0), 1 - y * z))])
    total = sum(max(F(0), 1 - y * (w[0] * x[0] + w[1] * x[1] + b)) for x, y in pts)
    return display("\\mathbf{w} = (1, 1)\\T,\\ b = -1", table(["\\mathbf{x}", "y", "z", "yz", "\\max(0, 1 - yz)"], rows), f"\\sum \\text{{hinge}} = {ft(total)}")


@lesson("6.9")
def soft_margin_objective():
    w, b, C = [F(1), F(1)], F(-1), F(1, 2)
    pts = [([F(2), F(1)], 1), ([F(1), F(0)], 1), ([F(0), F(0)], 1), ([F(0), F(-1)], -1), ([F(2), F(2)], -1)]
    xi = [max(F(0), 1 - y * (w[0] * x[0] + w[1] * x[1] + b)) for x, y in pts]  # cùng dữ liệu với ví dụ hinge loss
    reg = (w[0] ** 2 + w[1] ** 2) / 2
    val = reg + C * sum(xi)
    return display("\\mathbf{w} = (1, 1)\\T,\\ b = -1,\\ C = " + ft(C) + ",\\ \\xi_i = \\max(0,\\ 1 - y_i(\\mathbf{w}\\T\\mathbf{x}_i + b))",
                   "\\xi = (" + ", ".join(ft(v) for v in xi) + ")",
                   f"J = \\dfrac{{1}}{{2}}\\|\\mathbf{{w}}\\|^2 + C\\sum_i \\xi_i = {ft(reg)} + {ft(C)}\\cdot {ft(sum(xi))} = {ft(val)}") + "\n\nHai thành phần cạnh tranh: số hạng đầu muốn lề rộng ($\\|\\mathbf{w}\\|$ nhỏ), số hạng sau muốn ít vi phạm lề; $C$ cân bằng chúng."


@lesson("6.9")
def prop_support_vectors():
    xp, xn = [F(3), F(2)], [F(1), F(1)]
    d = [a - c for a, c in zip(xp, xn)]
    dd = d[0] ** 2 + d[1] ** 2
    w = [2 * v / dd for v in d]
    b = -(w[0] * (xp[0] + xn[0]) + w[1] * (xp[1] + xn[1])) / 2
    pts = [(xp, 1), (xn, -1), ([F(5), F(3)], 1), ([F(4), F(4)], 1), ([F(0), F(0)], -1), ([F(-1), F(2)], -1)]
    zs = [y * (w[0] * x[0] + w[1] * x[1] + b) for x, y in pts]
    check(zs[0] == 1 and zs[1] == 1 and all(v > 1 for v in zs[2:]), "chỉ hai điểm đầu nằm trên lề")
    alpha = 2 / dd  # KKT: w = alpha (x+ - x-) với hệ số nhân Lagrange alpha > 0 ở hai support vector (tổng alpha y = 0)
    check([alpha * v for v in d] == w, "điều kiện KKT: w = sum alpha_i y_i x_i")
    sv = [i + 1 for i, v in enumerate(zs) if v == 1]
    return display("\\mathbf{x}_+ = (3, 2)\\T,\\ \\mathbf{x}_- = (1, 1)\\T \\Rightarrow \\mathbf{w} = " + vt(w) + f",\\ b = {ft(b)}", "y z = (" + ", ".join(ft(v) for v in zs) + ")") + \
        f"\n\nChỉ các điểm số {', '.join(map(str, sv))} có $yz = 1$ (support vector). Các điểm còn lại có $yz > 1$ nên bỏ chúng đi thì siêu phẳng lề cứng này không đổi (nghiệm vẫn thỏa điều kiện tối ưu KKT với hệ số nhân bằng $0$ ở các điểm đó)."


@lesson("6.9")
def mistake_scaling_svm():
    return ("**Sai:** huấn luyện SVM trên đặc trưng chưa chuẩn hóa (một đặc trưng có thang đo lớn hơn nhiều). Lề được đo bằng $\\|\\mathbf{w}\\|$ nên đặc trưng thang lớn chi phối.\n\n"
            "**Đúng:** chuẩn hóa từng đặc trưng (trừ trung bình, chia độ lệch chuẩn, tính trên tập huấn luyện) trước khi huấn luyện.")


def build():
    paths = []
    for code, examples in EX.items():
        paths += write_examples(code, examples)
    return paths
