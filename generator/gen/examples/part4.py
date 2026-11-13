r"""Ví dụ cho Phần 4 (Xác suất – thống kê). Mọi con số là phân số chính xác, được tính bằng liệt kê hoặc công thức rồi đối chiếu."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import comb, log2

import numpy as np
from sympy import Matrix, Rational

from ..core import latex as L
from ..core.verify import check
from ..topics.prob._p import frac, pmf_array, table_array
from .common import display, write_examples

EX: dict[str, dict] = {}
R = Rational


def lesson(code: str):
    def deco(fn):
        EX.setdefault(code, {})[fn.__name__.replace("_", "-")] = fn
        return fn
    return deco


def _P(space, ev):
    return R(sum(1 for w in space if ev(w)), len(space))


D2 = list(product(range(1, 7), repeat=2))


# ------------------------------------------------------------------ 4.1
@lesson("4.1")
def dice_sum():
    counts = {s: sum(1 for a, b in D2 if a + b == s) for s in range(2, 13)}
    check(sum(counts.values()) == 36, "36 kết quả")
    tbl = table_array(["s"] + [str(s) for s in counts], [["n(s)"] + [str(c) for c in counts.values()], ["P(s)"] + [frac(R(c, 36)) for c in counts.values()]])
    return "Gọi $s$ là tổng hai mặt, $n(s)$ là số cặp cho tổng đó và $P(s) = n(s)/36$:\n\n" + display(tbl) + f"\n\nVí dụ: $P(\\text{{tổng}} = 7) = \\dfrac{{{counts[7]}}}{{36}} = {frac(R(counts[7], 36))}$, vì có {counts[7]} cặp trong $36$ cặp đồng khả năng."


@lesson("4.1")
def conditional_table():
    t = [[30, 10], [20, 40]]
    pa = R(t[0][0], sum(t[0]))
    pb = R(t[0][0], t[0][0] + t[1][0])
    return "Gọi $X$: đã xem quảng cáo (hàng $R_1$; hàng $R_2$ là chưa xem), $M$: có mua (cột $C_1$; cột $C_2$ là không mua).\n\n" + \
        display(table_array(["", "C_1", "C_2"], [["R_1", t[0][0], t[0][1]], ["R_2", t[1][0], t[1][1]]]),
                f"P(M\\mid X) = \\dfrac{{{t[0][0]}}}{{{sum(t[0])}}} = {frac(pa)},\\qquad P(X\\mid M) = \\dfrac{{{t[0][0]}}}{{{t[0][0] + t[1][0]}}} = {frac(pb)}") + \
        "\n\nHai xác suất có điều kiện này **khác nhau**: chiều điều kiện quan trọng."


@lesson("4.1")
def independence():
    ev_a = lambda w: w[0] % 2 == 0
    ev_b = lambda w: w[1] > 4
    ev_c = lambda w: w[0] + w[1] > 9
    pa, pb_, pab = _P(D2, ev_a), _P(D2, ev_b), _P(D2, lambda w: ev_a(w) and ev_b(w))
    pc, pac = _P(D2, ev_c), _P(D2, lambda w: ev_a(w) and ev_c(w))
    check(pab == pa * pb_ and pac != pa * pc, "độc lập / không độc lập")
    return display(f"P(A) = {frac(pa)},\\; P(B) = {frac(pb_)},\\; P(A\\cap B) = {frac(pab)} = P(A)P(B)") + \
        "\n\n$A$: mặt thứ nhất chẵn; $B$: mặt thứ hai $> 4$. Chúng **độc lập** vì $P(A\\cap B) = P(A)P(B)$.\n\n" + \
        display(f"P(A) = {frac(pa)},\\; P(C) = {frac(pc)},\\; P(A\\cap C) = {frac(pac)} \\ne P(A)P(C) = {frac(pa * pc)}") + "\n\nVới $C$: tổng $> 9$, ta có $A$ và $C$ **không** độc lập."


@lesson("4.1")
def prop_complement():
    ev = lambda w: w[0] == w[1]
    p = _P(D2, ev)
    return display(f"P(\\overline{{A}}) = 1 - P(A) = 1 - {frac(p)} = {frac(1 - p)}") + "\n\nVới $A$: hai mặt bằng nhau, ta đếm $6$ cặp $(k, k)$ nên $P(A) = 6/36$."


@lesson("4.1")
def prop_union():
    a, b = (lambda w: w[0] > 4), (lambda w: w[1] > 4)
    pa, pb_, pab, pu = _P(D2, a), _P(D2, b), _P(D2, lambda w: a(w) and b(w)), _P(D2, lambda w: a(w) or b(w))
    check(pu == pa + pb_ - pab, "hợp")
    return display(f"P(A\\cup B) = {frac(pa)} + {frac(pb_)} - {frac(pab)} = {frac(pu)}") + "\n\n$A$: mặt thứ nhất $> 4$; $B$: mặt thứ hai $> 4$."


@lesson("4.1")
def mistake_conditional():
    t = [[30, 10], [20, 40]]
    right = R(t[0][0], t[0][0] + t[1][0])
    wrong = R(t[0][0], sum(t[0]))
    return ("Hàng $R_1$: đã xem quảng cáo, $R_2$: chưa xem; cột $C_1$: có mua, $C_2$: không mua.\n\n"
            + display(table_array(["", "C_1", "C_2"], [["R_1", t[0][0], t[0][1]], ["R_2", t[1][0], t[1][1]]]))
            + f"\n\nCâu hỏi: trong số người **đã mua**, bao nhiêu phần đã xem quảng cáo?\n\n**Sai:** chia cho tổng hàng \"đã xem\": ${frac(wrong)}$ (đó là $P(\\text{{mua}}\\mid\\text{{đã xem}})$, ngược chiều điều kiện).\n\n"
            + f"**Đúng:** thu hẹp vào cột \"mua\" ({t[0][0] + t[1][0]} người): $\\dfrac{{{t[0][0]}}}{{{t[0][0] + t[1][0]}}} = {frac(right)}$.")


# ------------------------------------------------------------------ 4.2
@lesson("4.2")
def bayes_medical():
    N, prev, sens, fpr = 10000, R(1, 100), R(9, 10), R(5, 100)
    sick = int(N * prev)
    tp = int(sick * sens)
    fp = int((N - sick) * fpr)
    post = R(tp, tp + fp)
    check(post == sens * prev / (sens * prev + fpr * (1 - prev)), "Bayes = đếm")
    return (f"Trong ${N}$ người: ${sick}$ người bệnh, ${N - sick}$ người khỏe. Xét nghiệm dương với ${frac(sens)}$ người bệnh và ${frac(fpr)}$ người khỏe.\n\n"
            + display(f"\\text{{dương thật}} = {sick}\\cdot {frac(sens)} = {tp},\\qquad \\text{{dương giả}} = {N - sick}\\cdot {frac(fpr)} = {fp}".replace("\\text{dương thật}", "TP").replace("\\text{dương giả}", "FP"),
                       f"P(\\text{{bệnh}}\\mid +) = \\dfrac{{{tp}}}{{{tp + fp}}} = {frac(post)}".replace("\\text{bệnh}", "B")) +
            f"\n\nDù xét nghiệm \"chính xác {int(sens * 100)}\\%\", một người dương tính chỉ có khoảng ${float(post):.0%}$ khả năng thật sự bệnh, vì bệnh hiếm và có nhiều người khỏe cho dương giả.".replace("\\%", "%"))


@lesson("4.2")
def naive_bayes_example():
    rows = [(1, 1, "A"), (1, 0, "A"), (1, 1, "A"), (0, 1, "B"), (0, 0, "B"), (1, 0, "B")]
    x = (1, 1)
    score = {}
    for c in "AB":
        sub = [r for r in rows if r[2] == c]
        s = R(len(sub), len(rows))
        for j in range(2):
            s *= R(sum(1 for r in sub if r[j] == x[j]), len(sub))
        score[c] = s
    z = sum(score.values())
    post = {c: score[c] / z for c in "AB"}
    return (display(table_array(["x_1", "x_2", "C"], [[r[0], r[1], r[2]] for r in rows]))
            + f"\n\nPhân loại $(x_1, x_2) = {x}$: điểm của mỗi lớp là $P(C)\\,P(x_1\\mid C)\\,P(x_2\\mid C)$.\n\n"
            + display(f"A:\\; {frac(score['A'])},\\qquad B:\\; {frac(score['B'])}", f"P(A\\mid x) = {frac(post['A'])},\\qquad P(B\\mid x) = {frac(post['B'])}") + "\n\nLớp có điểm lớn hơn được chọn.")


@lesson("4.2")
def prop_total_probability():
    w, d = [R(5, 10), R(3, 10), R(2, 10)], [R(1, 100), R(2, 100), R(5, 100)]
    tot = sum(a * b for a, b in zip(w, d))
    return display("P(\\text{lỗi}) = \\sum_i P(\\text{lỗi}\\mid i)P(i) = " + " + ".join(f"{frac(b)}\\cdot {frac(a)}" for a, b in zip(w, d)) + f" = {frac(tot)}")


@lesson("4.2")
def mistake_base_rate():
    prev, sens, fpr = R(1, 100), R(9, 10), R(5, 100)
    post = sens * prev / (sens * prev + fpr * (1 - prev))
    return (f"Một bệnh có tỉ lệ mắc ${frac(prev)}$; xét nghiệm dương với ${frac(sens)}$ người bệnh và ${frac(fpr)}$ người khỏe.\n\n"
            f"**Sai:** \"xét nghiệm đúng ${frac(sens)}$ nên người dương tính bệnh với xác suất ${frac(sens)}$\" (quên tỉ lệ nền).\n\n"
            f"**Đúng:** $P(\\text{{bệnh}}\\mid +) = \\dfrac{{{frac(sens)}\\cdot {frac(prev)}}}{{{frac(sens)}\\cdot {frac(prev)} + {frac(fpr)}\\cdot {frac(1 - prev)}}} = {frac(post)}$.".replace("\\text{bệnh}", "B"))


# ------------------------------------------------------------------ 4.3
@lesson("4.3")
def expectation_variance():
    xs, ps = [0, 1, 2, 4], [R(1, 4), R(1, 2), R(1, 8), R(1, 8)]
    E = sum(x * p for x, p in zip(xs, ps))
    E2 = sum(x * x * p for x, p in zip(xs, ps))
    var = E2 - E**2
    check(var == sum((x - E) ** 2 * p for x, p in zip(xs, ps)), "hai công thức phương sai")
    return display(pmf_array(xs, ps), f"\\mathbb{{E}}[X] = {frac(E)},\\qquad \\mathbb{{E}}[X^2] = {frac(E2)}", f"\\mathrm{{Var}}(X) = \\mathbb{{E}}[X^2] - (\\mathbb{{E}}[X])^2 = {frac(E2)} - {frac(E ** 2)} = {frac(var)}")


@lesson("4.3")
def linear_transform():
    xs, ps = [0, 1, 3], [R(1, 2), R(1, 4), R(1, 4)]
    E = sum(x * p for x, p in zip(xs, ps))
    var = sum((x - E) ** 2 * p for x, p in zip(xs, ps))
    a, b = 3, -2
    ey = sum((a * x + b) * p for x, p in zip(xs, ps))
    vy = sum((a * x + b - ey) ** 2 * p for x, p in zip(xs, ps))
    check(ey == a * E + b and vy == a * a * var, "biến đổi tuyến tính")
    return display(pmf_array(xs, ps), f"\\mathbb{{E}}[{a}X {b:+d}] = {a}\\cdot {frac(E)} {b:+d} = {frac(ey)}", f"\\mathrm{{Var}}({a}X {b:+d}) = {a}^2\\cdot {frac(var)} = {frac(vy)}")


@lesson("4.3")
def covariance_example():
    data = np.array([[1, 2], [2, 3], [4, 7], [5, 8]])
    n = len(data)
    X = Matrix(data.tolist())
    mu = [R(sum(X[:, j]), n) for j in range(2)]
    Xc = X - Matrix.ones(n, 1) * Matrix([[mu[0], mu[1]]])
    C = Xc.T * Xc / (n - 1)
    check(np.allclose(np.array(C.tolist(), dtype=float), np.cov(data.T, ddof=1)), "numpy")
    return display(L.mat(X, "\\mathbf{X}"), f"\\bar{{x}} = {frac(mu[0])},\\; \\bar{{y}} = {frac(mu[1])}", L.mat(C, "\\mathbf{C}")) + \
        f"\n\nHiệp phương sai $s_{{xy}} = {frac(C[0, 1])} > 0$: $x$ và $y$ cùng tăng. Đường chéo là phương sai $s_x^2 = {frac(C[0, 0])}$ và $s_y^2 = {frac(C[1, 1])}$."


@lesson("4.3")
def prop_uncorrelated():
    xs = [-1, 0, 1]
    p = R(1, 3)
    EX = sum(x * p for x in xs)
    EY = sum(x * x * p for x in xs)
    EXY = sum(x * x * x * p for x in xs)
    cov = EXY - EX * EY
    return "$X$ nhận các giá trị $-1, 0, 1$ với xác suất bằng nhau và $Y = X^2$.\n\n" + display(f"\\mathrm{{Cov}}(X, Y) = \\mathbb{{E}}[XY] - \\mathbb{{E}}[X]\\mathbb{{E}}[Y] = {frac(EXY)} - {frac(EX)}\\cdot {frac(EY)} = {frac(cov)}".replace("\\text{ đều}", "")) + \
        "\n\n$\\mathrm{Cov} = 0$ (không tương quan tuyến tính) nhưng $Y$ **hoàn toàn phụ thuộc** vào $X$. Độc lập kéo theo không tương quan, nhưng ngược lại không đúng."


@lesson("4.3")
def mistake_variance_add():
    xs = [0, 1]
    # Y = X: tương quan hoàn toàn
    p = R(1, 2)
    var = sum((x - R(1, 2)) ** 2 * p for x in xs)
    var_sum = sum((2 * x - 1) ** 2 * p for x in xs)
    return display(f"\\mathrm{{Var}}(X) = {frac(var)}", f"\\mathrm{{Var}}(X + X) = \\mathrm{{Var}}(2X) = {frac(var_sum)} \\ne \\mathrm{{Var}}(X) + \\mathrm{{Var}}(X) = {frac(2 * var)}") + \
        "\n\n**Sai:** $\\mathrm{Var}(X+Y) = \\mathrm{Var}(X) + \\mathrm{Var}(Y)$ luôn đúng. **Đúng:** chỉ khi $X, Y$ không tương quan; tổng quát $\\mathrm{Var}(X+Y) = \\mathrm{Var}(X) + \\mathrm{Var}(Y) + 2\\mathrm{Cov}(X, Y)$."


# ------------------------------------------------------------------ 4.4
@lesson("4.4")
def binomial_table():
    n, p = 4, R(1, 2)
    ps = [comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    check(sum(ps) == 1, "tổng bằng 1")
    return display(pmf_array(list(range(n + 1)), ps)) + f"\n\n$X \\sim \\mathrm{{Binomial}}({n}, {frac(p)})$: ví dụ $P(X = 2) = \\binom{{{n}}}{{2}}\\left({frac(p)}\\right)^2\\left({frac(1 - p)}\\right)^2 = {frac(ps[2])}$."


@lesson("4.4")
def zscore_example():
    mu, sigma = 70, 10
    xs = [60, 75, 95]
    zs = [R(x - mu, sigma) for x in xs]
    return display(f"\\mu = {mu},\\; \\sigma = {sigma}", *[f"z({x}) = \\dfrac{{{x} - {mu}}}{{{sigma}}} = {frac(z)}" for x, z in zip(xs, zs)]) + "\n\n$z$ cho biết giá trị cách trung bình bao nhiêu độ lệch chuẩn; $z$ càng lớn về giá trị tuyệt đối thì giá trị càng bất thường."


@lesson("4.4")
def mahalanobis_example():
    mu = Matrix([0, 0])
    S = Matrix([[4, 0], [0, 1]])
    a, b = Matrix([2, 0]), Matrix([0, 2])
    da = (a - mu).T * S.inv() * (a - mu)
    db = (b - mu).T * S.inv() * (b - mu)
    return display(L.mat(S, "\\boldsymbol{\\Sigma}"), f"d^2(\\mathbf{{a}}) = {frac(da[0, 0])},\\qquad d^2(\\mathbf{{b}}) = {frac(db[0, 0])}") + \
        "\n\nHai điểm $\\mathbf{a} = (2, 0)\\T$ và $\\mathbf{b} = (0, 2)\\T$ cách gốc cùng khoảng Euclid, nhưng $\\mathbf{b}$ **bất thường hơn** vì dữ liệu ít biến thiên theo hướng thứ hai (phương sai nhỏ hơn)."


@lesson("4.4")
def prop_binomial_moments():
    n, p = 5, R(2, 5)
    pmf = [comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    E = sum(k * q for k, q in enumerate(pmf))
    V = sum((k - E) ** 2 * q for k, q in enumerate(pmf))
    check(E == n * p and V == n * p * (1 - p), "np, np(1-p)")
    return display(f"\\mathbb{{E}}[X] = {frac(E)} = np,\\qquad \\mathrm{{Var}}(X) = {frac(V)} = np(1-p)") + f"\n\nVới $n = {n}$, $p = {frac(p)}$."


@lesson("4.4")
def mistake_zscore():
    mu, sigma, x = 50, 4, 58
    return display(f"\\mu = {mu},\\; \\sigma^2 = {sigma * sigma},\\; x = {x}") + f"\n\n**Sai:** chia cho phương sai: $z = (x - \\mu)/\\sigma^2 = {frac(R(x - mu, sigma * sigma))}$.\n\n**Đúng:** chia cho **độ lệch chuẩn** $\\sigma = {sigma}$: $z = {frac(R(x - mu, sigma))}$."


# ------------------------------------------------------------------ 4.5
@lesson("4.5")
def mle_bernoulli():
    n, k = 5, 3
    grid = [R(j, 10) for j in range(1, 10)]
    like = {p: p**k * (1 - p) ** (n - k) for p in grid}
    best = max(like, key=lambda p: like[p])
    check(best == R(6, 10), "MLE trên lưới 0,1…0,9 là 0,6")
    return display(f"L(p) = p^{{{k}}}(1-p)^{{{n - k}}}", table_array(["p"] + [frac(p) for p in grid[2:8]], [["L(p)"] + [f"{float(like[p]):.4f}" for p in grid[2:8]]])) + \
        f"\n\nVới ${k}$ lần ngửa trên ${n}$ lần tung, khả năng lớn nhất trong các giá trị thử nằm ở $p = {frac(best)}$, gần $k/n = {frac(R(k, n))}$; MLE chính xác là $\\hat{{p}} = k/n$."


@lesson("4.5")
def mle_gaussian():
    a = [2, 4, 4, 6]
    m = R(sum(a), len(a))
    var = sum((v - m) ** 2 for v in a) / len(a)
    unb = sum((v - m) ** 2 for v in a) / (len(a) - 1)
    return display("x = (" + ", ".join(str(v) for v in a) + ")", f"\\hat{{\\mu}} = {frac(m)},\\qquad \\hat{{\\sigma}}^2_{{\\mathrm{{MLE}}}} = \\dfrac{{1}}{{n}}\\sum(x_i - \\hat{{\\mu}})^2 = {frac(var)}") + \
        f"\n\nƯớc lượng không chệch chia cho $n - 1$ sẽ cho ${frac(unb)}$; MLE của phương sai chia cho $n$."


@lesson("4.5")
def map_vs_mle():
    a0, b0, n, k = 2, 2, 5, 5
    a1, b1 = a0 + k, b0 + n - k
    mle, mp = R(k, n), R(a1 - 1, a1 + b1 - 2)
    return display(f"\\hat{{p}}_{{\\mathrm{{MLE}}}} = \\dfrac{{{k}}}{{{n}}} = {frac(mle)}", f"\\mathrm{{Beta}}({a0}, {b0}) \\to \\mathrm{{Beta}}({a1}, {b1}),\\qquad \\hat{{p}}_{{\\mathrm{{MAP}}}} = \\dfrac{{{a1} - 1}}{{{a1} + {b1} - 2}} = {frac(mp)}") + \
        f"\n\nTung ${n}$ lần đều ngửa: MLE cho $p = 1$ (tin chắc đồng xu luôn ngửa), còn MAP với prior \"gần công bằng\" cho ${frac(mp)}$, thận trọng hơn. Prior hoạt động như regularization."


@lesson("4.5")
def prop_prior_strength():
    n, k = 5, 5
    rows = []
    for a0 in (1, 2, 5, 10):
        a1, b1 = a0 + k, a0 + n - k
        rows.append([f"{a0}", frac(R(a1 - 1, a1 + b1 - 2))])
    return "Với prior $\\mathrm{Beta}(\\alpha, \\alpha)$, cột trái là $\\alpha$ và cột phải là ước lượng MAP:\n\n" + display(table_array(["\\alpha", "MAP"], rows)) + "\n\nPrior càng mạnh (tham số lớn) thì MAP càng bị kéo về $1/2$; khi dữ liệu nhiều, ảnh hưởng của prior giảm."


@lesson("4.5")
def mistake_mle_variance():
    a = [1, 3, 5]
    m = R(sum(a), len(a))
    ss = sum((v - m) ** 2 for v in a)
    return display("x = (" + ", ".join(str(v) for v in a) + ")", f"\\hat{{\\sigma}}^2_{{\\mathrm{{MLE}}}} = \\dfrac{{{frac(ss)}}}{{{len(a)}}} = {frac(ss / len(a))}") + \
        f"\n\n**Sai:** nói MLE của phương sai chia cho $n - 1$ (cho ${frac(ss / (len(a) - 1))}$). **Đúng:** MLE chia cho $n$; chia cho $n-1$ là ước lượng **không chệch** (hai khái niệm khác nhau)."


# ------------------------------------------------------------------ 4.6
@lesson("4.6")
def entropy_values():
    cases = [("đồng xu công bằng", [R(1, 2), R(1, 2)]), ("phân phối $(1/2, 1/4, 1/4)$", [R(1, 2), R(1, 4), R(1, 4)]), ("xúc xắc $8$ mặt", [R(1, 8)] * 8), ("gần như chắc chắn $(1/2, 1/2)$ lệch: $(1, 0)$", [R(1)])]
    out = []
    for name, ps in cases[:3]:
        H = sum(-p * R(int(round(log2(float(p))))) for p in ps)
        check(abs(float(H) - sum(-float(p) * log2(float(p)) for p in ps)) < 1e-12, "entropy numpy")
        out.append(f"- {name}: $H = {frac(H)}$ bit")
    return "\n".join(out) + "\n\nĐồng xu công bằng cần $1$ bit để mã hóa mỗi lần tung; xúc xắc $8$ mặt cần $3$ bit; phân phối lệch cần ít bit hơn phân phối đều cùng số giá trị."


@lesson("4.6")
def cross_entropy_kl():
    p, q = [R(1, 2), R(1, 4), R(1, 4)], [R(1, 4), R(1, 4), R(1, 2)]
    dp, dq = [1, 2, 2], [2, 2, 1]
    Hp = sum(a * d for a, d in zip(p, dp))
    Hpq = sum(a * d for a, d in zip(p, dq))
    kl = Hpq - Hp
    check(abs(float(kl) - sum(float(a) * log2(float(a) / float(b)) for a, b in zip(p, q))) < 1e-12, "KL numpy")
    return display(table_array(["x", "1", "2", "3"], [["p"] + [frac(v) for v in p], ["q"] + [frac(v) for v in q]]),
                   f"H(p) = {frac(Hp)},\\qquad H(p, q) = {' + '.join(f'{frac(a)}\\cdot {d}' for a, d in zip(p, dq))} = {frac(Hpq)}", f"D_{{\\mathrm{{KL}}}}(p\\,\\Vert\\,q) = H(p, q) - H(p) = {frac(kl)}")


@lesson("4.6")
def kl_asymmetry():
    p, q = [R(1, 2), R(1, 4), R(1, 4)], [R(1, 4), R(1, 4), R(1, 2)]
    dp, dq = [1, 2, 2], [2, 2, 1]
    kl_pq = sum(a * (d2 - d1) for a, d1, d2 in zip(p, dp, dq))
    kl_qp = sum(b * (d1 - d2) for b, d1, d2 in zip(q, dp, dq))
    return display(f"D_{{\\mathrm{{KL}}}}(p\\,\\Vert\\,q) = {frac(kl_pq)},\\qquad D_{{\\mathrm{{KL}}}}(q\\,\\Vert\\,p) = {frac(kl_qp)}") + "\n\nHai chiều KL cho giá trị **khác nhau**: KL không phải là khoảng cách đối xứng."


@lesson("4.6")
def prop_entropy_bounds():
    rows = []
    for ds in ([0], [1, 1], [1, 2, 2], [2, 2, 2, 2]):
        ps = [R(1, 2**d) for d in ds]
        H = sum(p * d for p, d in zip(ps, ds))
        rows.append([str(len(ds)), "(" + ", ".join(frac(p) for p in ps) + ")", frac(H)])
    return display(table_array(["n", "(p_1, ..., p_n)", "H"], rows)) + "\n\nCột $H$ tính bằng bit." + "\n\n$0 \\le H \\le \\log_2 n$: nhỏ nhất ($0$) khi chắc chắn, lớn nhất khi phân phối đều."


@lesson("4.6")
def mistake_kl_symmetric():
    p, q = [R(1, 2), R(1, 4), R(1, 4)], [R(1, 4), R(1, 4), R(1, 2)]
    dp, dq = [1, 2, 2], [2, 2, 1]
    a = sum(x * (d2 - d1) for x, d1, d2 in zip(p, dp, dq))
    b = sum(y * (d1 - d2) for y, d1, d2 in zip(q, dp, dq))
    return f"**Sai:** coi $D_{{\\mathrm{{KL}}}}(p\\,\\Vert\\,q) = D_{{\\mathrm{{KL}}}}(q\\,\\Vert\\,p)$ như khoảng cách. Với $p, q$ ở trên: ${frac(a)}$ và ${frac(b)}$, khác nhau.\n\n**Đúng:** KL có chiều: nó đo số bit thừa khi dùng mã tối ưu của $q$ để mã hóa dữ liệu thật sự theo $p$."


def build():
    paths = []
    for code, examples in EX.items():
        paths += write_examples(code, examples)
    return paths
