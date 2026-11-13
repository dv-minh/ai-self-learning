"""Bài 4.3 (kỳ vọng, phương sai, hiệp phương sai) và 4.4 (Bernoulli, nhị thức, chuẩn, Mahalanobis)."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
from sympy import Matrix, Rational, sqrt

from ...core import latex as L
from ...core.nice import nontrivial_matrix
from ...core.verify import check, np_close, to_np
from ._p import (Answer, Prompt, R, Step, TopicSpec, choice_ans, frac, mat_ans, num_ans, pb, pmf_array, random, shuffled, table_array, vec_ans)

L43, L44 = "4.3", "4.4"


def _fr(x: Fraction) -> Rational:
    return Rational(x.numerator, x.denominator)


def _pmf(rng, k=None):
    k = k or rng.choice([3, 3, 4])
    d = rng.choice([4, 5, 6, 8, 10])
    while True:
        cuts = sorted(rng.sample(range(1, d), k - 1))
        w = [b - a for a, b in zip([0] + cuts, cuts + [d])]
        if all(x > 0 for x in w):
            break
    xs = sorted(rng.sample(range(-4, 8), k))
    return xs, [R(wi, d) for wi in w], w, d


def gen_expect_var(rng: random.Random, level: int, seed: int):
    xs, ps, w, d = _pmf(rng)
    E = sum(x * p for x, p in zip(xs, ps))
    E2 = sum(x * x * p for x, p in zip(xs, ps))
    var = sum((x - E) ** 2 * p for x, p in zip(xs, ps))
    check(var == E2 - E**2, "Var = E[X²] − E[X]²")
    pop = [x for x, wi in zip(xs, w) for _ in range(wi)]  # quần thể d phần tử có cùng phân phối
    check(R(sum(pop), d) == E and R(sum((v - E) ** 2 for v in pop), d) == var, "kiểm bằng quần thể")
    disp = [pmf_array(xs, ps)]
    base = "Biến ngẫu nhiên rời rạc $X$ có phân phối:"
    if level in (1, 5):
        if level == 5:
            names = ["mô hình 1", "mô hình 2", "mô hình 3", "mô hình 4"][: len(xs)]
            text = ("Hệ thống chọn ngẫu nhiên một mô hình để trả lời mỗi yêu cầu; $X$ là mất mát (loss) của mô hình được chọn, với phân phối dưới đây (xác suất chọn tương ứng). "
                    "Tính mất mát kỳ vọng $\\mathbb{E}[X]$.")
        else:
            text = base + " Tính kỳ vọng $\\mathbb{E}[X]$."
        return pb(type_="expect-var", lesson=L43, level=level, seed=seed, prompt=Prompt(text=text, display=disp), answer=num_ans(E),
                  hints=["$\\mathbb{E}[X] = \\sum_x x\\,P(X = x)$."], steps=[Step(text="Cộng $x\\cdot P(X=x)$:", latex=f"\\mathbb{{E}}[X] = " + " + ".join(f"{x}\\cdot {frac(p)}" for x, p in zip(xs, ps)) + f" = {frac(E)}")],
                  verified=["enumeration"], tags=["mean"] + (["ml"] if level == 5 else []))
    if level == 2:
        return pb(type_="expect-var", lesson=L43, level=2, seed=seed, prompt=Prompt(text=base + " Tính phương sai $\\mathrm{Var}(X) = \\mathbb{E}[(X - \\mu)^2]$.", display=disp),
                  answer=num_ans(var), hints=["Tính $\\mu = \\mathbb{E}[X]$ trước, rồi $\\sum_x (x-\\mu)^2 P(X=x)$."],
                  steps=[Step(text="Kỳ vọng:", latex=f"\\mu = {frac(E)}"), Step(text="Phương sai:", latex=f"\\mathrm{{Var}}(X) = " + " + ".join(f"({x}-{L.num_paren(E)})^2\\cdot {frac(p)}" for x, p in zip(xs, ps)) + f" = {frac(var)}")],
                  verified=["enumeration", "two-formulas"], tags=["variance"])
    a, b = rng.choice([2, 3, -1, -2, 4]), rng.randint(-5, 5)
    if level == 3:
        which = rng.choice(["E", "Var"])
        if which == "E":
            val = a * E + b
            check(val == sum((a * x + b) * p for x, p in zip(xs, ps)), "E[aX+b]")
            text = f"Biến ngẫu nhiên $X$ có phân phối bên dưới. Đặt $Y = {a}X {b:+d}$. Tính $\\mathbb{{E}}[Y]$."
            step = f"\\mathbb{{E}}[Y] = {a}\\,\\mathbb{{E}}[X] {b:+d} = {a}\\cdot {frac(E)} {b:+d} = {frac(val)}"
        else:
            val = a * a * var
            check(val == sum(((a * x + b) - (a * E + b)) ** 2 * p for x, p in zip(xs, ps)), "Var(aX+b)")
            text = f"Biến ngẫu nhiên $X$ có phân phối bên dưới. Đặt $Y = {a}X {b:+d}$. Tính $\\mathrm{{Var}}(Y)$."
            step = f"\\mathrm{{Var}}(Y) = {a}^2\\,\\mathrm{{Var}}(X) = {a * a}\\cdot {frac(var)} = {frac(val)}"
        return pb(type_="expect-var", lesson=L43, level=3, seed=seed, prompt=Prompt(text=text, display=disp), answer=num_ans(val),
                  hints=["$\\mathbb{E}[aX+b] = a\\mathbb{E}[X] + b$ và $\\mathrm{Var}(aX+b) = a^2\\mathrm{Var}(X)$ (hằng số cộng không đổi độ phân tán)."],
                  steps=[Step(text="Kỳ vọng và phương sai của $X$:", latex=f"\\mathbb{{E}}[X] = {frac(E)},\\quad \\mathrm{{Var}}(X) = {frac(var)}"), Step(text="Áp dụng quy tắc biến đổi tuyến tính:", latex=step)],
                  verified=["enumeration"], tags=["linear-transform", which])
    return pb(type_="expect-var", lesson=L43, level=4, seed=seed,
              prompt=Prompt(text=base + " Tính $\\mathrm{Var}(X)$ bằng công thức $\\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$.", display=disp), answer=num_ans(var),
              hints=["Tính $\\mathbb{E}[X]$ và $\\mathbb{E}[X^2] = \\sum x^2P(X=x)$."],
              steps=[Step(text="$\\mathbb{E}[X]$ và $\\mathbb{E}[X^2]$:", latex=f"{frac(E)},\\quad {frac(E2)}"), Step(text="Phương sai:", latex=f"{frac(E2)} - ({frac(E)})^2 = {frac(var)}")],
              verified=["enumeration", "two-formulas"], tags=["shortcut"])


# ---------------------------------------------------------------- covariance
def gen_cov(rng: random.Random, level: int, seed: int):
    if level == 4:  # từ bảng xác suất đồng thời
        for _ in range(100):
            d = rng.choice([4, 8, 10])
            cells = [rng.randint(0, d // 2) for _ in range(3)]
            last = d - sum(cells)
            if last >= 0 and sum(cells) <= d:
                cells.append(last)
                break
        xs, ys = [0, 1], [rng.choice([1, 2, 3]), rng.choice([4, 5, 6])]
        joint = {(xs[0], ys[0]): R(cells[0], d), (xs[0], ys[1]): R(cells[1], d), (xs[1], ys[0]): R(cells[2], d), (xs[1], ys[1]): R(cells[3], d)}
        EX = sum(x * p for (x, y), p in joint.items())
        EY = sum(y * p for (x, y), p in joint.items())
        EXY = sum(x * y * p for (x, y), p in joint.items())
        cov = EXY - EX * EY
        check(cov == sum((x - EX) * (y - EY) * p for (x, y), p in joint.items()), "Cov hai công thức")
        tbl = table_array(["", f"Y={ys[0]}", f"Y={ys[1]}"], [[f"X={xs[0]}", frac(joint[(0, ys[0])]), frac(joint[(0, ys[1])])], [f"X={xs[1]}", frac(joint[(1, ys[0])]), frac(joint[(1, ys[1])])]])
        return pb(type_="covariance", lesson=L43, level=4, seed=seed, prompt=Prompt(text="Phân phối đồng thời $P(X = x, Y = y)$ cho ở bảng. Tính $\\mathrm{Cov}(X, Y)$.", display=[tbl]),
                  answer=num_ans(cov), hints=["$\\mathrm{Cov}(X, Y) = \\mathbb{E}[XY] - \\mathbb{E}[X]\\mathbb{E}[Y]$."],
                  steps=[Step(text="Các kỳ vọng:", latex=f"\\mathbb{{E}}[X] = {frac(EX)},\\; \\mathbb{{E}}[Y] = {frac(EY)},\\; \\mathbb{{E}}[XY] = {frac(EXY)}"), Step(text="Hiệp phương sai:", latex=f"{frac(EXY)} - {frac(EX)}\\cdot {frac(EY)} = {frac(cov)}")],
                  verified=["enumeration"], tags=["joint"])
    n = 4 if level in (2, 3) else 5
    data = np.array([[rng.randint(-4, 6), rng.randint(-4, 6)] for _ in range(n)])
    while np.linalg.matrix_rank(data - data.mean(axis=0)) < 2:
        data = np.array([[rng.randint(-4, 6), rng.randint(-4, 6)] for _ in range(n)])
    X = Matrix(data.tolist())
    mu = Matrix([[sum(X[:, j]) / n for j in range(2)]])
    Xc = X - Matrix.ones(n, 1) * mu
    C = Xc.T * Xc / (n - 1)
    np_close(C, np.cov(data.T, ddof=1))
    disp = [L.mat(X, "\\mathbf{X}")]
    if level == 2:
        return pb(type_="covariance", lesson=L43, level=2, seed=seed,
                  prompt=Prompt(text=f"Cho ${n}$ cặp $(x_i, y_i)$ (hai cột của $\\mathbf{{X}}$). Tính hiệp phương sai mẫu $s_{{xy}} = \\dfrac{{1}}{{n-1}}\\sum_i (x_i - \\bar{{x}})(y_i - \\bar{{y}})$.", display=disp),
                  answer=num_ans(C[0, 1]), hints=["Tính trung bình $\\bar{x}$ và $\\bar{y}$ trước, rồi cộng các tích độ lệch và chia cho $n - 1$."],
                  steps=[Step(text="Trung bình:", latex=f"\\bar{{x}} = {frac(mu[0])},\\; \\bar{{y}} = {frac(mu[1])}"), Step(text="Hiệp phương sai mẫu:", latex=f"s_{{xy}} = {frac(C[0, 1])}")],
                  verified=["numpy"], tags=["pairs"])
    if level == 3:
        return pb(type_="covariance", lesson=L43, level=3, seed=seed,
                  prompt=Prompt(text=f"Tính ma trận hiệp phương sai mẫu $\\mathbf{{C}} = \\dfrac{{1}}{{n-1}}\\mathbf{{X}}_c\\T\\mathbf{{X}}_c$ ($n = {n}$, hai đặc trưng; $\\mathbf{{X}}_c$ là dữ liệu đã trừ trung bình từng cột).", display=disp),
                  answer=mat_ans(C), hints=["Đường chéo là phương sai từng đặc trưng; phần tử ngoài chéo là hiệp phương sai."],
                  steps=[Step(text="Dữ liệu đã căn giữa:", latex=L.mat(Xc, "\\mathbf{X}_c")), Step(text="Ma trận hiệp phương sai:", latex=L.mat(C, "\\mathbf{C}"))],
                  verified=["numpy"], tags=["matrix"])
    return pb(type_="covariance", lesson=L43, level=5, seed=seed,
              prompt=Prompt(text=f"Ma trận dữ liệu $\\mathbf{{X}}$ có ${n}$ mẫu (hàng) và hai đặc trưng (cột). Tính ma trận hiệp phương sai mẫu $\\mathbf{{C}}$ (đối xứng, bán xác định dương), dùng trong PCA.", display=disp),
              answer=mat_ans(C), hints=["Căn giữa từng cột rồi tính $\\frac{1}{n-1}\\mathbf{X}_c\\T\\mathbf{X}_c$."], steps=[Step(text="Căn giữa rồi nhân:", latex=L.mat(C, "\\mathbf{C}"))],
              verified=["numpy"], tags=["ml", "pca"])


# ----------------------------------------------------------------- binom-prob
PS = [R(1, 2), R(1, 3), R(1, 4), R(2, 3), R(3, 4), R(1, 5), R(2, 5), R(3, 5), R(1, 6), R(5, 6), R(1, 10), R(3, 10), R(7, 10), R(9, 10), R(1, 8), R(3, 8), R(5, 8), R(7, 8)]


def _binom_pmf(n, p):
    return [comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def _enum_binom(n, p, pred):
    """Kiểm bằng cách liệt kê mọi chuỗi n lần thử."""
    tot = R(0)
    for seq in product([0, 1], repeat=n):
        k = sum(seq)
        w = p**k * (1 - p) ** (n - k)
        if pred(k):
            tot += w
    return tot


def gen_binom(rng: random.Random, level: int, seed: int):
    p = rng.choice(PS)
    n = rng.randint(2, 6)
    if level == 1:
        kind = rng.choice(["bern1", "bern0", "cat"])
        if kind == "cat":
            probs = rng.choice([(R(1, 2), R(1, 3), R(1, 6)), (R(1, 5), R(3, 10), R(1, 2)), (R(1, 4), R(1, 4), R(1, 2)), (R(1, 10), R(2, 5), R(1, 2))])
            j = rng.randrange(3)
            text = f"Biến phân loại (categorical) $Z$ nhận ba giá trị $1, 2, 3$ với xác suất $({', '.join(frac(q) for q in probs)})$. Tính xác suất $Z \\ne {j + 1}$."
            ans = 1 - probs[j]
        else:
            text = f"Biến Bernoulli $X$ có $P(X = 1) = {frac(p)}$. Tính $P(X = {1 if kind == 'bern1' else 0})$."
            ans = p if kind == "bern1" else 1 - p
        return pb(type_="binom-prob", lesson=L44, level=1, seed=seed, prompt=Prompt(text=text), answer=num_ans(ans),
                  hints=["Bernoulli: $P(X=1) = p$, $P(X=0) = 1-p$. Tổng xác suất bằng $1$."], steps=[Step(text="Kết quả:", latex=frac(ans))], verified=["sum-to-one"], tags=["bernoulli"])
    if level == 2:
        k = rng.randint(0, n)
        ans = comb(n, k) * p**k * (1 - p) ** (n - k)
        check(ans == _enum_binom(n, p, lambda t: t == k), "nhị thức = liệt kê")
        return pb(type_="binom-prob", lesson=L44, level=2, seed=seed,
                  prompt=Prompt(text=f"$X \\sim \\mathrm{{Binomial}}(n = {n}, p = {frac(p)})$ (số lần thành công trong ${n}$ lần thử độc lập). Tính $P(X = {k})$."), answer=num_ans(ans),
                  hints=["$P(X=k) = \\binom{n}{k}p^k(1-p)^{n-k}$."], steps=[Step(text="Công thức nhị thức:", latex=f"\\binom{{{n}}}{{{k}}}\\left({frac(p)}\\right)^{{{k}}}\\left({frac(1 - p)}\\right)^{{{n - k}}} = {frac(ans)}")],
                  verified=["enumeration"], tags=["pmf"])
    if level == 3:
        k = rng.randint(1, n)
        up = rng.random() < 0.5
        ans = sum(_binom_pmf(n, p)[k:]) if up else sum(_binom_pmf(n, p)[: k + 1])
        check(ans == _enum_binom(n, p, (lambda t: t >= k) if up else (lambda t: t <= k)), "tích lũy = liệt kê")
        return pb(type_="binom-prob", lesson=L44, level=3, seed=seed,
                  prompt=Prompt(text=f"$X \\sim \\mathrm{{Binomial}}(n = {n}, p = {frac(p)})$. Tính $P(X {'\\ge' if up else '\\le'} {k})$."), answer=num_ans(ans),
                  hints=["Cộng các $P(X=j)$ trong khoảng cần tính, hoặc dùng biến cố đối cho ngắn."], steps=[Step(text="Cộng các xác suất:", latex=" + ".join(frac(q) for q in (_binom_pmf(n, p)[k:] if up else _binom_pmf(n, p)[: k + 1])) + f" = {frac(ans)}")],
                  verified=["enumeration"], tags=["cdf"])
    if level == 4:
        E, V = n * p, n * p * (1 - p)
        m = sum(k * q for k, q in enumerate(_binom_pmf(n, p)))
        check(m == E and sum((k - E) ** 2 * q for k, q in enumerate(_binom_pmf(n, p))) == V, "np và np(1-p)")
        return pb(type_="binom-prob", lesson=L44, level=4, seed=seed,
                  prompt=Prompt(text=f"$X \\sim \\mathrm{{Binomial}}(n = {n}, p = {frac(p)})$. Nhập vector $(\\mathbb{{E}}[X],\\; \\mathrm{{Var}}(X))$."), answer=vec_ans([E, V]),
                  hints=["$\\mathbb{E}[X] = np$ và $\\mathrm{Var}(X) = np(1-p)$."], steps=[Step(text="Áp dụng công thức:", latex=f"np = {frac(E)},\\quad np(1-p) = {frac(V)}")],
                  verified=["enumeration"], tags=["moments"])
    keep = rng.choice(PS)
    units = rng.randint(2, 8)
    allkept = keep**units
    check(allkept == _enum_binom(units, keep, lambda t: t == units), "tất cả giữ = liệt kê")
    return pb(type_="binom-prob", lesson=L44, level=5, seed=seed,
              prompt=Prompt(text=f"Dropout: mỗi nơ-ron trong một lớp có ${units}$ nơ-ron được **giữ** độc lập với xác suất ${frac(keep)}$. Tính xác suất **cả ${units}$ nơ-ron đều được giữ**."),
              answer=num_ans(allkept), hints=["Các nơ-ron độc lập nên nhân xác suất."], steps=[Step(text="Nhân các xác suất độc lập:", latex=f"\\left({frac(keep)}\\right)^{{{units}}} = {frac(allkept)}")],
              verified=["enumeration"], tags=["ml", "dropout"])


# ------------------------------------------------------------------ normal-z
def gen_normal_z(rng: random.Random, level: int, seed: int):
    mu, sigma = rng.randint(-5, 20), rng.choice([1, 2, 3, 4, 5, 10])
    if level == 2:
        xv = mu + rng.randint(-3, 3) * sigma + rng.choice([0, sigma // 2 if sigma % 2 == 0 else 1])
        z = R(xv - mu, sigma)
        return pb(type_="normal-z", lesson=L44, level=2, seed=seed,
                  prompt=Prompt(text=f"$X \\sim \\mathcal{{N}}(\\mu = {mu}, \\sigma^2 = {sigma * sigma})$. Tính điểm chuẩn hóa (z-score) của $x = {xv}$: $z = (x - \\mu)/\\sigma$."), answer=num_ans(z),
                  hints=["$z$ cho biết $x$ cách trung bình bao nhiêu **độ lệch chuẩn**; $\\sigma$ là căn của phương sai."], steps=[Step(text="Áp dụng công thức:", latex=f"z = \\dfrac{{{xv} - {L.num_paren(mu)}}}{{{sigma}}} = {frac(z)}")],
                  verified=["python"], tags=["zscore"])
    if level == 3:
        xs = [mu + rng.randint(-2, 3) * sigma for _ in range(3)]
        zs = [R(v - mu, sigma) for v in xs]
        return pb(type_="normal-z", lesson=L44, level=3, seed=seed,
                  prompt=Prompt(text=f"Dữ liệu có trung bình $\\mu = {mu}$ và độ lệch chuẩn $\\sigma = {sigma}$. Chuẩn hóa vector $\\mathbf{{x}} = ({', '.join(str(v) for v in xs)})\\T$ thành $\\mathbf{{z}} = (\\mathbf{{x}} - \\mu)/\\sigma$."),
                  answer=vec_ans(zs), hints=["Trừ trung bình rồi chia độ lệch chuẩn cho từng phần tử."], steps=[Step(text="Từng phần tử:", latex=", ".join(f"\\dfrac{{{v} - {L.num_paren(mu)}}}{{{sigma}}} = {frac(zz)}" for v, zz in zip(xs, zs)))],
                  verified=["python"], tags=["vector"])
    if level == 4:
        z = R(rng.choice([-3, -2, -1, 1, 2, 3, 1, 2]), rng.choice([1, 2]))
        xv = mu + z * sigma
        return pb(type_="normal-z", lesson=L44, level=4, seed=seed,
                  prompt=Prompt(text=f"$X \\sim \\mathcal{{N}}(\\mu = {mu}, \\sigma^2 = {sigma * sigma})$. Giá trị $x$ nào có điểm chuẩn hóa $z = {frac(z)}$?"), answer=num_ans(xv),
                  hints=["Đảo công thức: $x = \\mu + z\\sigma$."], steps=[Step(text="Tính $x$:", latex=f"x = {mu} + {L.num_paren(z)}\\cdot {sigma} = {frac(xv)}")], verified=["python"], tags=["inverse"])
    # level 5: BatchNorm (phương sai tổng thể là số chính phương)
    for _ in range(2000):
        a = [rng.randint(-6, 10) for _ in range(4)]
        m = R(sum(a), 4)
        var = sum((v - m) ** 2 for v in a) / 4
        if m.q == 1 and var > 0 and sqrt(Rational(var)).is_Rational:
            break
    sd = sqrt(Rational(var))
    out = [(v - m) / sd for v in a]
    arr = np.array(a, dtype=float)
    np_close(Matrix(out), (arr - arr.mean()) / arr.std())
    return pb(type_="normal-z", lesson=L44, level=5, seed=seed,
              prompt=Prompt(text=f"BatchNorm chuẩn hóa các giá trị của một đặc trưng trong batch: $\\hat{{x}}_i = (x_i - \\mu_B)/\\sigma_B$ với $\\mu_B$ là trung bình và $\\sigma_B^2 = \\frac{{1}}{{N}}\\sum (x_i - \\mu_B)^2$ "
                                 f"(bỏ qua $\\varepsilon$). Batch: $\\mathbf{{x}} = ({', '.join(str(v) for v in a)})\\T$. Tính $\\hat{{\\mathbf{{x}}}}$."),
              answer=vec_ans(out), hints=["Tính $\\mu_B$, rồi $\\sigma_B^2$ (chia cho $N$), rồi $\\sigma_B$."],
              steps=[Step(text="Trung bình và phương sai:", latex=f"\\mu_B = {frac(m)},\\quad \\sigma_B^2 = {frac(var)},\\quad \\sigma_B = {frac(sd)}"), Step(text="Chuẩn hóa:", latex=L.mat(Matrix(out), "\\hat{\\mathbf{x}}"))],
              verified=["numpy"], tags=["ml", "batchnorm"])


# ---------------------------------------------------------------- mahalanobis
def gen_mahalanobis(rng: random.Random, level: int, seed: int):
    mu = Matrix([rng.randint(-3, 5), rng.randint(-3, 5)])
    xv = Matrix([rng.randint(-3, 7), rng.randint(-3, 7)])
    if level == 3:
        s1, s2 = rng.choice([1, 2, 3, 4, 5]), rng.choice([1, 2, 3, 4, 6])
        S = Matrix.diag(s1, s2)
        d2 = ((xv - mu).T * S.inv() * (xv - mu))[0, 0]
        np_close(Matrix([[d2]]), np.array([[(to_np(xv - mu).T @ np.linalg.inv(to_np(S)) @ to_np(xv - mu))[0, 0]]]))
        return pb(type_="mahalanobis", lesson=L44, level=3, seed=seed,
                  prompt=Prompt(text="Bình phương khoảng cách Mahalanobis $d^2 = (\\mathbf{x} - \\boldsymbol{\\mu})\\T\\boldsymbol{\\Sigma}^{-1}(\\mathbf{x} - \\boldsymbol{\\mu})$ đo độ lệch của $\\mathbf{x}$ so với phân phối chuẩn đa chiều. Tính $d^2$.",
                                display=[L.mat(xv, "\\mathbf{x}"), L.mat(mu, "\\boldsymbol{\\mu}"), L.mat(S, "\\boldsymbol{\\Sigma}")]),
                  answer=num_ans(d2), hints=["Với $\\boldsymbol{\\Sigma}$ đường chéo: $d^2 = \\sum_i (x_i - \\mu_i)^2/\\sigma_i^2$."],
                  steps=[Step(text="Độ lệch:", latex=L.mat(xv - mu, "\\mathbf{x} - \\boldsymbol{\\mu}")), Step(text="Chia bình phương độ lệch cho phương sai từng chiều rồi cộng:", latex=f"d^2 = {frac(d2)}")],
                  verified=["numpy"], tags=["diagonal"])
    for _ in range(300):
        A = nontrivial_matrix(rng, 2, 2, -2, 2)
        S = A.T * A + Matrix.eye(2)
        if S.det() != 0 and max(abs(v) for v in S) <= 10:
            break
    d2 = ((xv - mu).T * S.inv() * (xv - mu))[0, 0]
    np_close(Matrix([[d2]]), np.array([[(to_np(xv - mu).T @ np.linalg.inv(to_np(S)) @ to_np(xv - mu))[0, 0]]]))
    if level == 4:
        return pb(type_="mahalanobis", lesson=L44, level=4, seed=seed,
                  prompt=Prompt(text="Tính $d^2 = (\\mathbf{x} - \\boldsymbol{\\mu})\\T\\boldsymbol{\\Sigma}^{-1}(\\mathbf{x} - \\boldsymbol{\\mu})$ với $\\boldsymbol{\\Sigma}$ đối xứng xác định dương.",
                                display=[L.mat(xv, "\\mathbf{x}"), L.mat(mu, "\\boldsymbol{\\mu}"), L.mat(S, "\\boldsymbol{\\Sigma}")]),
                  answer=num_ans(d2), hints=["Tính $\\boldsymbol{\\Sigma}^{-1}$ (công thức $2\\times 2$, bài 1.9) rồi nhân từ phải sang trái."],
                  steps=[Step(text="Nghịch đảo:", latex=L.mat(S.inv(), "\\boldsymbol{\\Sigma}^{-1}")), Step(text="Dạng toàn phương:", latex=f"d^2 = {frac(d2)}")], verified=["numpy"], tags=["full"])
    # level 5: điểm nào xa hơn
    x2 = Matrix([rng.randint(-3, 7), rng.randint(-3, 7)])
    while x2 == xv:
        x2 = Matrix([rng.randint(-3, 7), rng.randint(-3, 7)])
    d2b = ((x2 - mu).T * S.inv() * (x2 - mu))[0, 0]
    if d2 == d2b:
        return gen_mahalanobis(random.Random(rng.random()), level, seed)
    far = 0 if d2 > d2b else 1
    return pb(type_="mahalanobis", lesson=L44, level=5, seed=seed,
              prompt=Prompt(text="Phát hiện ngoại lai: điểm có khoảng cách Mahalanobis lớn hơn được coi là \"bất thường\" hơn so với phân phối chuẩn $\\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})$. Điểm nào bất thường hơn?",
                            display=[L.mat(mu, "\\boldsymbol{\\mu}"), L.mat(S, "\\boldsymbol{\\Sigma}"), L.mat(xv, "\\mathbf{a}"), L.mat(x2, "\\mathbf{b}")]),
              answer=choice_ans(far), choices=["Điểm $\\mathbf{a}$", "Điểm $\\mathbf{b}$"], hints=["Tính $d^2$ cho từng điểm rồi so sánh (không cần lấy căn)."],
              steps=[Step(text="Hai khoảng cách bình phương:", latex=f"d^2(\\mathbf{{a}}) = {frac(d2)},\\quad d^2(\\mathbf{{b}}) = {frac(d2b)}")], verified=["numpy"], tags=["ml", "outlier"])


SPECS = [
    TopicSpec("expect-var", L43, (1, 2, 3, 4, 5), gen_expect_var),
    TopicSpec("covariance", L43, (2, 3, 4, 5), gen_cov),
    TopicSpec("binom-prob", L44, (1, 2, 3, 4, 5), gen_binom),
    TopicSpec("normal-z", L44, (2, 3, 4, 5), gen_normal_z),
    TopicSpec("mahalanobis", L44, (3, 4, 5), gen_mahalanobis),
]
