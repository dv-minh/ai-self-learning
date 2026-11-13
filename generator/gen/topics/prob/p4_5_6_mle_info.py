"""Bài 4.5 (MLE, MAP) và 4.6 (entropy, cross-entropy, KL divergence)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from sympy import Rational

from ...core import latex as L
from ...core.verify import check
from ._p import (Answer, Prompt, R, Step, TopicSpec, choice_ans, frac, num_ans, pb, random, table_array, vec_ans)

L45, L46 = "4.5", "4.6"


# ------------------------------------------------------------------------ mle
def gen_mle(rng: random.Random, level: int, seed: int):
    if level == 2:
        n = rng.randint(6, 12)
        seq = [rng.randint(0, 1) for _ in range(n)]
        k = sum(seq)
        if k in (0, n):
            return gen_mle(random.Random(rng.random()), level, seed)
        p_hat = R(k, n)
        # kiểm chứng: khả năng L(p) = p^k (1-p)^(n-k) đạt lớn nhất tại k/n trong lưới các phân số mẫu n
        best = max((Fraction(j, n) for j in range(1, n)), key=lambda p: p**k * (1 - p) ** (n - k))
        check(best == Fraction(k, n), "MLE Bernoulli")
        return pb(type_="mle", lesson=L45, level=2, seed=seed,
                  prompt=Prompt(text=f"Tung một đồng xu nhiều lần, được dãy kết quả ($1$ là ngửa): ${' '.join(str(v) for v in seq)}$. Giả sử mỗi lần là Bernoulli($p$). Ước lượng hợp lý cực đại (MLE) của $p$."),
                  answer=num_ans(p_hat), hints=["Khả năng: $L(p) = p^{k}(1-p)^{n-k}$; đạo hàm của $\\ln L$ bằng $0$ cho $\\hat{p} = k/n$."],
                  steps=[Step(text=f"Có $k = {k}$ lần ngửa trong $n = {n}$ lần:", latex=f"\\hat{{p}} = \\dfrac{{k}}{{n}} = {frac(p_hat)}")], verified=["likelihood-grid"], tags=["bernoulli"])
    if level == 3:
        for _ in range(500):
            a = [rng.randint(-5, 12) for _ in range(rng.choice([4, 5]))]
            m = R(sum(a), len(a))
            if m.q == 1 and len(set(a)) > 1:
                break
        var = sum((v - m) ** 2 for v in a) / len(a)
        arr = np.array(a, dtype=float)
        check(float(m) == arr.mean() and abs(float(var) - arr.var()) < 1e-12, "MLE Gauss = numpy (ddof=0)")
        return pb(type_="mle", lesson=L45, level=3, seed=seed,
                  prompt=Prompt(text=f"Dữ liệu $x = ({', '.join(str(v) for v in a)})$ được giả sử độc lập từ $\\mathcal{{N}}(\\mu, \\sigma^2)$. Tìm MLE $(\\hat{{\\mu}},\\; \\hat{{\\sigma}}^2)$ (nhập vector). Lưu ý MLE của phương sai chia cho $n$, không phải $n-1$."),
                  answer=vec_ans([m, var]), hints=["$\\hat{\\mu}$ là trung bình mẫu; $\\hat{\\sigma}^2 = \\frac{1}{n}\\sum(x_i - \\hat{\\mu})^2$."],
                  steps=[Step(text="Trung bình và phương sai (chia $n$):", latex=f"\\hat{{\\mu}} = {frac(m)},\\quad \\hat{{\\sigma}}^2 = {frac(var)}")], verified=["numpy"], tags=["gaussian"])
    if level == 4:
        counts = [rng.randint(1, 9) for _ in range(rng.choice([3, 3, 4]))]
        N = sum(counts)
        probs = [R(c, N) for c in counts]
        return pb(type_="mle", lesson=L45, level=4, seed=seed,
                  prompt=Prompt(text=f"Quan sát ${N}$ mẫu từ một phân phối phân loại (categorical) với {len(counts)} lớp; số lần xuất hiện mỗi lớp là $({', '.join(str(c) for c in counts)})$. Tìm MLE $(\\hat{{p}}_1, \\dots)$ (nhập vector)."),
                  answer=vec_ans(probs), hints=["MLE của phân phối phân loại là tần suất tương đối: $\\hat{p}_j = n_j/N$."],
                  steps=[Step(text="Chia số lần xuất hiện cho tổng:", latex=", ".join(f"\\dfrac{{{c}}}{{{N}}} = {frac(p)}" for c, p in zip(counts, probs)))], verified=["sum-to-one"], tags=["categorical"])
    # level 5: chọn tham số có khả năng cao nhất
    n = rng.randint(5, 10)
    k = rng.randint(1, n - 1)
    cands = rng.sample([R(1, 4), R(1, 3), R(1, 2), R(2, 3), R(3, 4), R(1, 5), R(4, 5)], 3)
    like = [p**k * (1 - p) ** (n - k) for p in cands]
    if len(set(like)) < 3:
        return gen_mle(random.Random(rng.random()), level, seed)
    best = max(range(3), key=lambda i: like[i])
    check(all(Fraction(like[best].p, like[best].q) >= Fraction(l.p, l.q) for l in like), "so sánh khả năng")
    return pb(type_="mle", lesson=L45, level=5, seed=seed,
              prompt=Prompt(text=f"Một bộ phân loại dự đoán đúng $k = {k}$ trên $n = {n}$ mẫu kiểm tra độc lập. Giả sử mỗi mẫu đúng với xác suất $p$ (Bernoulli). Trong các giá trị sau của $p$, giá trị nào làm **khả năng** (likelihood) của dữ liệu lớn nhất?"),
              answer=choice_ans(best), choices=[f"$p = {frac(p)}$" for p in cands],
              hints=["Khả năng $L(p) = p^k(1-p)^{n-k}$; so sánh số hữu tỉ chính xác, hoặc chọn $p$ gần $k/n$ nhất."],
              steps=[Step(text="Khả năng của từng giá trị:", latex=", ".join(f"L({frac(p)}) = {frac(l)}" for p, l in zip(cands, like))), Step(text=f"Lớn nhất ứng với $p = {frac(cands[best])}$, gần nhất với MLE $k/n = {frac(R(k, n))}$.")],
              verified=["exact-compare"], tags=["ml", "likelihood"])


# --------------------------------------------------------------------- map-beta
def gen_map(rng: random.Random, level: int, seed: int):
    a0, b0 = rng.randint(1, 5), rng.randint(1, 5)
    n = rng.randint(4, 12)
    k = rng.randint(0, n)
    a1, b1 = a0 + k, b0 + n - k
    base = (f"Prior $\\mathrm{{Beta}}(\\alpha = {a0}, \\beta = {b0})$ cho xác suất ngửa $p$ của một đồng xu. Tung ${n}$ lần, được ${k}$ lần ngửa. "
            f"Posterior là $\\mathrm{{Beta}}(\\alpha + k,\\; \\beta + n - k)$.")
    if level == 2:
        return pb(type_="map-beta", lesson=L45, level=2, seed=seed, prompt=Prompt(text=base + " Tìm các tham số của posterior: nhập vector $(\\alpha', \\beta')$."), answer=vec_ans([a1, b1]),
                  hints=["Cộng số lần ngửa vào $\\alpha$ và số lần sấp vào $\\beta$ (số \"lần ngửa/sấp ảo\" của prior)."], steps=[Step(text="Cập nhật:", latex=f"\\alpha' = {a0} + {k} = {a1},\\quad \\beta' = {b0} + {n - k} = {b1}")],
                  verified=["conjugacy"], tags=["posterior"])
    if a1 <= 1 or b1 <= 1:
        return gen_map(random.Random(rng.random()), level, seed)
    if level == 3:
        mp = R(a1 - 1, a1 + b1 - 2)
        d = lambda p: p ** (a1 - 1) * (1 - p) ** (b1 - 1)
        fm = Fraction(mp.p, mp.q)
        eps = Fraction(1, 200)
        check(d(fm) >= d(fm + eps) and d(fm) >= d(fm - eps), "MAP là cực đại của mật độ posterior")
        return pb(type_="map-beta", lesson=L45, level=3, seed=seed, prompt=Prompt(text=base + " Tìm ước lượng MAP $\\hat{p}_{\\mathrm{MAP}}$ (điểm làm mật độ posterior lớn nhất)."), answer=num_ans(mp),
                  hints=["Mode của $\\mathrm{Beta}(\\alpha', \\beta')$ là $\\dfrac{\\alpha' - 1}{\\alpha' + \\beta' - 2}$ (khi $\\alpha', \\beta' > 1$)."],
                  steps=[Step(text="Posterior:", latex=f"\\mathrm{{Beta}}({a1}, {b1})"), Step(text="Mode:", latex=f"\\dfrac{{{a1} - 1}}{{{a1} + {b1} - 2}} = {frac(mp)}")], verified=["density-compare"], tags=["map"])
    if level == 4:
        mean = R(a1, a1 + b1)
        return pb(type_="map-beta", lesson=L45, level=4, seed=seed, prompt=Prompt(text=base + " Tính trung bình posterior $\\mathbb{E}[p \\mid D]$ với $D$ là dữ liệu đã quan sát (ước lượng Bayes)."), answer=num_ans(mean),
                  hints=["Trung bình của $\\mathrm{Beta}(\\alpha', \\beta')$ là $\\alpha'/(\\alpha'+\\beta')$."], steps=[Step(text="Trung bình:", latex=f"\\dfrac{{{a1}}}{{{a1} + {b1}}} = {frac(mean)}")], verified=["python"], tags=["posterior-mean"])
    mle = R(k, n)
    mp = R(a1 - 1, a1 + b1 - 2)
    diff = mp - mle
    return pb(type_="map-beta", lesson=L45, level=5, seed=seed,
              prompt=Prompt(text=base + " Prior đóng vai trò **regularization**, kéo ước lượng về phía giá trị prior. Tính $\\hat{p}_{\\mathrm{MAP}} - \\hat{p}_{\\mathrm{MLE}}$ (MAP trừ MLE)."),
              answer=num_ans(diff), hints=["$\\hat{p}_{\\mathrm{MLE}} = k/n$ và $\\hat{p}_{\\mathrm{MAP}} = (\\alpha'-1)/(\\alpha'+\\beta'-2)$."],
              steps=[Step(text="Hai ước lượng:", latex=f"\\hat{{p}}_{{\\mathrm{{MLE}}}} = {frac(mle)},\\quad \\hat{{p}}_{{\\mathrm{{MAP}}}} = {frac(mp)}"), Step(text="Hiệu:", latex=frac(diff))], verified=["density-compare"], tags=["ml", "regularization"])


# --------------------------------------------------------------------- entropy
def _dyadic(rng: random.Random, n: int) -> list[int]:
    """Danh sách độ sâu k_i (p_i = 2^-k_i), tổng 2^-k_i = 1 (Kraft), bằng cách tách dần các lá của cây nhị phân."""
    leaves = [0]
    while len(leaves) < n:
        i = rng.randrange(len(leaves))
        d = leaves.pop(i)
        leaves += [d + 1, d + 1]
    rng.shuffle(leaves)
    check(sum(Fraction(1, 2**d) for d in leaves) == 1, "tổng xác suất = 1")
    return leaves


def _p(depths):
    return [R(1, 2**d) for d in depths]


LABEL_FAMILIES = [
    lambda n: [f"x_{i + 1}" for i in range(n)], lambda n: [chr(65 + i) for i in range(n)], lambda n: [chr(97 + i) for i in range(n)],
    lambda n: [str(i + 1) for i in range(n)], lambda n: [f"s_{i + 1}" for i in range(n)], lambda n: [f"w_{i + 1}" for i in range(n)],
]


def _dist_table(ps, labels=None, family=0):
    labels = labels or LABEL_FAMILIES[family](len(ps))
    return table_array(["x"] + labels, [["P(x)"] + [frac(p) for p in ps]])


def gen_entropy(rng: random.Random, level: int, seed: int):
    if level == 1:
        m = rng.randint(1, 10)
        n = 2**m
        ctx = rng.choice([
            f"Một biến ngẫu nhiên phân phối **đều** trên ${n}$ giá trị.", f"Chọn ngẫu nhiên đều một trong ${n}$ lá bài khác nhau.",
            f"Một con xúc xắc công bằng có ${n}$ mặt được gieo.", f"Một ký tự được chọn ngẫu nhiên đều từ bảng chữ cái gồm ${n}$ ký tự.",
            f"Một người dùng được chọn ngẫu nhiên đều trong ${n}$ người dùng.", f"Một token được chọn đều từ một từ điển gồm ${n}$ token.",
        ])
        text = ctx + " Tính entropy $H(X) = -\\sum_x P(x)\\log_2 P(x)$ theo **bit**."
        return pb(type_="entropy-bits", lesson=L46, level=1, seed=seed, prompt=Prompt(text=text), answer=num_ans(m),
                  hints=["Phân phối đều trên $n$ giá trị có $H = \\log_2 n$."], steps=[Step(text="Mỗi xác suất là $1/n$:", latex=f"H = \\log_2 {n} = {m}")], verified=["numpy-log2"], tags=["uniform"])
    if level == 5:  # ML: bit trung bình trên mỗi token
        n = rng.randint(3, 6)
        ds = [rng.randint(1, 5) for _ in range(n)]
        total = R(sum(ds), n)
        check(abs(float(total) - np.mean([-np.log2(2.0 ** -d) for d in ds])) < 1e-9, "numpy log2")
        probs = ", ".join(frac(R(1, 2**d)) for d in ds)
        return pb(type_="entropy-bits", lesson=L46, level=5, seed=seed,
                  prompt=Prompt(text=f"Một mô hình ngôn ngữ gán cho ${n}$ token đúng liên tiếp của một câu các xác suất $({probs})$. Tính mất mát trung bình $-\\frac{{1}}{{n}}\\sum_i \\log_2 p_i$ (bit trên mỗi token)."),
                  answer=num_ans(total), hints=["$-\\log_2(2^{-d}) = d$; cộng các $d$ rồi chia cho số token."],
                  steps=[Step(text="Mỗi token đóng góp $d_i$ bit:", latex=" + ".join(str(d) for d in ds) + f" = {sum(ds)};\\quad \\dfrac{{{sum(ds)}}}{{{n}}} = {frac(total)}")], verified=["numpy-log2"], tags=["ml", "lm"])
    n = rng.choice([3, 4, 5]) if level == 2 else rng.choice([4, 5, 6])
    fam = rng.randrange(len(LABEL_FAMILIES))
    ds = _dyadic(rng, n)
    ps = _p(ds)
    H = sum(p * d for p, d in zip(ps, ds))
    check(abs(float(H) - sum(-float(p) * np.log2(float(p)) for p in ps)) < 1e-9, "numpy entropy")
    if level == 4:
        return pb(type_="entropy-bits", lesson=L46, level=4, seed=seed,
                  prompt=Prompt(text="Mã tiền tố (prefix code) tối ưu cho nguồn có phân phối dưới đây có độ dài trung bình (bit/ký hiệu) bằng entropy khi mọi xác suất là lũy thừa của $1/2$. Tính độ dài trung bình đó.", display=[_dist_table(ps, family=fam)]),
                  answer=num_ans(H), hints=["Ký hiệu có xác suất $2^{-d}$ được mã bằng $d$ bit; độ dài trung bình là $\\sum p\\,d = H(X)$."],
                  steps=[Step(text="Độ dài mã của từng ký hiệu:", latex=", ".join(str(d) for d in ds)), Step(text="Trung bình có trọng số:", latex=" + ".join(f"{frac(p)}\\cdot {d}" for p, d in zip(ps, ds)) + f" = {frac(H)}")],
                  verified=["numpy-log2", "kraft"], tags=["coding"])
    return pb(type_="entropy-bits", lesson=L46, level=level, seed=seed,
              prompt=Prompt(text="Tính entropy $H(X) = -\\sum_x P(x)\\log_2 P(x)$ (bit) của phân phối dưới đây.", display=[_dist_table(ps, family=fam)]), answer=num_ans(H),
              hints=["Với $P(x) = 2^{-d}$ thì $-\\log_2 P(x) = d$ bit."], steps=[Step(text="Mỗi số hạng $P(x)\\cdot d$:", latex=" + ".join(f"{frac(p)}\\cdot {d}" for p, d in zip(ps, ds)) + f" = {frac(H)}")],
              verified=["numpy-log2"], tags=["dyadic"])


# -------------------------------------------------------------------- kl-cross
def gen_kl(rng: random.Random, level: int, seed: int):
    n = rng.choice([3, 4, 5])
    fam = rng.randrange(len(LABEL_FAMILIES))
    dp, dq = _dyadic(rng, n), _dyadic(rng, n)
    p, q = _p(dp), _p(dq)
    Hp = sum(pi * d for pi, d in zip(p, dp))
    Hpq = sum(pi * d for pi, d in zip(p, dq))
    kl = sum(pi * (b - a) for pi, a, b in zip(p, dp, dq))
    check(kl == Hpq - Hp, "KL = H(p,q) − H(p)")
    fl = sum(float(pi) * np.log2(float(pi) / float(qi)) for pi, qi in zip(p, q))
    check(abs(fl - float(kl)) < 1e-9, "numpy KL")
    check(kl >= 0, "KL ≥ 0")
    disp = [table_array(["x"] + LABEL_FAMILIES[fam](n), [["p(x)"] + [frac(v) for v in p], ["q(x)"] + [frac(v) for v in q]])]
    if level == 2:
        return pb(type_="kl-cross", lesson=L46, level=2, seed=seed, prompt=Prompt(text="Tính cross-entropy $H(p, q) = -\\sum_x p(x)\\log_2 q(x)$ (bit).", display=disp), answer=num_ans(Hpq),
                  hints=["Dùng $p$ làm trọng số và $-\\log_2 q$ làm \"số bit\"; $-\\log_2 q(x) = d$ nếu $q(x) = 2^{-d}$."],
                  steps=[Step(text="Cộng $p(x)\\cdot(-\\log_2 q(x))$:", latex=" + ".join(f"{frac(pi)}\\cdot {d}" for pi, d in zip(p, dq)) + f" = {frac(Hpq)}")], verified=["numpy-log2"], tags=["cross-entropy"])
    if level == 3:
        return pb(type_="kl-cross", lesson=L46, level=3, seed=seed, prompt=Prompt(text="Tính KL divergence $D_{\\mathrm{KL}}(p \\,\\Vert\\, q) = \\sum_x p(x)\\log_2\\dfrac{p(x)}{q(x)}$ (bit).", display=disp), answer=num_ans(kl),
                  hints=["$\\log_2\\dfrac{p}{q} = \\log_2 p - \\log_2 q = d_q - d_p$ khi $p = 2^{-d_p}$, $q = 2^{-d_q}$."],
                  steps=[Step(text="Từng số hạng:", latex=" + ".join(f"{frac(pi)}\\cdot ({b} - {a})" for pi, a, b in zip(p, dp, dq)) + f" = {frac(kl)}")], verified=["numpy-log2"], tags=["kl"])
    if level == 4:
        return pb(type_="kl-cross", lesson=L46, level=4, seed=seed,
                  prompt=Prompt(text=f"Cho hai phân phối $p$, $q$ như bảng, với $H(p) = {frac(Hp)}$ bit và cross-entropy $H(p, q) = {frac(Hpq)}$ bit. Dùng $H(p, q) = H(p) + D_{{\\mathrm{{KL}}}}(p\\,\\Vert\\,q)$ để tính $D_{{\\mathrm{{KL}}}}(p\\,\\Vert\\,q)$.", display=disp),
                  answer=num_ans(kl), hints=["KL là số bit **thừa** khi mã hóa $p$ bằng mã tối ưu của $q$."], steps=[Step(text="Trừ:", latex=f"D_{{\\mathrm{{KL}}}} = {frac(Hpq)} - {frac(Hp)} = {frac(kl)}")], verified=["numpy-log2"], tags=["identity"])
    # level 5: mất mát cross-entropy của một phân loại
    k = rng.randint(1, 4)
    probs = _p(_dyadic(rng, 4))
    t = rng.randrange(4)
    loss = -np.log2(float(probs[t]))
    ans = R(int(round(loss)))
    check(abs(float(ans) - loss) < 1e-9, "log2 nguyên")
    return pb(type_="kl-cross", lesson=L46, level=5, seed=seed,
              prompt=Prompt(text=f"Bộ phân loại (đầu ra softmax) dự đoán xác suất bốn lớp như bảng. Nhãn đúng là lớp {t + 1} (one-hot). Mất mát cross-entropy của mẫu này là $-\\log_2 q_{{{t + 1}}}$ (bit). Tính mất mát.",
                            display=[table_array(["k", "1", "2", "3", "4"], [["q"] + [frac(v) for v in probs]])]),
              answer=num_ans(ans), hints=["Với nhãn one-hot, cross-entropy chỉ còn $-\\log_2$ xác suất gán cho lớp đúng."],
              steps=[Step(text="Xác suất của lớp đúng:", latex=f"q_{{{t + 1}}} = {frac(probs[t])} \\Rightarrow -\\log_2 q_{{{t + 1}}} = {frac(ans)}")], verified=["numpy-log2"], tags=["ml", "loss"])


SPECS = [
    TopicSpec("mle", L45, (2, 3, 4, 5), gen_mle),
    TopicSpec("map-beta", L45, (2, 3, 4, 5), gen_map),
    TopicSpec("entropy-bits", L46, (1, 2, 3, 4, 5), gen_entropy),
    TopicSpec("kl-cross", L46, (2, 3, 4, 5), gen_kl),
]
