"""Bài 4.1 (xác suất cơ bản và có điều kiện) và 4.2 (Bayes, Naive Bayes)."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

from ._p import (Answer, Prompt, R, Rational, Step, TopicSpec, check, choice_ans, frac, num_ans, pb, random, shuffled, table_array)

L41, L42 = "4.1", "4.2"


def _fr(x: Fraction) -> Rational:
    return Rational(x.numerator, x.denominator)


def _prob(space, event) -> Fraction:
    return Fraction(sum(1 for w in space if event(w)), len(space))


DICE2 = list(product(range(1, 7), repeat=2))
DIE = list(range(1, 7))


# ---------------------------------------------------------------- prob-basic
def gen_prob_basic(rng: random.Random, level: int, seed: int):
    if level == 1:
        k = rng.choice([4, 6, 8, 10, 12, 20])
        space = list(range(1, k + 1))
        t, m = rng.randint(1, k - 1), rng.randint(2, 5)
        ev, desc = rng.choice([
            (lambda w: w % 2 == 0, "là số chẵn"), (lambda w: w > t, f"lớn hơn ${t}$"), (lambda w: w <= t, f"nhỏ hơn hoặc bằng ${t}$"),
            (lambda w: w % m == 0, f"chia hết cho ${m}$"), (lambda w: w == t, f"bằng ${t}$"), (lambda w: w % 2 == 1, "là số lẻ"),
            (lambda w: w >= t, f"lớn hơn hoặc bằng ${t}$"), (lambda w: w < t, f"nhỏ hơn ${t}$"),
        ])
        p = _prob(space, ev)
        text = f"Gieo một con xúc xắc công bằng có ${k}$ mặt (các mặt $1, 2, \\dots, {k}$). Tính xác suất mặt xuất hiện {desc}."
        steps = [Step(text=f"Có ${k}$ kết quả đồng khả năng; có ${sum(1 for w in space if ev(w))}$ kết quả thuận lợi.", latex=f"P = \\dfrac{{{sum(1 for w in space if ev(w))}}}{{{k}}} = {frac(_fr(p))}")]
    elif level == 2:
        s = rng.randint(3, 11)
        mode = rng.choice(["sum", "both", "max", "differ", "sum-ge", "sum-le", "min", "diff", "prod", "first-second"])
        t2 = rng.randint(1, 6)
        if mode == "sum-ge":
            ev, desc = (lambda w: w[0] + w[1] >= s), f"tổng hai mặt $\\ge {s}$"
        elif mode == "sum-le":
            ev, desc = (lambda w: w[0] + w[1] <= s), f"tổng hai mặt $\\le {s}$"
        elif mode == "min":
            ev, desc = (lambda w: min(w) == t2), f"mặt nhỏ nhất bằng ${t2}$"
        elif mode == "diff":
            ev, desc = (lambda w: abs(w[0] - w[1]) == t2 - 1), f"hai mặt hơn kém nhau đúng ${t2 - 1}$"
        elif mode == "prod":
            ev, desc = (lambda w: (w[0] * w[1]) % t2 == 0 if t2 > 1 else w[0] * w[1] > s), (f"tích hai mặt chia hết cho ${t2}$" if t2 > 1 else f"tích hai mặt $> {s}$")
        elif mode == "first-second":
            ev, desc = (lambda w: w[0] == t2 and w[1] >= s // 2), f"mặt thứ nhất bằng ${t2}$ và mặt thứ hai $\\ge {s // 2}$"
        elif mode == "sum":
            ev, desc = (lambda w: w[0] + w[1] == s), f"tổng hai mặt bằng ${s}$"
        elif mode == "both":
            t = rng.randint(2, 6)
            ev, desc = (lambda w: w[0] >= t and w[1] >= t), f"cả hai mặt đều $\\ge {t}$"
        elif mode == "max":
            t = rng.randint(2, 6)
            ev, desc = (lambda w: max(w) == t), f"mặt lớn nhất bằng ${t}$"
        else:
            ev, desc = (lambda w: w[0] != w[1] and (w[0] + w[1]) % 2 == 0), "hai mặt khác nhau và tổng chẵn"
        p = _prob(DICE2, ev)
        text = f"Gieo hai con xúc xắc công bằng $6$ mặt. Tính xác suất {desc}."
        steps = [Step(text=f"Không gian mẫu có $36$ cặp đồng khả năng; đếm được ${sum(1 for w in DICE2 if ev(w))}$ cặp thuận lợi.", latex=f"P = {frac(_fr(p))}")]
    elif level == 3:  # rút bi không hoàn lại
        red, blue = rng.randint(2, 6), rng.randint(2, 6)
        draw = rng.choice([2, 3])
        want_red = rng.randint(1, draw)
        balls = ["R"] * red + ["B"] * blue
        combos = list(combinations(range(len(balls)), draw))
        p = Fraction(sum(1 for c in combos if sum(1 for i in c if balls[i] == "R") == want_red), len(combos))
        from math import comb

        form = Fraction(comb(red, want_red) * comb(blue, draw - want_red), comb(red + blue, draw))
        check(form == p, "tổ hợp = liệt kê")
        text = f"Một hộp có ${red}$ bi đỏ và ${blue}$ bi xanh. Rút ngẫu nhiên **không hoàn lại** ${draw}$ bi. Tính xác suất có đúng ${want_red}$ bi đỏ."
        steps = [Step(text="Đếm bằng tổ hợp:", latex=f"P = \\dfrac{{\\binom{{{red}}}{{{want_red}}}\\binom{{{blue}}}{{{draw - want_red}}}}}{{\\binom{{{red + blue}}}{{{draw}}}}} = {frac(_fr(p))}")]
    else:  # hợp, giao, bù
        mode = rng.choice(["union", "complement", "indep"])
        ta, tb, sa = rng.randint(2, 5), rng.randint(2, 5), rng.randint(6, 10)
        ev_a, da = rng.choice([(lambda w: w[0] % 2 == 0, "mặt thứ nhất chẵn"), (lambda w: w[0] > ta, f"mặt thứ nhất $> {ta}$"), (lambda w: w[0] + w[1] > sa, f"tổng $> {sa}$"),
                               (lambda w: w[0] <= ta, f"mặt thứ nhất $\\le {ta}$"), (lambda w: w[0] + w[1] <= sa, f"tổng $\\le {sa}$")])
        ev_b, db = rng.choice([(lambda w: w[1] % 2 == 1, "mặt thứ hai lẻ"), (lambda w: w[1] < tb, f"mặt thứ hai $< {tb}$"), (lambda w: w[0] == w[1], "hai mặt bằng nhau"),
                               (lambda w: w[1] >= tb, f"mặt thứ hai $\\ge {tb}$"), (lambda w: w[1] % 2 == 0, "mặt thứ hai chẵn"), (lambda w: w[0] > w[1], "mặt thứ nhất lớn hơn mặt thứ hai")])
        pa, pb_, pab = _prob(DICE2, ev_a), _prob(DICE2, ev_b), _prob(DICE2, lambda w: ev_a(w) and ev_b(w))
        if mode == "union":
            p = _prob(DICE2, lambda w: ev_a(w) or ev_b(w))
            check(p == pa + pb_ - pab, "công thức hợp")
            text = f"Gieo hai xúc xắc. Gọi $A$: {da}; $B$: {db}. Tính $P(A \\cup B)$."
            steps = [Step(text="Công thức hợp: $P(A\\cup B) = P(A) + P(B) - P(A\\cap B)$:", latex=f"{frac(_fr(pa))} + {frac(_fr(pb_))} - {frac(_fr(pab))} = {frac(_fr(p))}")]
        elif mode == "complement":
            p = 1 - pa
            text = f"Gieo hai xúc xắc. Gọi $A$: {da}. Tính xác suất của biến cố đối $P(\\overline{{A}})$."
            steps = [Step(text="$P(\\overline{A}) = 1 - P(A)$:", latex=f"1 - {frac(_fr(pa))} = {frac(_fr(p))}")]
        else:
            p = pab
            text = f"Gieo hai xúc xắc. Gọi $A$: {da}; $B$: {db}. Tính $P(A \\cap B)$ bằng cách đếm trực tiếp."
            steps = [Step(text="Đếm các cặp thỏa cả hai điều kiện:", latex=f"P(A\\cap B) = {frac(_fr(pab))}")]
    return pb(type_="prob-basic", lesson=L41, level=level, seed=seed, prompt=Prompt(text=text + " Nhập phân số $p/q$ hoặc số thập phân."),
              answer=num_ans(_fr(p)), hints=["Xác suất = số kết quả thuận lợi / tổng số kết quả đồng khả năng.", "Liệt kê hoặc đếm bằng tổ hợp."],
              steps=steps, verified=["enumeration"], tags=["counting"])


# ----------------------------------------------------------------- cond-prob
def gen_cond_prob(rng: random.Random, level: int, seed: int):
    if level in (2, 5):  # bảng 2x2
        a, b, c, d = (rng.randint(1, 20) for _ in range(4))
        if level == 2:
            rows, cols = ("nam", "nữ"), ("thích", "không thích")
            key = f" Hàng $R_1$: {rows[0]}, $R_2$: {rows[1]}; cột $C_1$: {cols[0]}, $C_2$: {cols[1]}."
            tbl = [[a, b], [c, d]]
            qi, qj = rng.randrange(2), rng.randrange(2)
            given_row = rng.random() < 0.5
            if given_row:
                p = Fraction(tbl[qi][qj], sum(tbl[qi]))
                text = f"Khảo sát một nhóm người (bảng số người). Chọn ngẫu nhiên một người thuộc nhóm **{rows[qi]}**. Tính xác suất người đó **{cols[qj].lower()}**." + key
                steps = [Step(text=f"Chỉ xét {sum(tbl[qi])} người thuộc nhóm {rows[qi]}:", latex=f"P = \\dfrac{{{tbl[qi][qj]}}}{{{sum(tbl[qi])}}} = {frac(_fr(p))}")]
            else:
                col_sum = tbl[0][qj] + tbl[1][qj]
                p = Fraction(tbl[qi][qj], col_sum)
                text = f"Khảo sát một nhóm người (bảng số người). Chọn ngẫu nhiên một người trong số những người **{cols[qj].lower()}**. Tính xác suất người đó thuộc nhóm **{rows[qi]}**." + key
                steps = [Step(text=f"Chỉ xét {col_sum} người {cols[qj].lower()}:", latex=f"P = \\dfrac{{{tbl[qi][qj]}}}{{{col_sum}}} = {frac(_fr(p))}")]
            disp = [table_array(["", "C_1", "C_2"], [["R_1", a, b], ["R_2", c, d]])]
            tags = ["table"]
        else:  # ML: precision = P(thật dương | dự đoán dương)
            tp, fp, fn, tn = a, b, c, d
            p = Fraction(tp, tp + fp)
            text = ("Một bộ phân loại nhị phân cho ma trận nhầm lẫn dưới đây (số mẫu). Với một mẫu được dự đoán là **dương**, xác suất nó **thực sự dương** là bao nhiêu? "
                    "(Đại lượng này gọi là precision.) Hàng là nhãn thật ($+$ dương, $-$ âm); cột là nhãn dự đoán ($\\hat{+}$ dự đoán dương, $\\hat{-}$ dự đoán âm).")
            disp = [table_array(["", "\\hat{+}", "\\hat{-}"], [["+", tp, fn], ["-", fp, tn]])]
            steps = [Step(text=f"Trong số {tp + fp} mẫu dự đoán dương có {tp} mẫu thật dương:", latex=f"P(\\text{{dương thật}} \\mid \\text{{dự đoán dương}}) = \\dfrac{{{tp}}}{{{tp + fp}}} = {frac(_fr(p))}".replace("\\text{dương thật}", "T").replace("\\text{dự đoán dương}", "\\hat{P}"))]
            tags = ["ml", "precision"]
        return pb(type_="cond-prob", lesson=L41, level=level, seed=seed, prompt=Prompt(text=text, display=disp), answer=num_ans(_fr(p)),
                  hints=["$P(A\\mid B) = \\dfrac{P(A\\cap B)}{P(B)}$: thu hẹp không gian mẫu vào $B$."], steps=steps, verified=["enumeration"], tags=tags)
    if level == 3:  # hai xúc xắc
        k = rng.randint(1, 6)
        t = rng.randint(4, 10)
        mode = rng.choice(["first", "sum"])
        if mode == "first":
            space = [w for w in DICE2 if w[0] == k]
            p = _prob(space, lambda w: w[0] + w[1] >= t)
            text = f"Gieo hai xúc xắc. Biết mặt thứ nhất bằng ${k}$. Tính xác suất tổng hai mặt $\\ge {t}$."
        else:
            space = [w for w in DICE2 if w[0] + w[1] == t]
            p = _prob(space, lambda w: w[0] == k)
            text = f"Gieo hai xúc xắc. Biết tổng hai mặt bằng ${t}$. Tính xác suất mặt thứ nhất bằng ${k}$."
        if not space or p == 0 and False:
            return gen_cond_prob(random.Random(rng.random()), level, seed)
        steps = [Step(text=f"Không gian mẫu thu hẹp còn ${len(space)}$ cặp; trong đó ${sum(1 for w in space if (w[0] + w[1] >= t if mode == 'first' else w[0] == k))}$ cặp thuận lợi.", latex=f"P = {frac(_fr(p))}")]
        return pb(type_="cond-prob", lesson=L41, level=3, seed=seed, prompt=Prompt(text=text + " Nhập phân số $p/q$."), answer=num_ans(_fr(p)),
                  hints=["Liệt kê các kết quả còn lại sau khi biết điều kiện."], steps=steps, verified=["enumeration"], tags=["dice"])
    # level 4: rút không hoàn lại
    red, blue = rng.randint(2, 12), rng.randint(2, 12)
    n = red + blue
    p = Fraction(red - 1, n - 1)
    balls = ["R"] * red + ["B"] * blue
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and balls[i] == "R"]
    check(Fraction(sum(1 for i, j in pairs if balls[j] == "R"), len(pairs)) == p, "liệt kê")
    return pb(type_="cond-prob", lesson=L41, level=4, seed=seed,
              prompt=Prompt(text=f"Một hộp có ${red}$ bi đỏ và ${blue}$ bi xanh. Rút lần lượt hai bi **không hoàn lại**. Biết bi thứ nhất là đỏ, tính xác suất bi thứ hai cũng đỏ."),
              answer=num_ans(_fr(p)), hints=["Sau khi rút một bi đỏ, hộp còn bao nhiêu bi và bao nhiêu bi đỏ?"],
              steps=[Step(text=f"Còn ${n - 1}$ bi, trong đó ${red - 1}$ bi đỏ:", latex=f"P = \\dfrac{{{red - 1}}}{{{n - 1}}} = {frac(_fr(p))}")], verified=["enumeration"], tags=["no-replacement"])


# ---------------------------------------------------------------- bayes-rule
def gen_bayes(rng: random.Random, level: int, seed: int):
    N = 10000  # đủ lớn để mọi số đếm là số nguyên
    if level == 5:  # ML: bộ lọc thư rác
        prior = R(rng.choice([1, 2, 3, 4, 5]), 10)
        p_w_spam, p_w_ham = R(rng.randint(5, 9), 10), R(rng.randint(1, 4), 10)
        post = p_w_spam * prior / (p_w_spam * prior + p_w_ham * (1 - prior))
        spam = int(prior * N)
        ham = N - spam
        a = int(p_w_spam * spam)
        b = int(p_w_ham * ham)
        check(R(a, a + b) == post, "đếm quần thể")
        text = (f"Trong thư của một người dùng, tỉ lệ thư rác là ${frac(prior)}$. Từ \"khuyến mãi\" xuất hiện trong ${frac(p_w_spam)}$ thư rác và trong ${frac(p_w_ham)}$ thư thường. "
                f"Một thư có chứa từ \"khuyến mãi\". Tính xác suất thư đó là rác.")
        steps = [Step(text="Bayes với xác suất toàn phần ở mẫu:", latex=f"P(\\text{{rác}}\\mid w) = \\dfrac{{{frac(p_w_spam)}\\cdot {frac(prior)}}}{{{frac(p_w_spam)}\\cdot {frac(prior)} + {frac(p_w_ham)}\\cdot {frac(1 - prior)}}} = {frac(post)}".replace("\\text{rác}", "S"))]
        tags = ["ml", "spam"]
    elif level in (2, 3):
        prev = R(rng.choice([1, 2, 5, 10]), 100)
        sens = R(rng.choice([80, 90, 95, 99]), 100)
        fpr = R(rng.choice([1, 2, 5, 10]), 100)
        pos = sens * prev + fpr * (1 - prev)
        post = sens * prev / pos
        sick = int(prev * N)
        healthy = N - sick
        tp = int(sens * sick)
        fp = int(fpr * healthy)
        check(R(tp, tp + fp) == post, "đếm quần thể")
        if level == 2:
            text = (f"Một bệnh có tỉ lệ mắc ${frac(prev)}$. Xét nghiệm cho kết quả dương với ${frac(sens)}$ người bệnh và với ${frac(fpr)}$ người không bệnh. "
                    f"Một người có kết quả dương. Tính xác suất người đó thật sự mắc bệnh.")
            steps = [Step(text="Xác suất toàn phần của kết quả dương:", latex=f"P(+) = {frac(sens)}\\cdot {frac(prev)} + {frac(fpr)}\\cdot {frac(1 - prev)} = {frac(pos)}"),
                     Step(text="Bayes:", latex=f"P(\\text{{bệnh}}\\mid +) = \\dfrac{{{frac(sens)}\\cdot {frac(prev)}}}{{{frac(pos)}}} = {frac(post)}".replace("\\text{bệnh}", "B"))]
        else:
            text = (f"Trong ${N}$ người có ${sick}$ người mắc bệnh. Xét nghiệm dương với ${tp}$ trong số người bệnh và ${fp}$ trong số người khỏe. Một người có kết quả dương. "
                    f"Tính xác suất họ mắc bệnh (đếm trực tiếp, không cần công thức).")
            steps = [Step(text=f"Có ${tp + fp}$ người dương, trong đó ${tp}$ người thật sự bệnh:", latex=f"P = \\dfrac{{{tp}}}{{{tp + fp}}} = {frac(post)}")]
        tags = ["medical"]
    else:  # level 4: ba nhà máy
        shares = rng.choice([(5, 3, 2), (6, 3, 1), (4, 4, 2), (5, 4, 1)])
        defects = [R(rng.choice([1, 2, 4, 5]), 100) for _ in range(3)]
        w = [R(s, 10) for s in shares]
        tot = sum(wi * di for wi, di in zip(w, defects))
        k = rng.randrange(3)
        post = w[k] * defects[k] / tot
        text = (f"Ba nhà máy $A$, $B$, $C$ sản xuất lần lượt ${frac(w[0])}$, ${frac(w[1])}$, ${frac(w[2])}$ tổng sản lượng, với tỉ lệ sản phẩm lỗi ${frac(defects[0])}$, ${frac(defects[1])}$, ${frac(defects[2])}$. "
                f"Một sản phẩm được chọn ngẫu nhiên bị lỗi. Tính xác suất nó do nhà máy {'ABC'[k]} sản xuất.")
        steps = [Step(text="Xác suất toàn phần của lỗi:", latex=f"P(\\text{{lỗi}}) = " + " + ".join(f"{frac(wi)}\\cdot {frac(di)}" for wi, di in zip(w, defects)) + f" = {frac(tot)}"),
                 Step(text="Bayes:", latex=f"P({'ABC'[k]}\\mid \\text{{lỗi}}) = \\dfrac{{{frac(w[k])}\\cdot {frac(defects[k])}}}{{{frac(tot)}}} = {frac(post)}")]
        tags = ["total-probability"]
    return pb(type_="bayes-rule", lesson=L42, level=level, seed=seed, prompt=Prompt(text=text + " Nhập phân số $p/q$ hoặc số thập phân."), answer=num_ans(post),
              hints=["$P(A\\mid B) = \\dfrac{P(B\\mid A)P(A)}{P(B)}$ với $P(B)$ tính bằng xác suất toàn phần.", "Hoặc hình dung một quần thể $1000$ người và đếm."],
              steps=steps, verified=["population-count"], tags=tags)


# --------------------------------------------------------------- naive-bayes
def _dataset(rng):
    n = rng.choice([6, 8])
    rows = []
    for _ in range(n):
        rows.append((rng.randint(0, 1), rng.randint(0, 1), rng.choice(["A", "B"])))
    cs = {c for *_, c in rows}
    return rows if cs == {"A", "B"} and all(sum(1 for r in rows if r[2] == c) >= 2 for c in cs) else _dataset(rng)


def gen_naive_bayes(rng: random.Random, level: int, seed: int):
    rows = _dataset(rng)
    x = (rng.randint(0, 1), rng.randint(0, 1))
    alpha = 1 if level == 5 else 0
    classes = ["A", "B"]
    score = {}
    for c in classes:
        sub = [r for r in rows if r[2] == c]
        prior = R(len(sub), len(rows))
        lik = R(1)
        for j in range(2):
            cnt = sum(1 for r in sub if r[j] == x[j])
            lik *= R(cnt + alpha, len(sub) + 2 * alpha)
        score[c] = prior * lik
    z = sum(score.values())
    if z == 0:
        return gen_naive_bayes(random.Random(rng.random()), level, seed)
    post = {c: score[c] / z for c in classes}
    # đối chiếu độc lập: ước lượng từng xác suất bằng cách đếm thẳng trên bảng
    ind = {}
    for c in classes:
        sub = [r for r in rows if r[2] == c]
        ind[c] = Fraction(len(sub), len(rows))
        for j in range(2):
            ind[c] *= Fraction(sum(1 for r in sub if r[j] == x[j]) + alpha, len(sub) + 2 * alpha)
    check(all(Fraction(post[c].p, post[c].q) == ind[c] / sum(ind.values()) for c in classes), "Naive Bayes đếm trực tiếp")
    table = table_array(["x_1", "x_2", "C"], [[r[0], r[1], r[2]] for r in rows])
    base = f"Bảng huấn luyện gồm {len(rows)} mẫu với hai đặc trưng nhị phân $x_1, x_2$ và nhãn lớp $A$ hoặc $B$. Dùng Naive Bayes (giả định các đặc trưng độc lập có điều kiện theo lớp)" + \
           (" với **làm trơn Laplace** (cộng $1$ vào tử số mỗi đếm và cộng $2$ vào mẫu số của mỗi xác suất đặc trưng)" if alpha else "") + \
           f" để phân loại mẫu mới $(x_1, x_2) = ({x[0]}, {x[1]})$."
    steps = [Step(text=f"Điểm của lớp {c}: $P({c})\\prod_j P(x_j\\mid {c}) = {frac(score[c])}$.") for c in classes] + [Step(text="Chuẩn hóa:", latex=f"P(A\\mid x) = {frac(post['A'])},\\quad P(B\\mid x) = {frac(post['B'])}")]
    if level == 4:
        best = 0 if post["A"] > post["B"] else 1
        if post["A"] == post["B"]:
            return gen_naive_bayes(random.Random(rng.random()), level, seed)
        return pb(type_="naive-bayes", lesson=L42, level=4, seed=seed, prompt=Prompt(text=base + " Mẫu được gán vào lớp nào?", display=[table]),
                  answer=choice_ans(best), choices=["Lớp $A$", "Lớp $B$"], hints=["So sánh $P(C)\\prod P(x_j\\mid C)$ của hai lớp; không cần chuẩn hóa."], steps=steps,
                  verified=["count"], tags=["classify"])
    ask = rng.choice(["A", "B"])
    return pb(type_="naive-bayes", lesson=L42, level=level, seed=seed,
              prompt=Prompt(text=base + f" Tính xác suất hậu nghiệm $P({ask}\\mid x_1, x_2)$ (phân số $p/q$).", display=[table]),
              answer=num_ans(post[ask]),
              hints=["$P(C\\mid \\mathbf{x}) \\propto P(C)\\prod_j P(x_j \\mid C)$; ước lượng mỗi xác suất bằng cách đếm trong từng lớp.", "Chuẩn hóa bằng cách chia cho tổng điểm hai lớp."],
              steps=steps, verified=["count"], tags=["laplace" if alpha else "plain"] + (["ml"] if level == 5 else []))


SPECS = [
    TopicSpec("prob-basic", L41, (1, 2, 3, 4), gen_prob_basic),
    TopicSpec("cond-prob", L41, (2, 3, 4, 5), gen_cond_prob),
    TopicSpec("bayes-rule", L42, (2, 3, 4, 5), gen_bayes),
    TopicSpec("naive-bayes", L42, (3, 4, 5), gen_naive_bayes),
]
