"""Bài 6.3 Overfitting, bias–variance, cross-validation: kfold-cv, bias-variance. Bài 6.4 Độ đo phân loại: confusion-metrics, auc-pairs."""
from __future__ import annotations

import random

import numpy as np

from ._m import (F, L, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, cb, check, choice_exercise, data_table, fp, ft, nice, num_ans, pick, vec_ans, vt)

L63, L64 = "6.3", "6.4"


# -------------------------------------------------------------------- kfold-cv
def _fold_sizes(n: int, k: int) -> list[int]:
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def gen_kfold_cv(rng: random.Random, level: int, seed: int):
    if level == 1:
        k = rng.choice([3, 4, 5, 6, 10])
        n = k * rng.randint(2, 8)
        ask = rng.choice(["fold", "train"])
        sizes = _fold_sizes(n, k)
        check(len(set(sizes)) == 1 and sum(sizes) == n, "chia đều")
        val = sizes[0] if ask == "fold" else n - sizes[0]
        text = (f"Chia ${n}$ mẫu thành ${k}$ fold bằng nhau cho cross-validation ${k}$-fold. " + ("Mỗi fold có bao nhiêu mẫu? " if ask == "fold" else "Mỗi lần lặp, tập huấn luyện có bao nhiêu mẫu? ") + NUM_HINT)
        steps = [Step(text="Mỗi lần lặp dùng một fold làm validation và $k - 1$ fold còn lại để huấn luyện.", latex=f"n/k = {n}/{k} = {sizes[0]}" + ("" if ask == "fold" else f",\\quad n - n/k = {n - sizes[0]}"))]
        return cb(type_="kfold-cv", lesson=L63, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Cross-validation $k$-fold: chia dữ liệu thành $k$ phần, mỗi phần lần lượt làm tập validation.", "Số mẫu huấn luyện mỗi lần là $n$ trừ đi kích thước một fold."],
                  steps=steps, verified=["enumeration"], tags=["folds"])
    if level == 2:
        k = rng.choice([3, 4, 5, 6, 7])
        n = k * rng.randint(2, 6) + rng.randint(1, k - 1)
        sizes = _fold_sizes(n, k)
        ask = rng.choice(["max", "min", "train-max"])
        check(sum(sizes) == n and max(sizes) - min(sizes) == 1, "kích thước fold")
        val = {"max": max(sizes), "min": min(sizes), "train-max": n - min(sizes)}[ask]
        what = {"max": "Fold lớn nhất có bao nhiêu mẫu?", "min": "Fold nhỏ nhất có bao nhiêu mẫu?", "train-max": "Tập huấn luyện lớn nhất (ở lần lặp mà validation là fold nhỏ nhất) có bao nhiêu mẫu?"}[ask]
        text = f"Chia ${n}$ mẫu thành ${k}$ fold sao cho kích thước các fold chênh nhau tối đa $1$ (các fold đầu nhận thêm phần dư). {what} " + NUM_HINT
        steps = [Step(text=f"${n} = {k}\\cdot {n // k} + {n % k}$ nên ${n % k}$ fold có ${n // k + 1}$ mẫu và ${k - n % k}$ fold có ${n // k}$ mẫu.", latex="\\text{sizes} = (" + ", ".join(str(s) for s in sizes) + ")")]
        return cb(type_="kfold-cv", lesson=L63, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Chia nguyên: $n = k\\cdot q + r$. Có $r$ fold lớn hơn một mẫu.", "Tập huấn luyện = $n$ trừ kích thước fold validation."], steps=steps, verified=["enumeration"], tags=["folds-uneven"])
    if level == 3:
        k = rng.choice([3, 4, 5])
        errs = [F(rng.randint(1, 20), rng.choice([20, 10, 5])) for _ in range(k)]
        errs = [min(e, F(9, 10)) for e in errs]
        val = sum(errs) / k
        check(abs(float(val) - np.mean([float(e) for e in errs])) < 1e-12, "trung bình NumPy")
        text = f"Cross-validation ${k}$-fold (các fold cùng kích thước) cho lỗi validation từng fold như dưới đây. Lỗi CV (trung bình) là bao nhiêu? " + NUM_HINT
        return cb(type_="kfold-cv", lesson=L63, level=level, seed=seed,
                  prompt=Prompt(text=text, display=[data_table(["e"], [errs])]), answer=num_ans(val),
                  hints=["Lỗi CV là trung bình cộng của lỗi trên các fold validation (khi các fold cùng cỡ)."],
                  steps=[Step(text="Cộng rồi chia cho số fold:", latex=f"\\dfrac{{1}}{{{k}}}\\sum_k e_k = {ft(val)}")], verified=["numpy-mean"], tags=["cv-mean"])
    if level == 4:
        k = rng.choice([3, 4])
        n = rng.randint(2 * k + 1, 5 * k)
        while n % k == 0:  # đề nói "các fold có kích thước khác nhau" nên n không được chia hết cho k
            n = rng.randint(2 * k + 1, 5 * k)
        sizes = _fold_sizes(n, k)
        wrong = [rng.randint(0, s) for s in sizes]
        rate = F(sum(wrong), n)
        naive = sum(F(w, s) for w, s in zip(wrong, sizes)) / k
        # kiểm chứng độc lập: liệt kê từng mẫu (đúng/sai) rồi đếm
        flags = [1] * 0
        for w, s in zip(wrong, sizes):
            flags += [1] * w + [0] * (s - w)
        check(F(sum(flags), len(flags)) == rate and len(flags) == n, "đếm từng mẫu")
        text = (f"Cross-validation ${k}$-fold trên ${n}$ mẫu với các fold có kích thước khác nhau. Dưới đây là kích thước và số mẫu bị dự đoán sai ở từng fold. Tính tỉ lệ lỗi tổng thể (tính trên cả ${n}$ mẫu). " + NUM_HINT)
        steps = [Step(text="Tỉ lệ lỗi tổng thể = tổng số mẫu sai / tổng số mẫu (trung bình có trọng số theo kích thước fold):", latex=f"\\dfrac{{{sum(wrong)}}}{{{n}}} = {ft(rate)}"),
                 Step(text=f"(Lưu ý: trung bình đơn giản của các tỉ lệ từng fold là ${ft(naive)}$, có thể khác khi các fold không cùng cỡ.)")]
        return cb(type_="kfold-cv", lesson=L63, level=level, seed=seed,
                  prompt=Prompt(text=text, display=[data_table(["n_k", "\\text{wrong}_k"], [[F(s) for s in sizes], [F(w) for w in wrong]])]), answer=num_ans(rate),
                  hints=["Gộp mọi mẫu validation của mọi fold: mỗi mẫu được dự đoán đúng một lần (khi nó nằm trong fold validation).", "Tỉ lệ lỗi = tổng số sai / $n$."],
                  steps=steps, verified=["per-sample-enumeration"], tags=["cv-weighted"])
    # level 5: chọn siêu tham số theo CV
    lams = rng.sample([F(1, 100), F(1, 10), F(1), F(10), F(100)], 3)
    lams.sort()
    k = rng.choice([4, 5])
    while True:
        table = [[F(rng.randint(2, 18), 20) for _ in range(k)] for _ in lams]
        means = [sum(r) / k for r in table]
        if len(set(means)) == 3:
            break
    best = min(range(3), key=lambda i: means[i])
    names = ["\\lambda = " + ft(v) for v in lams]
    tbl = "\\begin{array}{c|" + "c" * k + "|c} & " + " & ".join(f"f_{j + 1}" for j in range(k)) + " & \\text{mean} \\\\ \\hline " + " \\\\ ".join(
        f"{names[i]} & " + " & ".join(ft(v) for v in table[i]) + f" & {ft(means[i])}" for i in range(3)) + " \\end{array}"
    # bảng đề bài không chứa cột mean (người học tự tính): dựng lại
    tbl_q = "\\begin{array}{c|" + "c" * k + "} & " + " & ".join(f"f_{j + 1}" for j in range(k)) + " \\\\ \\hline " + " \\\\ ".join(f"{names[i]} & " + " & ".join(ft(v) for v in table[i]) for i in range(3)) + " \\end{array}"
    cands = [f"$\\lambda = {ft(v)}$" for v in lams]
    choices, ans = choice_exercise(rng, cands[best], [c for i, c in enumerate(cands) if i != best])
    check(int(np.argmin([float(m) for m in means])) == best, "NumPy argmin")
    steps = [Step(text="Lỗi CV trung bình của từng giá trị $\\lambda$ (trung bình các fold):", latex=", ".join(f"{names[i]}: {ft(means[i])}" for i in range(3))),
             Step(text=f"Giá trị có lỗi CV nhỏ nhất là **{cands[best]}**.")]
    return cb(type_="kfold-cv", lesson=L63, level=level, seed=seed,
              prompt=Prompt(text=f"Chọn siêu tham số $\\lambda$ bằng cross-validation ${k}$-fold (các fold cùng cỡ). Bảng cho lỗi validation trên từng fold. Chọn giá trị $\\lambda$ tốt nhất.", display=[tbl_q]),
              answer=ans, choices=choices,
              hints=["Tính lỗi CV trung bình (theo hàng) cho mỗi $\\lambda$.", "Chọn $\\lambda$ có lỗi CV **nhỏ nhất**. Không dùng tập kiểm tra cuối cùng để chọn."],
              steps=steps, verified=["numpy-argmin"], tags=["model-selection"])


# --------------------------------------------------------------- bias-variance
def gen_bias_variance(rng: random.Random, level: int, seed: int):
    k = rng.choice([4, 5, 6])
    truth = F(rng.randint(-5, 10))

    def model():
        return [truth + F(rng.randint(-6, 6)) for _ in range(k)]

    preds = model()
    mean = sum(preds) / k
    bias = mean - truth
    var = sum((p - mean) ** 2 for p in preds) / k
    s = F(rng.choice([1, 2, 3]))
    # kiểm chứng độc lập: liệt kê mọi (bộ dữ liệu, nhiễu ±s) và tính kỳ vọng sai số bình phương trực tiếp
    def mse(pr):
        m = sum(pr) / len(pr)
        return sum(((p - (truth + e)) ** 2) for p in pr for e in (s, -s)) / (2 * len(pr)), (m - truth) ** 2 + sum((p - m) ** 2 for p in pr) / len(pr) + s * s
    direct, formula = mse(preds)
    check(direct == formula, "MSE trực tiếp ≠ bias² + variance + noise")
    tbl = data_table(["\\hat{f}_k(x_0)"], [preds])
    base = f"Mô hình được huấn luyện trên ${k}$ bộ dữ liệu khác nhau, dự đoán tại một điểm $x_0$ lần lượt như bảng (giá trị thật $f(x_0) = {ft(truth)}$). "
    if level == 1:
        text, val, why = base + "Tính bias (trung bình dự đoán trừ giá trị thật). " + NUM_HINT, bias, "Bias $= \\mathbb{E}[\\hat{f}(x_0)] - f(x_0)$."
        steps = [Step(text="Trung bình dự đoán rồi trừ giá trị thật:", latex=f"\\mathbb{{E}}[\\hat{{f}}] = {ft(mean)},\\quad \\text{{bias}} = {ft(mean)} - {fp(truth)} = {ft(bias)}")]
    elif level == 2:
        text, val, why = base + "Tính variance của dự đoán (chia cho số bộ dữ liệu, tức phương sai tổng thể). " + NUM_HINT, var, "Variance $= \\mathbb{E}\\big[(\\hat{f} - \\mathbb{E}[\\hat{f}])^2\\big]$."
        steps = [Step(text="Độ lệch bình phương so với trung bình rồi lấy trung bình:", latex=f"\\mathbb{{E}}[\\hat{{f}}] = {ft(mean)},\\quad \\mathrm{{Var}} = {ft(var)}")]
    elif level == 3:
        text, val, why = base + "Tính $\\text{bias}^2$. " + NUM_HINT, bias**2, "Bias bình phương."
        steps = [Step(text="Bias rồi bình phương:", latex=f"\\text{{bias}} = {ft(bias)},\\quad \\text{{bias}}^2 = {ft(bias ** 2)}")]
    elif level == 4:
        val = bias**2 + var + s * s
        text = base + f"Nhãn quan sát $y = f(x_0) + \\varepsilon$ với nhiễu $\\varepsilon$ bằng $+{ft(s)}$ hoặc $-{ft(s)}$ với xác suất bằng nhau (phương sai nhiễu $\\sigma^2 = {ft(s * s)}$). Tính kỳ vọng sai số bình phương $\\mathbb{{E}}[(\\hat{{f}}(x_0) - y)^2]$. " + NUM_HINT
        why = "Phân rã: $\\mathbb{E}[(\\hat{f} - y)^2] = \\text{bias}^2 + \\text{variance} + \\sigma^2$."
        steps = [Step(text="Cộng ba thành phần:", latex=f"{ft(bias ** 2)} + {ft(var)} + {ft(s * s)} = {ft(val)}")]
    else:
        preds2 = model()
        mean2 = sum(preds2) / k
        e1 = bias**2 + var
        e2 = (mean2 - truth) ** 2 + sum((p - mean2) ** 2 for p in preds2) / k
        while e1 == e2:
            preds2 = model()
            mean2 = sum(preds2) / k
            e2 = (mean2 - truth) ** 2 + sum((p - mean2) ** 2 for p in preds2) / k
        better = "Mô hình A" if e1 < e2 else "Mô hình B"
        choices, ans = choice_exercise(rng, better, ["Mô hình B" if better == "Mô hình A" else "Mô hình A"])
        d2, f2 = mse(preds2)
        check(d2 == f2, "MSE trực tiếp mô hình B")
        text = ("Hai mô hình A và B được huấn luyện trên " + f"${k}$ bộ dữ liệu; dự đoán tại $x_0$ (giá trị thật ${ft(truth)}$) nằm ở hai hàng của bảng. Mô hình nào có sai số kỳ vọng $\\text{{bias}}^2 + \\text{{variance}}$ nhỏ hơn (cùng nhiễu $\\sigma^2$)?")
        steps = [Step(text="Tính $\\text{bias}^2 + \\text{variance}$ của từng mô hình:", latex=f"A: {ft(e1)},\\qquad B: {ft(e2)}"), Step(text=f"**{better}** có sai số kỳ vọng nhỏ hơn.")]
        return cb(type_="bias-variance", lesson=L63, level=level, seed=seed,
                  prompt=Prompt(text=text, display=[data_table(["A", "B"], [preds, preds2])]), answer=ans, choices=choices,
                  hints=["Với mỗi mô hình: bias² + variance (nhiễu $\\sigma^2$ giống nhau nên không ảnh hưởng so sánh).", "Mô hình đơn giản thường có bias lớn, variance nhỏ; mô hình phức tạp thì ngược lại."],
                  steps=steps, verified=["enumerate-datasets-and-noise"], tags=["compare"])
    return cb(type_="bias-variance", lesson=L63, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=num_ans(val),
              hints=[why, "Bias đo độ lệch của dự đoán **trung bình**; variance đo độ dao động của dự đoán giữa các bộ dữ liệu."], steps=steps,
              verified=["enumerate-datasets-and-noise"], tags=["bias-variance"])


# ----------------------------------------------------------- confusion-metrics
def _counts(rng, lo=2, hi=30):
    return [rng.randint(lo, hi) for _ in range(4)]  # tp, fp, fn, tn


def gen_confusion_metrics(rng: random.Random, level: int, seed: int):
    if level == 5:  # ma trận nhầm lẫn 3 lớp, macro-F1
        for _ in range(500):
            m = [[rng.randint(0, 12) for _ in range(3)] for _ in range(3)]
            for i in range(3):
                m[i][i] += rng.randint(3, 10)
            f1s = []
            ok = True
            for c in range(3):
                tp = m[c][c]
                fp_ = sum(m[r][c] for r in range(3)) - tp
                fn = sum(m[c]) - tp
                if 2 * tp + fp_ + fn == 0:
                    ok = False
                    break
                f1s.append(F(2 * tp, 2 * tp + fp_ + fn))
            if not ok:
                continue
            val = sum(f1s) / 3
            if nice([val], 3000, 2):
                break
        # kiểm chứng độc lập: mở rộng thành danh sách nhãn rồi tính F1 theo trung bình điều hòa precision, recall
        yt, yp = [], []
        for i in range(3):
            for j in range(3):
                yt += [i] * m[i][j]
                yp += [j] * m[i][j]
        f1_alt = []
        for c in range(3):
            tp = sum(1 for a, b in zip(yt, yp) if a == c and b == c)
            pr = F(tp, sum(1 for b in yp if b == c))
            rc = F(tp, sum(1 for a in yt if a == c))
            f1_alt.append(2 * pr * rc / (pr + rc) if pr + rc else F(0))
        check(sum(f1_alt) / 3 == val, "macro-F1 theo trung bình điều hòa ≠ 2TP/(2TP+FP+FN)")
        tbl = "\\begin{array}{c|ccc} & \\hat{y}=1 & \\hat{y}=2 & \\hat{y}=3 \\\\ \\hline " + " \\\\ ".join(f"y={i + 1} & " + " & ".join(str(v) for v in m[i]) for i in range(3)) + " \\end{array}"
        steps = [Step(text=f"F1 từng lớp $= \\dfrac{{2\\,TP}}{{2\\,TP + FP + FN}}$ với $TP$ là phần tử đường chéo, $FP$ là tổng cột trừ $TP$, $FN$ là tổng hàng trừ $TP$:", latex="F_1 = (" + ", ".join(ft(v) for v in f1s) + ")"),
                 Step(text="Macro-F1 là trung bình cộng của F1 các lớp:", latex=f"\\text{{macro-}}F_1 = {ft(val)}")]
        return cb(type_="confusion-metrics", lesson=L64, level=level, seed=seed,
                  prompt=Prompt(text="Ma trận nhầm lẫn $3$ lớp (hàng: lớp thật, cột: lớp dự đoán). Tính macro-F1 (trung bình cộng F1 của ba lớp, mỗi lớp coi là \"dương\" còn các lớp khác là \"âm\"). " + NUM_HINT, display=[tbl]),
                  answer=num_ans(val), hints=["Với mỗi lớp $c$: $TP = M_{cc}$, $FP$ = tổng cột $c$ trừ $TP$, $FN$ = tổng hàng $c$ trừ $TP$.", "$F_1 = \\dfrac{2\\,TP}{2\\,TP + FP + FN}$.", "Macro: trung bình cộng các F1, mọi lớp có trọng số như nhau."],
                  steps=steps, verified=["label-expansion-harmonic-mean"], tags=["multiclass"])
    for _ in range(500):
        tp, fp_, fn, tn = _counts(rng)
        if tp + fp_ and tp + fn and fp_ + tn and fn + tn:
            break
    yt = [1] * (tp + fn) + [0] * (fp_ + tn)
    yp = [1] * tp + [0] * fn + [1] * fp_ + [0] * tn
    # tính độ đo độc lập từ danh sách nhãn
    acc = F(sum(1 for a, b in zip(yt, yp) if a == b), len(yt))
    prec = F(sum(1 for a, b in zip(yt, yp) if a == b == 1), sum(yp))
    rec = F(sum(1 for a, b in zip(yt, yp) if a == b == 1), sum(yt))
    spec = F(sum(1 for a, b in zip(yt, yp) if a == b == 0), sum(1 for a in yt if a == 0))
    f1 = 2 * prec * rec / (prec + rec)
    check(f1 == F(2 * tp, 2 * tp + fp_ + fn) and acc == F(tp + tn, tp + fp_ + fn + tn), "công thức ≠ liệt kê")
    kind = {1: "acc", 2: rng.choice(["prec", "rec"]), 3: "f1", 4: rng.choice(["spec", "bacc", "f2"])}[level]
    val, name, form = {
        "acc": (acc, "accuracy", "\\dfrac{TP + TN}{TP + FP + FN + TN}"),
        "prec": (prec, "precision", "\\dfrac{TP}{TP + FP}"),
        "rec": (rec, "recall (độ nhạy)", "\\dfrac{TP}{TP + FN}"),
        "f1": (f1, "F1", "\\dfrac{2\\,TP}{2\\,TP + FP + FN}"),
        "spec": (spec, "specificity (độ đặc hiệu)", "\\dfrac{TN}{TN + FP}"),
        "bacc": ((rec + spec) / 2, "balanced accuracy", "\\dfrac{1}{2}\\left(\\dfrac{TP}{TP + FN} + \\dfrac{TN}{TN + FP}\\right)"),
        "f2": (5 * prec * rec / (4 * prec + rec), "$F_2$ (ưu tiên recall: $F_\\beta = \\frac{(1+\\beta^2)PR}{\\beta^2 P + R}$ với $\\beta = 2$)", "\\dfrac{5\\,TP}{5\\,TP + 4\\,FN + FP}"),
    }[kind]
    if kind == "f2":
        check(val == F(5 * tp, 5 * tp + 4 * fn + fp_), "F2")
    tbl = "\\begin{array}{c|cc} & \\hat{y}=1 & \\hat{y}=0 \\\\ \\hline y=1 & " + f"{tp} & {fn}" + " \\\\ y=0 & " + f"{fp_} & {tn}" + " \\end{array}"
    return cb(type_="confusion-metrics", lesson=L64, level=level, seed=seed,
              prompt=Prompt(text=f"Ma trận nhầm lẫn của bộ phân loại nhị phân (lớp dương là $1$): $TP = {tp}$, $FN = {fn}$, $FP = {fp_}$, $TN = {tn}$. Tính {name if not name.startswith('$') else name}. " + NUM_HINT, display=[tbl]),
              answer=num_ans(val),
              hints=["$TP$: dương thật được dự đoán dương; $FP$: âm thật bị dự đoán dương; $FN$: dương thật bị dự đoán âm; $TN$: âm thật được dự đoán âm.",
                     "Precision = trong các mẫu dự đoán dương, bao nhiêu phần đúng. Recall = trong các mẫu dương thật, bao nhiêu phần được tìm ra.", f"Công thức: ${form}$."],
              steps=[Step(text=f"Áp dụng công thức {name if not name.startswith('$') else 'F_2'}:", latex=f"{form} = {ft(val)}")], verified=["label-list-recompute"], tags=["binary", kind])


# -------------------------------------------------------------------- auc-pairs
def _auc_pairs(pos, neg):
    return sum((F(1) if p > n else F(1, 2) if p == n else F(0)) for p in pos for n in neg) / (len(pos) * len(neg))


def _auc_trapezoid(pos, neg):
    """AUC bằng tích phân hình thang đường ROC (quét ngưỡng giảm dần), độc lập với đếm cặp."""
    scores = sorted(set(pos + neg), reverse=True)
    pts = [(F(0), F(0))]
    for t in scores:
        tpr = F(sum(1 for p in pos if p >= t), len(pos))
        fpr = F(sum(1 for n in neg if n >= t), len(neg))
        pts.append((fpr, tpr))
    return sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:]))


def gen_auc_pairs(rng: random.Random, level: int, seed: int):
    npos, nneg = {1: (2, 2), 2: (3, 3), 3: (3, 3), 4: (4, 3), 5: (4, 4)}[level]
    ties = level >= 4
    for _ in range(500):
        pool = [F(k, 10) for k in range(1, 10)]
        pos = sorted((rng.choice(pool) for _ in range(npos)), reverse=True) if ties else sorted(rng.sample(pool, npos), reverse=True)
        neg = sorted((rng.choice(pool) for _ in range(nneg)), reverse=True) if ties else sorted([v for v in rng.sample(pool, nneg + 2) if v not in pos][:nneg], reverse=True)
        if len(neg) < nneg:
            continue
        if not ties and set(pos) & set(neg):
            continue
        if ties and not (set(pos) & set(neg)):
            continue
        break
    tbl = "\\begin{array}{c|" + "c" * (npos + nneg) + "} \\text{score} & " + " & ".join(ft(v) for v in pos + neg) + " \\\\ y & " + " & ".join(["1"] * npos + ["0"] * nneg) + " \\end{array}"
    if level <= 2:
        t = rng.choice(sorted(set(pos + neg)))
        tpr = F(sum(1 for p in pos if p >= t), npos)
        fpr = F(sum(1 for n in neg if n >= t), nneg)
        text = f"Bộ phân loại cho điểm số (score) như bảng, nhãn $y = 1$ là dương. Với ngưỡng $\\tau = {ft(t)}$ (dự đoán dương khi score $\\ge \\tau$), tính điểm trên đường ROC: $(\\mathrm{{FPR}}, \\mathrm{{TPR}})$. " + VEC_HINT
        steps = [Step(text=f"Dự đoán dương: các mẫu có score $\\ge {ft(t)}$. $TPR = TP/P = {sum(1 for p in pos if p >= t)}/{npos}$ và $FPR = FP/N = {sum(1 for n in neg if n >= t)}/{nneg}$.", latex=f"(\\mathrm{{FPR}}, \\mathrm{{TPR}}) = ({ft(fpr)}, {ft(tpr)})")]
        return cb(type_="auc-pairs", lesson=L64, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=vec_ans([fpr, tpr]),
                  hints=["$\\mathrm{TPR} = TP/P$ và $\\mathrm{FPR} = FP/N$, với $P$ là số mẫu dương thật và $N$ là số mẫu âm thật.", "Dự đoán dương khi score $\\ge \\tau$ (kể cả bằng)."], steps=steps, verified=["recount"], tags=["roc-point"])
    auc = _auc_pairs(pos, neg)
    check(auc == _auc_trapezoid(pos, neg), "AUC theo cặp ≠ AUC tích phân hình thang ROC")
    wins = sum(1 for p in pos for n in neg if p > n)
    tie = sum(1 for p in pos for n in neg if p == n)
    text = ("AUC (diện tích dưới đường ROC) bằng xác suất một mẫu dương được chấm điểm cao hơn một mẫu âm, tính trên mọi cặp (dương, âm); cặp có score bằng nhau được tính $1/2$. "
            "Tính AUC cho dữ liệu dưới đây. " + NUM_HINT)
    steps = [Step(text=f"Có ${npos}\\times {nneg} = {npos * nneg}$ cặp: ${wins}$ cặp dương cao hơn" + (f", ${tie}$ cặp bằng nhau (mỗi cặp tính $1/2$)" if tie else "") + ".", latex=f"\\mathrm{{AUC}} = \\dfrac{{{wins}" + (f" + {tie}\\cdot\\frac{{1}}{{2}}" if tie else "") + f"}}{{{npos * nneg}}} = {ft(auc)}")]
    return cb(type_="auc-pairs", lesson=L64, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=num_ans(auc),
              hints=["Đếm số cặp (mẫu dương, mẫu âm) mà score dương lớn hơn score âm.", "Cặp bằng nhau đóng góp $1/2$.", f"Tổng số cặp là $P \\times N = {npos * nneg}$."],
              steps=steps, verified=["pair-counting", "roc-trapezoid"], tags=["auc", "ties" if tie else "no-ties"])


SPECS = [
    TopicSpec("kfold-cv", L63, (1, 2, 3, 4, 5), gen_kfold_cv),
    TopicSpec("bias-variance", L63, (1, 2, 3, 4, 5), gen_bias_variance),
    TopicSpec("confusion-metrics", L64, (1, 2, 3, 4, 5), gen_confusion_metrics),
    TopicSpec("auc-pairs", L64, (1, 2, 3, 4, 5), gen_auc_pairs),
]
