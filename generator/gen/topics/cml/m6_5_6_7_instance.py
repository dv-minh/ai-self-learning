"""Bài 6.5 KNN: knn-classify, knn-regress. Bài 6.6 K-means: kmeans-step. Bài 6.7 Cây quyết định và random forest: gini-split, forest-average."""
from __future__ import annotations

import random
from itertools import product

import numpy as np

from ._m import (F, L, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, cb, check, choice_exercise, data_table, fp, ft, mat_ans, mat_tex, nice, num_ans, pick, vec_ans, vt)

L65, L66, L67 = "6.5", "6.6", "6.7"
A, B, C = "\\mathrm{A}", "\\mathrm{B}", "\\mathrm{C}"


def _d2(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


def _d1(p, q):
    return sum(abs(a - b) for a, b in zip(p, q))


def _knn_points(rng, n, dim, spread=6):
    """n điểm nguyên phân biệt."""
    pts = set()
    while len(pts) < n:
        pts.add(tuple(F(rng.randint(-spread, spread)) for _ in range(dim)))
    pts = list(pts)
    rng.shuffle(pts)
    return pts


# ---------------------------------------------------------------- knn-classify
def gen_knn_classify(rng: random.Random, level: int, seed: int):
    dim = 1 if level == 1 else 2
    n = {1: 4, 2: 5, 3: 6, 4: 7, 5: 7}[level]
    k = {1: 1, 2: 3, 3: 3, 4: 5, 5: 3}[level]
    metric = _d1 if level == 3 else _d2
    for _ in range(2000):
        pts = _knn_points(rng, n, dim)
        labs = [rng.choice([0, 1]) for _ in pts]
        q = tuple(F(rng.randint(-6, 6)) for _ in range(dim))
        if q in pts or len(set(labs)) < 2:
            continue
        ds = [metric(p, q) for p in pts]
        order = sorted(range(n), key=lambda i: ds[i])
        if len(set(ds)) < n:  # không có khoảng cách bằng nhau
            continue
        nb = order[:k]
        if level == 5:
            w = {c: sum(F(1) / ds[i] for i in nb if labs[i] == c) for c in (0, 1)}
            if w[0] == w[1] or 0 in ds:
                continue
            pred = 0 if w[0] > w[1] else 1
            # lấy trường hợp trọng số khác kết quả bầu chọn đơn giản để bài có ý nghĩa
            votes = sum(labs[i] for i in nb)
            if (votes * 2 > k) == (pred == 1) and rng.random() < 0.6:
                continue
        else:
            votes = sum(labs[i] for i in nb)
            pred = 1 if votes * 2 > k else 0
        break
    # kiểm chứng độc lập: NumPy argsort
    arr = np.array([[float(c) for c in p] for p in pts])
    qa = np.array([float(c) for c in q])
    dn = np.abs(arr - qa).sum(1) if level == 3 else ((arr - qa) ** 2).sum(1)
    check(list(np.argsort(dn, kind="stable")[:k]) == nb, "k láng giềng NumPy ≠ sắp xếp Fraction")
    name = lambda c: A if c == 0 else B
    label = lambda c: "Lớp A" if c == 0 else "Lớp B"
    cols = [[p[j] for p in pts] for j in range(dim)] + [[name(c) for c in labs]]
    tbl = data_table((["x"] if dim == 1 else ["x_1", "x_2"]) + ["\\text{class}"], cols)
    dname = "khoảng cách Manhattan $|x_1 - q_1| + |x_2 - q_2|$" if level == 3 else ("khoảng cách Euclid bình phương" if dim == 2 else "khoảng cách $|x - q|$")
    if dim == 1:
        ds = [abs(p[0] - q[0]) for p in pts]
        order = sorted(range(n), key=lambda i: ds[i])
        nb = order[:k]
    qt = ft(q[0]) if dim == 1 else vt(q)
    rule = ("trọng số $1/d^2$ ($d^2$ là khoảng cách Euclid bình phương), tổng trọng số lớn nhất thắng" if level == 5 else "bầu chọn đa số")
    text = f"Phân loại điểm $q = {qt}$ bằng KNN với $k = {k}$, dùng {dname} và {rule}. Dữ liệu huấn luyện ($\\mathrm{{A}}$ hoặc $\\mathrm{{B}}$) ở bảng dưới."
    choices, ans = choice_exercise(rng, label(pred), [label(1 - pred)])
    steps = [Step(text=f"{dname.capitalize()} từ $q$ đến từng điểm (theo thứ tự trong bảng):", latex="(" + ", ".join(ft(d) for d in ds) + ")"),
             Step(text=f"${k}$ điểm gần nhất là các điểm số " + ", ".join(str(i + 1) for i in nb) + ", với nhãn " + ", ".join(("A" if labs[i] == 0 else "B") for i in nb) + ".")]
    if level == 5:
        steps.append(Step(text="Tổng trọng số $1/d^2$ theo lớp:", latex=f"\\mathrm{{A}}: {ft(w[0])},\\quad \\mathrm{{B}}: {ft(w[1])}"))
    steps.append(Step(text=f"Kết luận: **{label(pred)}**."))
    return cb(type_="knn-classify", lesson=L65, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=ans, choices=choices,
              hints=["Tính khoảng cách từ $q$ đến mọi điểm huấn luyện rồi chọn $k$ điểm nhỏ nhất.", "Bình phương khoảng cách không đổi thứ tự so với khoảng cách Euclid (không cần lấy căn).",
                     "Với trọng số $1/d^2$: điểm càng gần càng có tiếng nói lớn." if level == 5 else "$k$ lẻ giúp tránh hòa phiếu khi có hai lớp."],
              steps=steps, verified=["numpy-argsort"], tags=["knn", metric.__name__])


# ----------------------------------------------------------------- knn-regress
def gen_knn_regress(rng: random.Random, level: int, seed: int):
    dim = 1 if level in (1,) else 2
    n = {1: 4, 2: 5, 3: 5, 4: 6, 5: 6}[level]
    k = {1: 2, 2: 3, 3: 3, 4: 3, 5: 4}[level]
    weighted = level >= 3
    for _ in range(3000):
        pts = _knn_points(rng, n, dim)
        ys = [F(rng.randint(-8, 12)) for _ in pts]
        q = tuple(F(rng.randint(-6, 6)) for _ in range(dim))
        if q in pts:
            continue
        ds = [_d2(p, q) for p in pts]
        if len(set(ds)) < n or 0 in ds:
            continue
        order = sorted(range(n), key=lambda i: ds[i])
        nb = order[:k]
        if weighted:
            w = [F(1) / ds[i] for i in nb]
            val = sum(wi * ys[i] for wi, i in zip(w, nb)) / sum(w)
        else:
            val = sum(ys[i] for i in nb) / k
        if nice([val], 400, 100):
            break
    arr = np.array([[float(c) for c in p] for p in pts])
    qa = np.array([float(c) for c in q])
    dn = ((arr - qa) ** 2).sum(1)
    idx = np.argsort(dn, kind="stable")[:k]
    check(list(idx) == nb, "láng giềng NumPy")
    if weighted:
        wn = 1 / dn[idx]
        check(abs((wn * np.array([float(ys[i]) for i in idx])).sum() / wn.sum() - float(val)) < 1e-9, "trung bình có trọng số NumPy")
    else:
        check(abs(np.mean([float(ys[i]) for i in idx]) - float(val)) < 1e-9, "trung bình NumPy")
    cols = [[p[j] for p in pts] for j in range(dim)] + [ys]
    tbl = data_table((["x"] if dim == 1 else ["x_1", "x_2"]) + ["y"], cols)
    qt = ft(q[0]) if dim == 1 else vt(q)
    how = "trung bình có trọng số $w_i = 1/d_i^2$ của $y$ ($d_i^2$ là khoảng cách Euclid bình phương)" if weighted else "trung bình cộng của $y$"
    ds_all = [(p[0] - q[0]) ** 2 if dim == 1 else _d2(p, q) for p in pts]
    steps = [Step(text="Khoảng cách Euclid bình phương từ $q$ đến từng điểm:", latex="(" + ", ".join(ft(d) for d in ds_all) + ")"),
             Step(text=f"${k}$ điểm gần nhất: " + ", ".join(str(i + 1) for i in nb) + f". Dự đoán là {how}.",
                  latex=(f"\\hat{{y}} = \\dfrac{{\\sum y_i/d_i^2}}{{\\sum 1/d_i^2}} = {ft(val)}" if weighted else f"\\hat{{y}} = \\dfrac{{1}}{{{k}}}\\sum y_i = {ft(val)}"))]
    return cb(type_="knn-regress", lesson=L65, level=level, seed=seed,
              prompt=Prompt(text=f"Hồi quy KNN với $k = {k}$ tại $q = {qt}$: dự đoán là {how} trên các láng giềng gần nhất (khoảng cách Euclid). " + NUM_HINT, display=[tbl]),
              answer=num_ans(val), hints=["Tìm $k$ điểm gần $q$ nhất (so sánh khoảng cách bình phương).", "Trung bình có trọng số: $\\hat{y} = \\dfrac{\\sum w_iy_i}{\\sum w_i}$." if weighted else "Hồi quy KNN: trung bình các giá trị $y$ của $k$ láng giềng."],
              steps=steps, verified=["numpy-argsort", "numpy-mean"], tags=["knn-regress", "weighted" if weighted else "plain"])


# ---------------------------------------------------------------- kmeans-step
def gen_kmeans_step(rng: random.Random, level: int, seed: int):
    kk = 3 if level >= 4 else 2
    n = {1: 5, 2: 5, 3: 6, 4: 7, 5: 6}[level]
    for _ in range(5000):
        pts = _knn_points(rng, n, 2, 7)
        cents = _knn_points(rng, kk, 2, 7)
        ds = [[_d2(p, c) for c in cents] for p in pts]
        if any(len(set(r)) < kk for r in ds):  # không có điểm nào cách đều hai tâm
            continue
        assign = [min(range(kk), key=lambda j: r[j]) for r in ds]
        if len(set(assign)) < kk:
            continue
        new = []
        for j in range(kk):
            members = [p for p, a in zip(pts, assign) if a == j]
            new.append([sum(m[t] for m in members) / len(members) for t in range(2)])
        if level == 5:
            inertia = sum(_d2(p, new[a]) for p, a in zip(pts, assign))
            if not nice([inertia], 300, 400):
                continue
        if nice([v for c in new for v in c], 6, 20):
            break
    # kiểm chứng độc lập: NumPy
    pa = np.array([[float(c) for c in p] for p in pts])
    ca = np.array([[float(c) for c in cc] for cc in cents])
    an = ((pa[:, None, :] - ca[None, :, :]) ** 2).sum(2).argmin(1)
    check(list(an) == assign, "gán cụm NumPy")
    nn = np.array([pa[an == j].mean(0) for j in range(kk)])
    check(np.allclose(nn, [[float(v) for v in c] for c in new]), "tâm mới NumPy")
    tbl = data_table(["x_1", "x_2"], [[p[0] for p in pts], [p[1] for p in pts]])
    ct = ", ".join(f"\\boldsymbol{{\\mu}}_{j + 1} = {vt(c)}" for j, c in enumerate(cents))
    base = f"K-means với $K = {kk}$ cụm, tâm khởi tạo ${ct}$, dữ liệu ở bảng ($n = {n}$ điểm, khoảng cách Euclid). "
    cl = [Step(text="Khoảng cách Euclid bình phương từ mỗi điểm đến từng tâm, gán cụm gần nhất:", latex="\\text{cluster} = (" + ", ".join(str(a + 1) for a in assign) + ")")]
    if level == 1:
        i = rng.randrange(n)
        label = f"Cụm {assign[i] + 1}"
        choices, ans = choice_exercise(rng, label, [f"Cụm {j + 1}" for j in range(kk) if j != assign[i]])
        return cb(type_="kmeans-step", lesson=L66, level=level, seed=seed,
                  prompt=Prompt(text=base + f"Bước gán: điểm số ${i + 1}$ (đánh số theo bảng) thuộc cụm nào?", display=[tbl]), answer=ans, choices=choices,
                  hints=["Tính khoảng cách (bình phương) từ điểm đến mỗi tâm, chọn tâm gần nhất.", "Khoảng cách bình phương: $(x_1 - \\mu_1)^2 + (x_2 - \\mu_2)^2$."],
                  steps=cl + [Step(text=f"Điểm số {i + 1} gần tâm {assign[i] + 1} nhất: **{label}**.")], verified=["numpy-argmin"], tags=["assign"])
    if level == 2:
        j = rng.randrange(kk)
        members = [i + 1 for i, a in enumerate(assign) if a == j]
        steps = cl + [Step(text=f"Cụm {j + 1} gồm các điểm số {', '.join(map(str, members))}; tâm mới là trung bình các điểm đó:", latex=f"\\boldsymbol{{\\mu}}_{j + 1} = {vt(new[j])}")]
        return cb(type_="kmeans-step", lesson=L66, level=level, seed=seed,
                  prompt=Prompt(text=base + f"Sau một bước gán và một bước cập nhật, tâm mới của cụm {j + 1} là gì? " + VEC_HINT, display=[tbl]), answer=vec_ans(new[j]),
                  hints=["Bước gán: mỗi điểm vào cụm của tâm gần nhất.", "Bước cập nhật: tâm mới = trung bình cộng các điểm trong cụm."], steps=steps, verified=["numpy-mean"], tags=["update"])
    if level in (3, 4):
        steps = cl + [Step(text="Tâm mới là trung bình các điểm của từng cụm:", latex="\\mathbf{M} = " + mat_tex(new))]
        return cb(type_="kmeans-step", lesson=L66, level=level, seed=seed,
                  prompt=Prompt(text=base + f"Sau một vòng (gán rồi cập nhật), tìm các tâm mới. Nhập thành ma trận ${kk}\\times 2$ mà hàng thứ $j$ là tâm của cụm $j$.", display=[tbl]), answer=mat_ans(new),
                  hints=["Gán điểm → cụm gần nhất; sau đó mỗi tâm mới là trung bình các điểm của cụm đó.", "Mỗi cụm có ít nhất một điểm trong các đề này."], steps=steps, verified=["numpy-assign-and-mean"], tags=["update", f"K{kk}"])
    steps = cl + [Step(text="Tâm mới:", latex="\\mathbf{M} = " + mat_tex(new)),
                  Step(text="Inertia (WCSS) = tổng bình phương khoảng cách từ mỗi điểm đến tâm của cụm nó (với tâm mới):", latex=f"J = {ft(inertia)}")]
    in_np = sum(((pa[i] - nn[an[i]]) ** 2).sum() for i in range(n))
    check(abs(in_np - float(inertia)) < 1e-9, "inertia NumPy")
    return cb(type_="kmeans-step", lesson=L66, level=level, seed=seed,
              prompt=Prompt(text=base + "Sau bước gán và bước cập nhật tâm, tính inertia $J = \\sum_i \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_{c(i)}\\|^2$ (với các tâm **mới**). " + NUM_HINT, display=[tbl]),
              answer=num_ans(inertia), hints=["Trước hết tìm cụm của mỗi điểm và tâm mới.", "Inertia dùng khoảng cách **bình phương** tới tâm cụm của chính điểm đó.", "Mỗi vòng K-means không bao giờ làm inertia tăng."],
              steps=steps, verified=["numpy-inertia"], tags=["inertia"])


# ------------------------------------------------------------------- gini-split
def _gini(counts):
    n = sum(counts)
    return 1 - sum(F(c, n) ** 2 for c in counts)


def gen_gini_split(rng: random.Random, level: int, seed: int):
    if level in (1, 2, 3, 5):
        nc = 3 if level == 5 else 2
        left = [rng.randint(0, 8) for _ in range(nc)]
        right = [rng.randint(0, 8) for _ in range(nc)]
        while sum(left) == 0 or sum(right) == 0:
            left = [rng.randint(0, 8) for _ in range(nc)]
            right = [rng.randint(0, 8) for _ in range(nc)]
        parent = [a + b for a, b in zip(left, right)]
        # kiểm chứng độc lập: mở rộng thành danh sách nhãn rồi tính từ tần suất
        def g_labels(cnt):
            labs = [i for i, c in enumerate(cnt) for _ in range(c)]
            return 1 - sum((labs.count(c) / len(labs)) ** 2 for c in range(len(cnt)))
        gl, gr, gp = _gini(left), _gini(right), _gini(parent)
        check(abs(float(gp) - g_labels(parent)) < 1e-12 and abs(float(gl) - g_labels(left)) < 1e-12, "Gini theo nhãn")
        wgt = F(sum(left), sum(parent)) * gl + F(sum(right), sum(parent)) * gr
        gain = gp - wgt
        cn = lambda c: "(" + ", ".join(str(v) for v in c) + ")"
        if level == 1:
            text = f"Một nút của cây quyết định có số mẫu mỗi lớp là ${cn(left)}$. Tính độ tạp Gini $G = 1 - \\sum_c p_c^2$ của nút. " + NUM_HINT
            val, steps = gl, [Step(text=f"$n = {sum(left)}$, $p_c = $ tần suất mỗi lớp:", latex=f"G = 1 - " + " - ".join(f"\\left(\\tfrac{{{c}}}{{{sum(left)}}}\\right)^2" for c in left) + f" = {ft(gl)}")]
            tag = "gini"
        elif level == 2:
            text = (f"Một phép tách chia nút thành nút trái có số mẫu mỗi lớp ${cn(left)}$ và nút phải ${cn(right)}$. Tính Gini có trọng số sau khi tách $\\dfrac{{n_L}}{{n}}G_L + \\dfrac{{n_R}}{{n}}G_R$. " + NUM_HINT)
            val, steps = wgt, [Step(text=f"$G_L = {ft(gl)}$, $G_R = {ft(gr)}$, $n_L = {sum(left)}$, $n_R = {sum(right)}$, $n = {sum(parent)}$:", latex=f"\\dfrac{{{sum(left)}}}{{{sum(parent)}}}\\cdot {ft(gl)} + \\dfrac{{{sum(right)}}}{{{sum(parent)}}}\\cdot {ft(gr)} = {ft(wgt)}")]
            tag = "weighted"
        else:
            text = (f"Nút cha có số mẫu mỗi lớp ${cn(parent)}$ được tách thành nút trái ${cn(left)}$ và nút phải ${cn(right)}$. Tính mức giảm Gini (information gain theo Gini) $G_{{\\text{{parent}}}} - \\left(\\dfrac{{n_L}}{{n}}G_L + \\dfrac{{n_R}}{{n}}G_R\\right)$. " + NUM_HINT)
            val = gain
            steps = [Step(text="Gini cha, Gini có trọng số sau tách và hiệu:", latex=f"G_P = {ft(gp)},\\quad \\text{{weighted}} = {ft(wgt)},\\quad \\text{{gain}} = {ft(gain)}")]
            tag = "gain"
        if level == 5:
            text = f"Bài toán $3$ lớp. " + text
            tag += "-3class"
        return cb(type_="gini-split", lesson=L67, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Gini của nút: $G = 1 - \\sum_c p_c^2$ với $p_c$ là tỉ lệ lớp $c$ trong nút. $G = 0$ khi nút thuần nhất.", "Gini sau tách là trung bình có trọng số (theo số mẫu mỗi nhánh) của Gini hai nhánh.", "Mức giảm Gini = Gini cha − Gini sau tách (luôn $\\ge 0$)."],
                  steps=steps, verified=["label-list-recompute"], tags=["tree", tag])
    # level 4: chọn ngưỡng tách tốt nhất trên 1 đặc trưng
    for _ in range(1000):
        n = 8
        xs = sorted(rng.sample(range(1, 15), n))
        ys = [rng.randint(0, 1) for _ in range(n)]
        if len(set(ys)) < 2:
            continue
        cands = sorted(rng.sample([i for i in range(1, n)], 3))
        thr = [F(xs[i - 1] + xs[i], 2) for i in cands]
        scores = []
        for i in cands:
            l, r = ys[:i], ys[i:]
            gl = _gini([l.count(0), l.count(1)])
            gr = _gini([r.count(0), r.count(1)])
            scores.append(F(len(l), n) * gl + F(len(r), n) * gr)
        if len(set(scores)) == 3:
            break
    best = min(range(3), key=lambda i: scores[i])
    # kiểm chứng độc lập: lặp theo ngưỡng, so từng mẫu với x <= t
    for t, s in zip(thr, scores):
        l = [y for x, y in zip(xs, ys) if x <= t]
        r = [y for x, y in zip(xs, ys) if x > t]
        sc = F(len(l), n) * _gini([l.count(0), l.count(1)]) + F(len(r), n) * _gini([r.count(0), r.count(1)])
        check(sc == s, "Gini theo ngưỡng")
    cands_txt = [f"$x \\le {ft(t)}$" for t in thr]
    choices, ans = choice_exercise(rng, cands_txt[best], [c for i, c in enumerate(cands_txt) if i != best])
    tbl = data_table(["x", "y"], [[F(v) for v in xs], [F(v) for v in ys]])
    steps = [Step(text="Gini có trọng số sau tách cho từng ngưỡng ứng viên:", latex=", ".join(f"{cands_txt[i].strip('$')}: {ft(scores[i])}" for i in range(3))),
             Step(text=f"Ngưỡng cho Gini sau tách nhỏ nhất (tách tốt nhất) là **{cands_txt[best]}**.")]
    return cb(type_="gini-split", lesson=L67, level=level, seed=seed,
              prompt=Prompt(text="Cây quyết định chọn ngưỡng tách một đặc trưng $x$ làm Gini có trọng số sau tách **nhỏ nhất**. Với dữ liệu hai lớp ($y \\in \\{0, 1\\}$) dưới đây, ngưỡng nào trong ba ứng viên là tốt nhất? (nhánh trái: $x \\le t$)", display=[tbl]),
              answer=ans, choices=choices, hints=["Với mỗi ngưỡng, chia dữ liệu thành $x \\le t$ và $x > t$, đếm số mẫu từng lớp mỗi nhánh.", "Tính Gini có trọng số sau tách và chọn giá trị nhỏ nhất."],
              steps=steps, verified=["threshold-scan"], tags=["tree", "best-split"])


# --------------------------------------------------------------- forest-average
def gen_forest_average(rng: random.Random, level: int, seed: int):
    if level == 1:
        while True:
            votes = [rng.choice([0, 1, 2]) for _ in range(5)]
            cnt = [votes.count(c) for c in range(3)]
            if sorted(cnt, reverse=True)[0] > sorted(cnt, reverse=True)[1]:
                break
        win = cnt.index(max(cnt))
        names = ["A", "B", "C"]
        choices, ans = choice_exercise(rng, f"Lớp {names[win]}", [f"Lớp {names[c]}" for c in range(3) if c != win])
        tbl = data_table(["\\text{tree}"], [[f"\\mathrm{{{names[v]}}}" for v in votes]])
        return cb(type_="forest-average", lesson=L67, level=level, seed=seed,
                  prompt=Prompt(text="Một random forest gồm $5$ cây; dự đoán lớp của từng cây cho một mẫu ở bảng. Rừng phân loại bằng bầu chọn đa số. Kết quả là lớp nào?", display=[tbl]), answer=ans, choices=choices,
                  hints=["Random forest phân loại: bầu chọn đa số giữa các cây.", "Đếm số phiếu mỗi lớp."],
                  steps=[Step(text="Số phiếu: " + ", ".join(f"{names[c]}: {cnt[c]}" for c in range(3)) + f". Thắng: **Lớp {names[win]}**.")], verified=["count"], tags=["vote"])
    if level == 2:
        t = rng.choice([3, 4, 5])
        ps = [F(rng.randint(0, 10), 10) for _ in range(t)]
        val = sum(ps) / t
        check(abs(float(val) - np.mean([float(p) for p in ps])) < 1e-12, "trung bình NumPy")
        pred = "Lớp 1" if val > F(1, 2) else "Lớp 0" if val < F(1, 2) else None
        tbl = data_table(["P(y=1)"], [ps])
        return cb(type_="forest-average", lesson=L67, level=level, seed=seed,
                  prompt=Prompt(text=f"Random forest $\\text{{{''}}}$ ${t}$ cây (mỗi cây cho xác suất lớp $1$ ở bảng). Rừng lấy trung bình xác suất. Xác suất lớp $1$ của cả rừng là bao nhiêu? ".replace("$\\text{}$ ", "") + NUM_HINT, display=[tbl]),
                  answer=num_ans(val), hints=["Trung bình các xác suất của các cây (soft voting).", "Dự đoán lớp $1$ nếu xác suất trung bình $> 1/2$."],
                  steps=[Step(text="Trung bình cộng:", latex=f"\\bar{{p}} = \\dfrac{{1}}{{{t}}}\\sum_k p_k = {ft(val)}")], verified=["numpy-mean"], tags=["soft-vote"])
    if level in (3, 4):
        n = 6
        trees = [[rng.randrange(n) for _ in range(n)] for _ in range(5 if level == 4 else 1)]
        if level == 3:
            boot = trees[0]
            oob = sorted(set(range(n)) - set(boot))
            # kiểm chứng: đếm trực tiếp
            check(len(oob) == sum(1 for i in range(n) if i not in boot), "OOB")
            return cb(type_="forest-average", lesson=L67, level=level, seed=seed,
                      prompt=Prompt(text=f"Một cây của random forest được huấn luyện trên mẫu bootstrap gồm ${n}$ lần rút có hoàn lại từ ${n}$ mẫu (đánh số $1$ đến ${n}$): $({', '.join(str(i + 1) for i in boot)})$. Có bao nhiêu mẫu **không** xuất hiện trong mẫu bootstrap này (out-of-bag, OOB)? " + NUM_HINT),
                      answer=num_ans(F(len(oob))), hints=["Mẫu OOB là mẫu chưa bao giờ được rút trong $n$ lần rút.", "Liệt kê các chỉ số khác nhau đã xuất hiện, lấy $n$ trừ đi số chỉ số khác nhau."],
                      steps=[Step(text=f"Các chỉ số khác nhau đã xuất hiện: ${len(set(boot))}$ chỉ số, nên OOB có ${n} - {len(set(boot))} = {len(oob)}$ mẫu: " + (", ".join(str(i + 1) for i in oob) or "không có") + ".")], verified=["direct-count"], tags=["oob-count"])
        # level 4: dự đoán OOB cho mẫu i
        while True:
            i = rng.randrange(n)
            preds = [rng.choice([0, 1]) for _ in trees]
            oob_trees = [t for t, b in enumerate(trees) if i not in b]
            ones = sum(preds[t] for t in oob_trees)
            if len(oob_trees) % 2 == 1 and len(oob_trees) >= 1:
                break
        win = 1 if ones * 2 > len(oob_trees) else 0
        choices, ans = choice_exercise(rng, f"Lớp {win}", [f"Lớp {1 - win}"])
        tbl = "\\begin{array}{c|c|c} \\text{tree} & \\text{bootstrap} & \\text{pred}(" + str(i + 1) + ") \\\\ \\hline " + " \\\\ ".join(f"{t + 1} & ({', '.join(str(v + 1) for v in trees[t])}) & {preds[t]}" for t in range(len(trees))) + " \\end{array}"
        return cb(type_="forest-average", lesson=L67, level=level, seed=seed,
                  prompt=Prompt(text=f"Random forest $5$ cây, mỗi cây huấn luyện trên mẫu bootstrap ở bảng (chỉ số các mẫu được rút). Cột cuối là dự đoán của từng cây cho mẫu số ${i + 1}$. Dự đoán OOB cho mẫu ${i + 1}$ là bầu chọn đa số **chỉ trong các cây không dùng mẫu này để huấn luyện**. Kết quả?", display=[tbl]),
                  answer=ans, choices=choices,
                  hints=["Tìm những cây mà mẫu số $" + str(i + 1) + "$ **không** xuất hiện trong bootstrap của chúng.", "Chỉ bầu chọn trong các cây đó; ước lượng OOB không cần tập validation riêng."],
                  steps=[Step(text=f"Các cây OOB cho mẫu {i + 1}: " + ", ".join(str(t + 1) for t in oob_trees) + f" với dự đoán " + ", ".join(str(preds[t]) for t in oob_trees) + f". Đa số: **Lớp {win}**.")], verified=["direct-count"], tags=["oob-vote"])
    # level 5: xác suất một mẫu là OOB
    n = rng.choice([2, 3, 4, 5, 6])
    m = rng.randint(1, 6)
    kind = rng.choice(["oob", "inbag", "expected", "both"] if n >= 3 else ["oob", "inbag", "expected"])
    p_out = F(n - 1, n) ** m
    val, ask, form = {
        "oob": (p_out, "xác suất một mẫu cho trước **không** nằm trong mẫu bootstrap (tức là OOB)", f"\\left(\\dfrac{{{n - 1}}}{{{n}}}\\right)^{{{m}}}"),
        "inbag": (1 - p_out, "xác suất một mẫu cho trước nằm trong mẫu bootstrap **ít nhất một lần**", f"1 - \\left(\\dfrac{{{n - 1}}}{{{n}}}\\right)^{{{m}}}"),
        "expected": (n * p_out, "**số mẫu OOB kỳ vọng** (trung bình trên mọi mẫu bootstrap có thể)", f"{n}\\cdot\\left(\\dfrac{{{n - 1}}}{{{n}}}\\right)^{{{m}}}"),
        "both": (F(n - 2, n) ** m, "xác suất **cả hai** mẫu cho trước cùng là OOB", f"\\left(\\dfrac{{{n - 2}}}{{{n}}}\\right)^{{{m}}}"),
    }[kind]
    # kiểm chứng độc lập: liệt kê mọi dãy rút (n^m dãy) và đếm trực tiếp
    total = n**m
    seqs = list(product(range(n), repeat=m))
    if kind == "oob":
        got = F(sum(1 for s in seqs if 0 not in s), total)
    elif kind == "inbag":
        got = F(sum(1 for s in seqs if 0 in s), total)
    elif kind == "expected":
        got = F(sum(n - len(set(s)) for s in seqs), total)
    else:
        got = F(sum(1 for s in seqs if 0 not in s and 1 not in s), total)
    check(got == val, "liệt kê bootstrap")
    return cb(type_="forest-average", lesson=L67, level=level, seed=seed,
              prompt=Prompt(text=f"Mẫu bootstrap gồm ${m}$ lần rút có hoàn lại từ ${n}$ mẫu, mỗi lần đồng khả năng một trong ${n}$ mẫu. Tính {ask}. " + NUM_HINT),
              answer=num_ans(val), hints=["Mỗi lần rút, xác suất không chọn một mẫu cho trước là $\\dfrac{n-1}{n}$; các lần rút độc lập.", "Số mẫu OOB kỳ vọng $= n \\times p_{\\mathrm{OOB}}$ với $p_{\\mathrm{OOB}}$ là xác suất một mẫu là OOB (tuyến tính của kỳ vọng).", "Khi $m = n \\to \\infty$ xác suất OOB tiến về $e^{-1} \\approx 0{,}368$."],
              steps=[Step(text="Áp dụng quy tắc nhân cho các lần rút độc lập:", latex=f"{form} = {ft(val)}")], verified=["enumeration-n^m"], tags=["oob-prob", kind])


SPECS = [
    TopicSpec("knn-classify", L65, (1, 2, 3, 4, 5), gen_knn_classify),
    TopicSpec("knn-regress", L65, (1, 2, 3, 4, 5), gen_knn_regress),
    TopicSpec("kmeans-step", L66, (1, 2, 3, 4, 5), gen_kmeans_step),
    TopicSpec("gini-split", L67, (1, 2, 3, 4, 5), gen_gini_split),
    TopicSpec("forest-average", L67, (1, 2, 3, 4, 5), gen_forest_average),
]
