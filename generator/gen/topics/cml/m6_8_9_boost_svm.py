"""Bài 6.8 Gradient boosting: gbm-round. Bài 6.9 SVM: svm-margin, svm-hinge."""
from __future__ import annotations

import random
from math import isqrt

import numpy as np

from ._m import (F, L, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, cb, check, choice_exercise, data_table, fp, ft, nice, num_ans, pick, points_table, vec_ans, vt)

L68, L69 = "6.8", "6.9"


# --------------------------------------------------------------------- gbm-round
def _stump(xs, res, t):
    left = [r for x, r in zip(xs, res) if x <= t]
    right = [r for x, r in zip(xs, res) if x > t]
    lv, rv = sum(left) / len(left), sum(right) / len(right)
    sse = sum((r - lv) ** 2 for r in left) + sum((r - rv) ** 2 for r in right)
    return lv, rv, sse


def gen_gbm_round(rng: random.Random, level: int, seed: int):
    n = rng.choice([5, 6])
    for _ in range(2000):
        xs = sorted(rng.sample(range(1, 13), n))
        ys = [F(rng.randint(-4, 12)) for _ in range(n)]
        f0 = sum(ys) / n
        res = [y - f0 for y in ys]
        ts = [F(xs[i - 1] + xs[i], 2) for i in range(1, n)]
        sses = [_stump(xs, res, t)[2] for t in ts]
        order = sorted(range(len(ts)), key=lambda i: sses[i])
        if sses[order[0]] == sses[order[1]]:
            continue
        nu = rng.choice([F(1, 2), F(1, 10), F(1, 4), F(1)])
        if nice([f0], 6, 30):
            break
    bi = order[0]
    lv, rv, _ = _stump(xs, res, ts[bi])
    # kiểm chứng độc lập: NumPy, quét mọi ngưỡng
    yn = np.array([float(v) for v in ys])
    rn = yn - yn.mean()
    best_np = min(range(len(ts)), key=lambda i: sum((rn[j] - (rn[:i + 1].mean() if j <= i else rn[i + 1:].mean())) ** 2 for j in range(n)))
    check(best_np == bi, "ngưỡng tốt nhất theo NumPy")
    check(abs(float(lv) - rn[:bi + 1].mean()) < 1e-9 and abs(float(rv) - rn[bi + 1:].mean()) < 1e-9, "giá trị lá NumPy")
    tbl = data_table(["x", "y"], [[F(v) for v in xs], ys])
    base = "Gradient boosting với hàm mất mát bình phương $\\ell = \\frac{1}{2}(y - F)^2$ (nên phần dư $r_i = y_i - F(x_i)$ chính là âm gradient). Khởi tạo $F_0$ là hằng số tối ưu (trung bình của $y$). "
    if level == 1:
        text, ans, steps = base + "Tính $F_0$. " + NUM_HINT, num_ans(f0), [Step(text="$F_0 = \\bar{y}$:", latex=f"F_0 = \\dfrac{{{ft(sum(ys))}}}{{{n}}} = {ft(f0)}")]
        tag = "f0"
    elif level == 2:
        i = rng.randrange(n)
        text = base + f"Tính phần dư $r = y - F_0$ của mẫu số ${i + 1}$ (đánh số theo bảng). " + NUM_HINT
        ans, steps = num_ans(res[i]), [Step(text=f"$F_0 = {ft(f0)}$:", latex=f"r_{i + 1} = {ft(ys[i])} - {fp(f0)} = {ft(res[i])}")]
        tag = "residual"
    elif level == 3:
        j = rng.randrange(len(ts))
        lv_j, rv_j, _ = _stump(xs, res, ts[j])
        text = base + f"Phần dư $r_i = y_i - F_0$ được khớp bằng một cây quyết định một nút (stump) tách ở $x \\le {ft(ts[j])}$; giá trị mỗi lá là trung bình phần dư trong lá. Tìm (giá trị lá trái, giá trị lá phải). " + VEC_HINT
        ans = vec_ans([lv_j, rv_j])
        steps = [Step(text=f"$F_0 = {ft(f0)}$, phần dư:", latex="r = (" + ", ".join(ft(r) for r in res) + ")"), Step(text="Trung bình phần dư mỗi nhánh:", latex=f"(h_L, h_R) = ({ft(lv_j)}, {ft(rv_j)})")]
        tag = "leaf"
    elif level == 4:
        pick3 = sorted(rng.sample(range(len(ts)), 3))
        if bi not in pick3:
            pick3[0] = bi
            pick3.sort()
        cands = [f"$x \\le {ft(ts[i])}$" for i in pick3]
        sc = [sses[i] for i in pick3]
        best = min(range(3), key=lambda i: sc[i])
        choices, ans = choice_exercise(rng, cands[best], [c for i, c in enumerate(cands) if i != best])
        text = base + "Một stump được khớp vào phần dư; chọn ngưỡng làm **tổng bình phương sai số (SSE)** của phần dư sau khi tách nhỏ nhất (giá trị mỗi lá là trung bình phần dư). Ngưỡng nào tốt nhất?"
        steps = [Step(text="SSE của phần dư với từng ngưỡng ứng viên:", latex=", ".join(f"{cands[i].strip('$')}: {ft(sc[i])}" for i in range(3))), Step(text=f"Tốt nhất: **{cands[best]}**.")]
        return cb(type_="gbm-round", lesson=L68, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=ans, choices=choices,
                  hints=["Tính phần dư $r_i = y_i - F_0$ trước.", "Với mỗi ngưỡng, giá trị mỗi lá là trung bình phần dư trong lá; SSE = tổng bình phương độ lệch so với giá trị lá."], steps=steps, verified=["numpy-threshold-scan"], tags=["best-threshold"])
    else:
        xq = F(rng.randint(1, 12)) + F(rng.choice([0, 1, 2]), 2)
        f1 = f0 + nu * (lv if xq <= ts[bi] else rv)
        text = (base + f"Một vòng boosting: khớp stump (chọn ngưỡng làm SSE phần dư nhỏ nhất, giá trị lá là trung bình phần dư), rồi cập nhật $F_1(x) = F_0 + \\nu\\,h(x)$ với tốc độ học (shrinkage) $\\nu = {ft(nu)}$. Tính $F_1({ft(xq)})$. " + NUM_HINT)
        ans = num_ans(f1)
        steps = [Step(text=f"$F_0 = {ft(f0)}$; ngưỡng tốt nhất $x \\le {ft(ts[bi])}$ (SSE nhỏ nhất ${ft(sses[bi])}$), giá trị lá $(h_L, h_R) = ({ft(lv)}, {ft(rv)})$."),
                 Step(text=f"$x = {ft(xq)}$ rơi vào lá " + ("trái" if xq <= ts[bi] else "phải") + ":", latex=f"F_1 = {ft(f0)} + {fp(nu)}\\cdot {fp(lv if xq <= ts[bi] else rv)} = {ft(f1)}")]
        tag = "round"
        return cb(type_="gbm-round", lesson=L68, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=ans,
                  hints=["Bước 1: $F_0 = \\bar{y}$ và phần dư $r_i = y_i - F_0$.", "Bước 2: thử mọi ngưỡng (trung điểm giữa hai $x$ liền kề), chọn ngưỡng cho SSE nhỏ nhất.", "Bước 3: $F_1 = F_0 + \\nu h$; shrinkage $\\nu$ làm mỗi cây chỉ sửa một phần sai số."],
                  steps=steps, verified=["numpy-threshold-scan", "leaf-means"], tags=["round"])
    return cb(type_="gbm-round", lesson=L68, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=ans,
              hints=["$F_0 = \\bar{y}$ (cực tiểu của $\\sum (y_i - c)^2$ theo hằng số $c$).", "Phần dư $r_i = y_i - F_0$.", "Giá trị lá = trung bình phần dư các mẫu trong lá."], steps=steps, verified=["numpy"], tags=[tag])


# -------------------------------------------------------------------- svm-margin
_PY = [(3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6), (8, 15), (15, 8), (9, 12), (12, 9)]


def _w_pyth(rng):
    a, b = rng.choice(_PY)
    return [F(a * rng.choice([1, -1])), F(b * rng.choice([1, -1]))]


def gen_svm_margin(rng: random.Random, level: int, seed: int):
    w = _w_pyth(rng)
    norm = F(isqrt(int(w[0] ** 2 + w[1] ** 2)))
    check(norm * norm == w[0] ** 2 + w[1] ** 2, "||w|| nguyên")
    b = F(rng.randint(-8, 8))
    hyper = f"\\mathbf{{w}}\\T\\mathbf{{x}} + b = 0,\\quad \\mathbf{{w}} = {vt(w)},\\ b = {ft(b)}"
    if level == 1:
        val = 2 / norm
        text = f"Siêu phẳng SVM ${hyper}$ với hai lề là $\\mathbf{{w}}\\T\\mathbf{{x}} + b = \\pm 1$. Tính độ rộng lề (khoảng cách giữa hai lề). " + NUM_HINT
        steps = [Step(text=f"$\\|\\mathbf{{w}}\\| = \\sqrt{{{ft(w[0] ** 2)} + {ft(w[1] ** 2)}}} = {ft(norm)}$ và độ rộng lề là $2/\\|\\mathbf{{w}}\\|$:", latex=f"\\dfrac{{2}}{{{ft(norm)}}} = {ft(val)}")]
        check(abs(float(val) - 2 / np.hypot(float(w[0]), float(w[1]))) < 1e-12, "độ rộng lề NumPy")
        return cb(type_="svm-margin", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Hai lề $\\mathbf{w}\\T\\mathbf{x} + b = \\pm 1$ cách siêu phẳng mỗi bên $1/\\|\\mathbf{w}\\|$.", "Độ rộng lề $= 2/\\|\\mathbf{w}\\|_2$."], steps=steps, verified=["numpy-norm"], tags=["margin-width"])
    x = [F(rng.randint(-6, 6)), F(rng.randint(-6, 6))]
    z = w[0] * x[0] + w[1] * x[1] + b
    if level == 2:
        val = abs(z) / norm
        text = f"Siêu phẳng ${hyper}$. Tính khoảng cách Euclid từ điểm $\\mathbf{{x}} = {vt(x)}$ đến siêu phẳng. " + NUM_HINT
        steps = [Step(text="Khoảng cách $= |\\mathbf{w}\\T\\mathbf{x} + b| / \\|\\mathbf{w}\\|$:", latex=f"\\dfrac{{|{ft(z)}|}}{{{ft(norm)}}} = {ft(val)}")]
        check(abs(float(val) - abs(float(z)) / np.hypot(float(w[0]), float(w[1]))) < 1e-12, "khoảng cách NumPy")
        return cb(type_="svm-margin", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Khoảng cách từ điểm đến siêu phẳng: $|\\mathbf{w}\\T\\mathbf{x} + b|/\\|\\mathbf{w}\\|$.", "$\\|\\mathbf{w}\\|$ ở đây là số nguyên (bộ ba Pythagoras)."], steps=steps, verified=["numpy-distance"], tags=["distance"])
    if level == 3:
        yv = rng.choice([1, -1])
        val = yv * z
        text = f"Siêu phẳng ${hyper}$. Điểm $\\mathbf{{x}} = {vt(x)}$ có nhãn $y = {yv:+d}$. Tính **lề hàm (functional margin)** $y\\,(\\mathbf{{w}}\\T\\mathbf{{x}} + b)$. " + NUM_HINT
        steps = [Step(text="Tính $z$ rồi nhân với nhãn:", latex=f"z = {ft(z)},\\quad y z = {yv}\\cdot {fp(z)} = {ft(val)}"),
                 Step(text=("Dương nên điểm được phân loại đúng" + (" và nằm ngoài lề (≥ 1)." if val >= 1 else " nhưng nằm trong lề (< 1).")) if val > 0 else "Âm nên điểm bị phân loại sai." if val < 0 else "Bằng $0$: điểm nằm trên siêu phẳng.")]
        return cb(type_="svm-margin", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Lề hàm $y(\\mathbf{w}\\T\\mathbf{x} + b) > 0$ nghĩa là phân loại đúng; $\\ge 1$ nghĩa là đúng và ngoài lề.", "Support vector có lề hàm bằng đúng $1$."], steps=steps, verified=["recompute"], tags=["functional-margin"])
    if level == 4:  # đếm support vector trong 5 điểm, trong đó có điểm đặt đúng trên lề
        pts, labs = [], []
        while len(pts) < 5:
            p = [F(rng.randint(-6, 6)), F(rng.randint(-6, 6))]
            if p in pts:
                continue
            pts.append(p)
            labs.append(rng.choice([1, -1]))
        # đặt 2 điểm đúng lên lề (đi theo hướng w): x = x0 + t w, với w.x0+b = 0
        for idx in (0, 1):
            yv = labs[idx]
            # tìm điểm nguyên nằm trên lề: w.x + b = yv
            sols = [[F(a), F(c)] for a in range(-8, 9) for c in range(-8, 9) if w[0] * a + w[1] * c + b == yv]
            if sols:
                pts[idx] = rng.choice(sols)
        cnt = sum(1 for p, yv in zip(pts, labs) if yv * (w[0] * p[0] + w[1] * p[1] + b) == 1)
        arr = np.array([[float(c) for c in p] for p in pts])
        check(cnt == int(np.sum(np.array(labs) * (arr @ np.array([float(c) for c in w]) + float(b)) == 1.0)), "đếm support vector NumPy")
        zs = [yv * (w[0] * p[0] + w[1] * p[1] + b) for p, yv in zip(pts, labs)]
        text = f"Siêu phẳng SVM ${hyper}$. Bảng dưới là $5$ điểm huấn luyện. Có bao nhiêu điểm là **support vector** (nằm đúng trên lề, tức $y(\\mathbf{{w}}\\T\\mathbf{{x}} + b) = 1$)? " + NUM_HINT
        steps = [Step(text="Lề hàm $y(\\mathbf{w}\\T\\mathbf{x} + b)$ của từng điểm:", latex="(" + ", ".join(ft(v) for v in zs) + ")"), Step(text=f"Có ${cnt}$ điểm với lề hàm đúng bằng $1$.")]
        return cb(type_="svm-margin", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text, display=[points_table(pts, [F(v) for v in labs])]), answer=num_ans(F(cnt)),
                  hints=["Support vector là điểm nằm đúng trên lề: $y(\\mathbf{w}\\T\\mathbf{x} + b) = 1$ (trong SVM lề cứng, chỉ chúng quyết định siêu phẳng).", "Tính lề hàm của từng điểm rồi đếm các giá trị bằng $1$."], steps=steps, verified=["numpy-count"], tags=["support-vectors"])
    # level 5: tìm w, b từ hai support vector (một dương, một âm)
    for _ in range(500):
        xp = [F(rng.randint(-4, 6)), F(rng.randint(-4, 6))]
        xn = [F(rng.randint(-4, 6)), F(rng.randint(-4, 6))]
        d = [a - c for a, c in zip(xp, xn)]
        dd = d[0] ** 2 + d[1] ** 2
        if dd == 0:
            continue
        w5 = [2 * di / dd for di in d]
        b5 = -(w5[0] * (xp[0] + xn[0]) + w5[1] * (xp[1] + xn[1])) / 2
        if nice(w5 + [b5], 25, 20) and d[0] != 0 and d[1] != 0:
            break
    check(w5[0] * xp[0] + w5[1] * xp[1] + b5 == 1 and w5[0] * xn[0] + w5[1] * xn[1] + b5 == -1, "hai support vector nằm trên hai lề")
    check(w5[0] * d[1] - w5[1] * d[0] == 0, "w song song với xp - xn")
    text = (f"Một SVM lề cứng có đúng hai support vector: $\\mathbf{{x}}_+ = {vt(xp)}$ (nhãn $+1$) và $\\mathbf{{x}}_- = {vt(xn)}$ (nhãn $-1$). Với hai điểm này, $\\mathbf{{w}}$ song song với $\\mathbf{{x}}_+ - \\mathbf{{x}}_-$ và hai điểm nằm đúng trên hai lề "
            "($\\mathbf{w}\\T\\mathbf{x}_+ + b = 1$, $\\mathbf{w}\\T\\mathbf{x}_- + b = -1$). Tìm $(w_1, w_2, b)$. " + VEC_HINT)
    steps = [Step(text="$\\mathbf{w} = \\lambda\\mathbf{d}$ với $\\mathbf{d} = \\mathbf{x}_+ - \\mathbf{x}_-$; trừ hai phương trình lề cho $\\mathbf{w}\\T\\mathbf{d} = 2$, nên $\\lambda = 2/\\|\\mathbf{d}\\|^2$:", latex=f"\\mathbf{{d}} = {vt(d)},\\ \\|\\mathbf{{d}}\\|^2 = {ft(dd)},\\ \\mathbf{{w}} = {vt(w5)}"),
             Step(text="Cộng hai phương trình lề: $\\mathbf{w}\\T(\\mathbf{x}_+ + \\mathbf{x}_-) + 2b = 0$:", latex=f"b = {ft(b5)}")]
    return cb(type_="svm-margin", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text), answer=vec_ans(w5 + [b5]),
              hints=["Hướng của $\\mathbf{w}$ là hướng nối hai support vector.", "Hai điều kiện lề: $\\mathbf{w}\\T\\mathbf{x}_+ + b = 1$ và $\\mathbf{w}\\T\\mathbf{x}_- + b = -1$. Trừ hai phương trình cho $\\mathbf{w}\\T\\mathbf{d} = 2$.", "Siêu phẳng đi qua trung điểm của hai support vector."],
              steps=steps, verified=["margin-conditions", "parallel-check"], tags=["fit-from-sv"])


# ----------------------------------------------------------------------- svm-hinge
def _hinge(ys, zs):
    return [max(F(0), 1 - y * z) for y, z in zip(ys, zs)]


def gen_svm_hinge(rng: random.Random, level: int, seed: int):
    n = {1: 1, 2: 4, 3: 4, 4: 4, 5: 5}[level]
    for _ in range(2000):
        w = [F(rng.randint(-3, 3)), F(rng.randint(-3, 3))]
        b = F(rng.randint(-3, 3))
        xs = [[F(rng.randint(-3, 3)), F(rng.randint(-3, 3))] for _ in range(n)]
        ys = [rng.choice([1, -1]) for _ in range(n)]
        zs = [w[0] * x[0] + w[1] * x[1] + b for x in xs]
        ms = [y * z for y, z in zip(ys, zs)]
        if any(m == 1 for m in ms) or (w[0] == 0 and w[1] == 0):
            continue
        viol = [i for i, m in enumerate(ms) if m < 1]
        if level >= 4 and not viol:
            continue
        if level == 2 and not (1 <= len(viol) < n):
            continue
        break
    hl = _hinge(ys, zs)
    C = F(rng.choice([1, 2, 5, 10, 1])) / rng.choice([1, 2, 5, 10]) if level >= 3 else None
    reg = (w[0] ** 2 + w[1] ** 2) / 2
    # kiểm chứng độc lập (NumPy)
    wa, xa, ya = np.array([float(c) for c in w]), np.array([[float(c) for c in x] for x in xs]), np.array(ys, dtype=float)
    hn = np.maximum(0, 1 - ya * (xa @ wa + float(b)))
    check(np.allclose(hn, [float(h) for h in hl]), "hinge NumPy")
    wstr = f"\\mathbf{{w}} = {vt(w)},\\ b = {ft(b)}"
    tbl = points_table(xs, [F(y) for y in ys])
    if level == 1:
        text = f"Cho $\\mathbf{{w}} = {vt(w)}$, $b = {ft(b)}$ và một mẫu $\\mathbf{{x}} = {vt(xs[0])}$ với $y = {ys[0]:+d}$. Tính hinge loss $\\max(0,\\ 1 - y(\\mathbf{{w}}\\T\\mathbf{{x}} + b))$. " + NUM_HINT
        val = hl[0]
        steps = [Step(text="$z = \\mathbf{w}\\T\\mathbf{x} + b$ rồi hinge loss:", latex=f"z = {ft(zs[0])},\\quad \\max(0,\\ 1 - {ys[0]}\\cdot {fp(zs[0])}) = {ft(val)}")]
        return cb(type_="svm-hinge", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Hinge loss $= \\max(0,\\ 1 - y z)$ với $z = \\mathbf{w}\\T\\mathbf{x} + b$.", "Loss bằng $0$ khi điểm đúng và nằm ngoài lề ($yz \\ge 1$)."], steps=steps, verified=["numpy"], tags=["single"])
    if level == 2:
        val = sum(hl)
        text = f"Với $\\mathbf{{w}} = {vt(w)}$, $b = {ft(b)}$, tính tổng hinge loss $\\sum_i \\max(0,\\ 1 - y_i(\\mathbf{{w}}\\T\\mathbf{{x}}_i + b))$ trên $4$ điểm dưới đây. " + NUM_HINT
        steps = [Step(text="$y_iz_i$ và hinge loss từng điểm:", latex="y z = (" + ", ".join(ft(m) for m in ms) + "),\\quad \\text{hinge} = (" + ", ".join(ft(h) for h in hl) + ")"), Step(text="Cộng lại:", latex=f"\\sum = {ft(val)}")]
        return cb(type_="svm-hinge", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=num_ans(val),
                  hints=["Tính $z_i = \\mathbf{w}\\T\\mathbf{x}_i + b$ rồi $\\max(0, 1 - y_iz_i)$ cho từng điểm.", "Điểm đúng và ngoài lề đóng góp $0$."], steps=steps, verified=["numpy"], tags=["sum"])
    if level in (3, 5):
        val = reg + C * sum(hl)
        text = (f"SVM lề mềm với mục tiêu $J = \\dfrac{{1}}{{2}}\\|\\mathbf{{w}}\\|^2 + C\\sum_i \\max(0,\\ 1 - y_i(\\mathbf{{w}}\\T\\mathbf{{x}}_i + b))$. Với $\\mathbf{{w}} = {vt(w)}$, $b = {ft(b)}$, $C = {ft(C)}$ và ${n}$ điểm dưới đây, tính $J$. " + NUM_HINT)
        steps = [Step(text="Số hạng chính quy hóa và tổng hinge loss:", latex=f"\\dfrac{{1}}{{2}}\\|\\mathbf{{w}}\\|^2 = {ft(reg)},\\quad \\sum\\text{{hinge}} = {ft(sum(hl))}"), Step(text="Ghép lại:", latex=f"J = {ft(reg)} + {fp(C)}\\cdot {ft(sum(hl))} = {ft(val)}")]
        return cb(type_="svm-hinge", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=num_ans(val),
                  hints=["$\\dfrac{1}{2}\\|\\mathbf{w}\\|^2 = \\dfrac{1}{2}(w_1^2 + w_2^2)$.", "$C$ lớn thì phạt vi phạm lề mạnh (lề hẹp hơn); $C$ nhỏ thì ưu tiên lề rộng."], steps=steps, verified=["numpy"], tags=["objective", f"n{n}"])
    # level 4: một bước gradient descent trên J
    eta = rng.choice([F(1, 10), F(1, 5), F(1, 4), F(1, 2)])
    gw = [w[j] - C * sum(ys[i] * xs[i][j] for i in viol) for j in range(2)]
    gb = -C * sum(ys[i] for i in viol)
    new = [w[0] - eta * gw[0], w[1] - eta * gw[1], b - eta * gb]
    # kiểm chứng độc lập: sai phân hữu hạn của J tại (w, b) (không có điểm nào nằm đúng trên lề)
    def J(wv, bv):
        return 0.5 * float(wv @ wv) + float(C) * np.maximum(0, 1 - ya * (xa @ wv + bv)).sum()
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        check(abs((J(wa + e, float(b)) - J(wa - e, float(b))) / (2 * h) - float(gw[j])) < 1e-5, "gradient w sai phân hữu hạn")
    check(abs((J(wa, float(b) + h) - J(wa, float(b) - h)) / (2 * h) - float(gb)) < 1e-5, "gradient b sai phân hữu hạn")
    text = (f"SVM lề mềm với $J = \\dfrac{{1}}{{2}}\\|\\mathbf{{w}}\\|^2 + C\\sum_i \\max(0,\\ 1 - y_i(\\mathbf{{w}}\\T\\mathbf{{x}}_i + b))$, $C = {ft(C)}$. Một bước gradient descent với $\\eta = {ft(eta)}$ từ $\\mathbf{{w}} = {vt(w)}$, $b = {ft(b)}$. "
            "Tìm $(w_1, w_2, b)$ mới (dùng gradient: các điểm có $y_i z_i < 1$ góp phần, các điểm còn lại góp $0$). " + VEC_HINT)
    steps = [Step(text=f"Các điểm vi phạm lề ($y_iz_i < 1$): " + ", ".join(str(i + 1) for i in viol) + ".", latex="y z = (" + ", ".join(ft(m) for m in ms) + ")"),
             Step(text="Gradient: $\\nabla_{\\mathbf{w}} J = \\mathbf{w} - C\\sum_{i\\in V} y_i\\mathbf{x}_i$, $\\partial J/\\partial b = -C\\sum_{i\\in V} y_i$:", latex=f"\\nabla_{{\\mathbf{{w}}}} J = {vt(gw)},\\quad \\dfrac{{\\partial J}}{{\\partial b}} = {ft(gb)}"),
             Step(text="Cập nhật:", latex=f"(w_1, w_2, b) = {vt(new)}")]
    return cb(type_="svm-hinge", lesson=L69, level=level, seed=seed, prompt=Prompt(text=text, display=[tbl]), answer=vec_ans(new),
              hints=["Chỉ các điểm vi phạm lề ($y_iz_i < 1$) có gradient hinge khác $0$: đạo hàm của $1 - y_i(\\mathbf{w}\\T\\mathbf{x}_i + b)$ theo $\\mathbf{w}$ là $-y_i\\mathbf{x}_i$ và theo $b$ là $-y_i$.", "Số hạng $\\frac{1}{2}\\|\\mathbf{w}\\|^2$ có gradient $\\mathbf{w}$ (không ảnh hưởng $b$).", "Cập nhật đồng thời $\\mathbf{w}$ và $b$ bằng giá trị cũ."],
              steps=steps, verified=["finite-difference", "numpy-hinge"], tags=["gd-step"])


SPECS = [
    TopicSpec("gbm-round", L68, (1, 2, 3, 4, 5), gen_gbm_round),
    TopicSpec("svm-margin", L69, (1, 2, 3, 4, 5), gen_svm_margin),
    TopicSpec("svm-hinge", L69, (1, 2, 3, 4, 5), gen_svm_hinge),
]
