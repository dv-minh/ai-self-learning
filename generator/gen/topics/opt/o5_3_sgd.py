"""Bài 5.3 SGD và mini-batch: sgd-minibatch, sgd-epoch, sgd-unbiased."""
from __future__ import annotations

from itertools import combinations

from sympy import Rational, symbols

from ._o import F, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, fp, ft, nice, num_ans, ob, pick, random, vec_ans

L53 = "5.3"
w_, b_ = symbols("w b")
ETAS = [F(1, 2), F(1, 4), F(1, 8), F(1, 5), F(1, 10), F(1, 20)]
POW2 = [F(1, 2), F(1, 4), F(1, 8), F(1)]


def _q(v: F) -> Rational:
    return Rational(v.numerator, v.denominator)


def _data(rng, n):
    xs = [F(pick(rng, -5, 5)) for _ in range(n)]
    while len(set(xs)) < 2:
        xs = [F(pick(rng, -5, 5)) for _ in range(n)]
    return xs, [F(rng.randint(-9, 9)) for _ in range(n)]


def _table(xs, ys) -> str:
    n = len(xs)
    return ("\\begin{array}{c|" + "c" * n + "} i & " + " & ".join(str(i + 1) for i in range(n)) + " \\\\ x_i & " + " & ".join(ft(x) for x in xs)
            + " \\\\ y_i & " + " & ".join(ft(y) for y in ys) + " \\end{array}")


def _batch_grad(xs, ys, idx, w, b, with_b):
    """Gradient của mất mát mini-batch (1/|B|) sum 1/2 (w x + b - y)^2 theo (w, b), kèm kiểm chứng SymPy."""
    res = [w * xs[i] + b - ys[i] for i in idx]
    gw = sum(r * xs[i] for r, i in zip(res, idx)) / len(idx)
    gb = sum(res) / len(idx) if with_b else F(0)
    loss = sum((w_ * _q(xs[i]) + (b_ if with_b else 0) - _q(ys[i])) ** 2 for i in idx) / (2 * len(idx))
    sub = {w_: _q(w), b_: _q(b)}
    check(loss.diff(w_).subs(sub) == _q(gw), "gradient theo w sai")
    if with_b:
        check(loss.diff(b_).subs(sub) == _q(gb), "gradient theo b sai")
    return gw, gb, res


# --------------------------------------------------------------- sgd-minibatch
def gen_sgd_minibatch(rng: random.Random, level: int, seed: int):
    with_b = level == 3
    n = {1: 4, 2: 5, 3: 5}[level]
    bsz = {1: 2, 2: 3, 3: rng.choice([2, 3])}[level]
    for _ in range(300):
        xs, ys = _data(rng, n)
        idx = sorted(rng.sample(range(n), bsz))
        w0, b0 = F(rng.randint(-3, 3)), (F(rng.randint(-3, 3)) if with_b else F(0))
        eta = rng.choice(ETAS)
        gw, gb, res = _batch_grad(xs, ys, idx, w0, b0, with_b)
        w1, b1 = w0 - eta * gw, b0 - eta * gb
        if nice([w1, b1], 300, 100):
            break
    names = ", ".join(str(i + 1) for i in idx)
    model = "\\hat{y} = wx + b" if with_b else "\\hat{y} = wx"
    start = f"$(w, b) = ({ft(w0)}, {ft(b0)})$" if with_b else f"$w = {ft(w0)}$"
    steps = [Step(text=f"Sai số dự đoán $e_i = \\hat{{y}}_i - y_i$ của các mẫu trong mini-batch ({names}):", latex="e = (" + ", ".join(ft(r) for r in res) + ")"),
             Step(text="Gradient của mất mát mini-batch (trung bình trên các mẫu **trong batch**):",
                  latex=f"\\dfrac{{\\partial \\ell}}{{\\partial w}} = \\dfrac{{1}}{{{bsz}}}\\sum e_i x_i = {ft(gw)}" + (f",\\qquad \\dfrac{{\\partial \\ell}}{{\\partial b}} = \\dfrac{{1}}{{{bsz}}}\\sum e_i = {ft(gb)}" if with_b else "")),
             Step(text="Cập nhật:", latex=f"w \\leftarrow {ft(w0)} - {fp(eta)}\\cdot {fp(gw)} = {ft(w1)}" + (f",\\qquad b \\leftarrow {ft(b0)} - {fp(eta)}\\cdot {fp(gb)} = {ft(b1)}" if with_b else ""))]
    return ob(type_="sgd-minibatch", lesson=L53, level=level, seed=seed,
              prompt=Prompt(text=f"Mô hình ${model}$, mất mát trên một mini-batch $\\mathcal{{B}}$ là $\\ell = \\dfrac{{1}}{{|\\mathcal{{B}}|}}\\sum_{{i \\in \\mathcal{{B}}}} \\dfrac{{1}}{{2}}(\\hat{{y}}_i - y_i)^2$. "
                                 f"Một bước SGD với mini-batch gồm các mẫu số ${names}$ (đánh số theo bảng), $\\eta = {ft(eta)}$, bắt đầu từ {start}. Tính " + ("$(w, b)$ mới. " + VEC_HINT if with_b else "$w$ mới. " + NUM_HINT),
                            display=[_table(xs, ys)]),
              answer=vec_ans([w1, b1]) if with_b else num_ans(w1),
              hints=["Chỉ dùng các mẫu trong mini-batch, các mẫu khác **bỏ qua** ở bước này.", "$\\dfrac{\\partial \\ell}{\\partial w} = \\dfrac{1}{|\\mathcal{B}|}\\sum_{i\\in\\mathcal{B}} e_i x_i$ với $e_i = \\hat{y}_i - y_i$; chia cho số mẫu **trong batch**, không phải toàn bộ dữ liệu.",
                     "Cập nhật: tham số $\\leftarrow$ tham số $- \\eta \\times$ gradient."],
              steps=steps, verified=["sympy-gradient", "fraction-simulation"], tags=["ml", "with-b" if with_b else "no-b", f"batch{bsz}"])


# ------------------------------------------------------------------- sgd-epoch
def gen_sgd_epoch(rng: random.Random, level: int, seed: int):
    n, bsz = {1: (4, 2), 2: (4, 2), 3: (4, 2), 4: (4, 1), 5: (6, 2)}[level]
    with_b = level >= 3
    for _ in range(500):
        xs, ys = _data(rng, n)
        w, b = F(rng.randint(-3, 3)), (F(rng.randint(-3, 3)) if with_b else F(0))
        eta = rng.choice(POW2 if level >= 4 else ETAS)
        rows = []
        for s in range(0, n, bsz):
            idx = list(range(s, s + bsz))
            gw, gb, res = _batch_grad(xs, ys, idx, w, b, with_b)
            rows.append((idx, w, b, gw, gb))
            w, b = w - eta * gw, b - eta * gb
        if nice([w, b] + [r[3] for r in rows], 2000 if level >= 4 else 300, 200):
            break
    steps = []
    for k, (idx, wk, bk, gw, gb) in enumerate(rows, 1):
        nm = ", ".join(str(i + 1) for i in idx)
        nxt_w = wk - eta * gw
        nxt_b = bk - eta * gb
        steps.append(Step(text=f"Bước {k} (mini-batch: mẫu {nm}), gradient tại hiện tại là ${ft(gw)}$" + (f" (theo $w$) và ${ft(gb)}$ (theo $b$)" if with_b else "") + ":",
                          latex=f"w \\leftarrow {ft(wk)} - {fp(eta)}\\cdot {fp(gw)} = {ft(nxt_w)}" + (f",\\qquad b \\leftarrow {ft(bk)} - {fp(eta)}\\cdot {fp(gb)} = {ft(nxt_b)}" if with_b else "")))
    model = "\\hat{y} = wx + b" if with_b else "\\hat{y} = wx"
    w0t = ft(rows[0][1])
    start = f"$(w, b) = ({ft(rows[0][1])}, {ft(rows[0][2])})$" if with_b else f"$w = {w0t}$"
    return ob(type_="sgd-epoch", lesson=L53, level=level, seed=seed,
              prompt=Prompt(text=f"Mô hình ${model}$ với mất mát mỗi mini-batch $\\ell = \\dfrac{{1}}{{|\\mathcal{{B}}|}}\\sum \\dfrac{{1}}{{2}}(\\hat{{y}}_i - y_i)^2$. Chạy **một epoch** SGD (không xáo trộn, cỡ batch ${bsz}$, tức các mẫu "
                                 f"theo thứ tự trong bảng chia thành {n // bsz} batch liên tiếp), $\\eta = {ft(eta)}$, bắt đầu từ {start}. Tính " + ("$(w, b)$ sau epoch. " + VEC_HINT if with_b else "$w$ sau epoch. " + NUM_HINT),
                            display=[_table(xs, ys)]),
              answer=vec_ans([w, b]) if with_b else num_ans(w),
              hints=["Một epoch = đi qua toàn bộ dữ liệu một lần; có $N / B$ lần cập nhật.", "Mỗi lần cập nhật dùng tham số **mới nhất** (sau lần cập nhật trước), chỉ tính gradient trên batch hiện tại.",
                     "$\\dfrac{\\partial \\ell}{\\partial w} = \\dfrac{1}{B}\\sum e_i x_i$, $\\dfrac{\\partial \\ell}{\\partial b} = \\dfrac{1}{B}\\sum e_i$."],
              steps=steps, verified=["sympy-gradient", "fraction-simulation"], tags=["ml", f"batch{bsz}", f"{n // bsz}updates"])


# ---------------------------------------------------------------- sgd-unbiased
def gen_sgd_unbiased(rng: random.Random, level: int, seed: int):
    n, bsz = {1: (3, 2), 2: (4, 2), 3: (4, 3), 4: (5, 2), 5: (6, 3)}[level]
    xs, ys = _data(rng, n)
    w = F(rng.randint(-3, 3))
    g = [(w * x - y) * x for x, y in zip(xs, ys)]  # gradient theo w của 1/2 (w x - y)^2 cho từng mẫu
    for i in range(n):
        loss_i = (w_ * _q(xs[i]) - _q(ys[i])) ** 2 / 2
        check(loss_i.diff(w_).subs({w_: _q(w)}) == _q(g[i]), "gradient từng mẫu sai")
    batches = list(combinations(range(n), bsz))
    means = [sum(g[i] for i in bt) / bsz for bt in batches]
    avg = sum(means) / len(means)
    full = sum(g) / n
    check(avg == full, "trung bình gradient mọi batch ≠ gradient toàn bộ")
    rows = "; ".join(f"\\{{{','.join(str(i + 1) for i in bt)}\\}}: {ft(m)}" for bt, m in zip(batches, means)) if len(batches) <= 6 else None
    steps = [Step(text="Gradient riêng của từng mẫu $g_i = (wx_i - y_i)\\,x_i$ tại $w$ đã cho:", latex="g = (" + ", ".join(ft(v) for v in g) + ")"),
             Step(text=f"Có $\\binom{{{n}}}{{{bsz}}} = {len(batches)}$ mini-batch. Mỗi mẫu xuất hiện trong cùng số batch ({len(batches) * bsz // n}), nên trung bình gradient các batch bằng trung bình gradient toàn bộ dữ liệu:",
                  latex=f"\\dfrac{{1}}{{{len(batches)}}}\\sum_{{\\mathcal{{B}}}} \\dfrac{{1}}{{{bsz}}}\\sum_{{i\\in\\mathcal{{B}}}} g_i = \\dfrac{{1}}{{{n}}}\\sum_{{i=1}}^{{{n}}} g_i = {ft(full)}")]
    return ob(type_="sgd-unbiased", lesson=L53, level=level, seed=seed,
              prompt=Prompt(text=f"Mô hình $\\hat{{y}} = wx$, mất mát mỗi mẫu $\\dfrac{{1}}{{2}}(\\hat{{y}}_i - y_i)^2$, tại $w = {ft(w)}$. Với mọi cách chọn mini-batch ${bsz}$ mẫu từ ${n}$ mẫu (không hoàn lại, mỗi tập con xuất hiện một lần), "
                                 "tính gradient theo $w$ của mất mát trung bình trong batch, rồi **lấy trung bình trên mọi batch**. Kết quả là bao nhiêu? " + NUM_HINT, display=[_table(xs, ys)]),
              answer=num_ans(avg),
              hints=["Gradient mini-batch là ước lượng **không chệch (unbiased)** của gradient toàn bộ dữ liệu.", "Mỗi mẫu xuất hiện trong số batch như nhau, nên trung bình trên mọi batch chính là gradient trung bình trên toàn bộ dữ liệu.",
                     "Tính $g_i = (wx_i - y_i)x_i$ cho từng mẫu rồi lấy trung bình."],
              steps=steps, verified=["enumeration-all-batches", "sympy-gradient"], tags=["ml", "unbiased", f"n{n}b{bsz}"])


SPECS = [
    TopicSpec("sgd-minibatch", L53, (1, 2, 3), gen_sgd_minibatch),
    TopicSpec("sgd-epoch", L53, (1, 2, 3, 4, 5), gen_sgd_epoch),
    TopicSpec("sgd-unbiased", L53, (1, 2, 3, 4, 5), gen_sgd_unbiased),
]
