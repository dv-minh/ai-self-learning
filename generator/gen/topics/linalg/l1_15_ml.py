"""Bài 1.15 Ma trận trong ML: ml-shape, normal-eq, attention-shape."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, sqrt

from ...core import latex as L
from ...core.nice import nontrivial_matrix
from ...core.steps import matmul_steps
from ...core.verify import check, np_close, to_np, verify_matmul
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, build, mat_ans, num_ans, shape_answer, shape_tex, vec_ans

LESSON = "1.15"
B, N = L.bold, L.num_paren


# ------------------------------------------------------------------ ml-shape
def gen_ml_shape(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        n, d, h = (rng.randint(2, 6) for _ in range(3))
        z = np.zeros((n, d)) @ np.zeros((d, h)) + np.zeros(h)
        return build(
            type_="ml-shape", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text=f"Một lớp tuyến tính tính $\\mathbf{{Z}} = \\mathbf{{X}}\\mathbf{{W}} + \\mathbf{{b}}$ với $\\mathbf{{X}}$ cỡ ${n}\\times {d}$ (một batch ${n}$ mẫu), "
                               f"$\\mathbf{{W}}$ cỡ ${d}\\times {h}$ và $\\mathbf{{b}}$ là vector ${h}$ phần tử (cộng vào từng hàng). $\\mathbf{{Z}}$ có kích thước bao nhiêu?"),
            answer=shape_answer(z.shape),
            hints=["Kích thước của $\\mathbf{X}\\mathbf{W}$ trước, rồi xem phép cộng $\\mathbf{b}$ có đổi kích thước không."],
            steps=[Step(text="$\\mathbf{X}\\mathbf{W}$:", latex=f"({n}\\times {d})({d}\\times {h}) \\to {n}\\times {h}"),
                   Step(text="Cộng $\\mathbf{b}$ vào từng hàng không đổi kích thước:", latex=shape_tex(z.shape))],
            verified=["numpy", "shape"], tags=["ml"],
        )
    if level == 3:
        n, d, h, k = (rng.randint(2, 6) for _ in range(4))
        kind = rng.choice(["out", "grad-w1", "grad-w2"])
        if kind == "out":
            res = (np.zeros((n, d)) @ np.zeros((d, h)) @ np.zeros((h, k))).shape
            q = f"Kích thước của $\\mathbf{{Y}} = \\phi(\\mathbf{{X}}\\mathbf{{W}}_1)\\mathbf{{W}}_2$ (với $\\phi$ là hàm kích hoạt áp dụng từng phần tử) là bao nhiêu?"
            why = [Step(text="$\\phi$ không đổi kích thước:", latex=f"\\mathbf{{X}}\\mathbf{{W}}_1: {n}\\times {h}"), Step(text="Nhân với $\\mathbf{W}_2$:", latex=f"({n}\\times {h})({h}\\times {k}) \\to {n}\\times {k}")]
        elif kind == "grad-w1":
            res = (np.zeros((d, n)) @ np.zeros((n, h))).shape
            q = f"Gradient của hàm mất mát theo $\\mathbf{{W}}_1$ có cùng kích thước với $\\mathbf{{W}}_1$ và bằng $\\mathbf{{X}}\\T\\boldsymbol{{\\delta}}_1$, với $\\boldsymbol{{\\delta}}_1$ cỡ ${n}\\times {h}$. Kích thước của $\\mathbf{{X}}\\T\\boldsymbol{{\\delta}}_1$?"
            why = [Step(text="$\\mathbf{X}\\T$ cỡ " + f"${d}\\times {n}$:", latex=f"({d}\\times {n})({n}\\times {h}) \\to {d}\\times {h}")]
        else:
            res = (np.zeros((h, n)) @ np.zeros((n, k))).shape
            q = f"Gradient theo $\\mathbf{{W}}_2$ bằng $\\mathbf{{A}}\\T\\boldsymbol{{\\delta}}_2$ với $\\mathbf{{A}} = \\phi(\\mathbf{{X}}\\mathbf{{W}}_1)$ cỡ ${n}\\times {h}$ và $\\boldsymbol{{\\delta}}_2$ cỡ ${n}\\times {k}$. Kích thước của $\\mathbf{{A}}\\T\\boldsymbol{{\\delta}}_2$?"
            why = [Step(text="$\\mathbf{A}\\T$ cỡ " + f"${h}\\times {n}$:", latex=f"({h}\\times {n})({n}\\times {k}) \\to {h}\\times {k}")]
        return build(
            type_="ml-shape", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text=f"Mạng hai lớp: $\\mathbf{{X}}$ cỡ ${n}\\times {d}$, $\\mathbf{{W}}_1$ cỡ ${d}\\times {h}$, $\\mathbf{{W}}_2$ cỡ ${h}\\times {k}$. " + q),
            answer=shape_answer(res),
            hints=["Viết kích thước các thừa số cạnh nhau, kiểm tra hai số ở giữa khớp nhau."],
            steps=why, verified=["numpy", "shape"], tags=["ml", kind],
        )
    if level == 4:  # đếm tham số
        layers = [rng.randint(2, 10) for _ in range(rng.randint(3, 4))]
        params = sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))
        shapes = [(np.zeros((layers[i], layers[i + 1])).size + layers[i + 1]) for i in range(len(layers) - 1)]
        check(sum(shapes) == params, "đếm tham số numpy")
        arch = " \\to ".join(str(x) for x in layers)
        return build(
            type_="ml-shape", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"Một mạng neural kết nối đầy đủ có số nơ-ron mỗi lớp lần lượt (từ đầu vào đến đầu ra): ${arch}$. "
                               f"Mỗi lớp tuyến tính có ma trận trọng số và vector hệ số bù (bias). Tổng số tham số học được là bao nhiêu?"),
            answer=num_ans(params),
            hints=["Lớp nối $a$ nơ-ron với $b$ nơ-ron có $a\\cdot b$ trọng số và $b$ hệ số bù.", "Cộng số tham số của mọi lớp."],
            steps=[Step(text=f"Lớp {i + 1}:", latex=f"{layers[i]}\\cdot {layers[i + 1]} + {layers[i + 1]} = {layers[i] * layers[i + 1] + layers[i + 1]}") for i in range(len(layers) - 1)]
                  + [Step(text="Tổng:", latex=f"{params}")],
            verified=["numpy"], tags=["ml", "params"],
        )
    # level 5: embedding rồi nối các token
    v, d, n, ln = rng.randint(5, 12), rng.randint(2, 6), rng.randint(2, 5), rng.randint(2, 5)
    emb = np.zeros((v, d))
    ids = np.zeros((n, ln), dtype=int)
    out = emb[ids].reshape(n, -1).shape
    return build(
        type_="ml-shape", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text=f"Bảng embedding $\\mathbf{{E}}$ có cỡ ${v}\\times {d}$ (mỗi hàng là vector của một token trong từ điển ${v}$ token). "
                           f"Một batch gồm ${n}$ câu, mỗi câu ${ln}$ token. Tra cứu embedding cho mọi token rồi nối các vector của ${ln}$ token trong mỗi câu thành một hàng. "
                           f"Ma trận kết quả có kích thước bao nhiêu?"),
        answer=shape_answer(out),
        hints=["Mỗi token cho một vector ${d}$ chiều.".replace("{d}", str(d)), "Mỗi câu nối các vector của mọi token thành một hàng."],
        steps=[Step(text=f"Mỗi câu có ${ln}$ token, mỗi token ${d}$ số:", latex=f"{ln}\\cdot {d} = {ln * d}"),
               Step(text="Số hàng là số câu:", latex=shape_tex(out))],
        verified=["numpy", "shape"], tags=["ml", "embedding"],
    )


# ------------------------------------------------------------------ normal-eq
def _fit(rng: random.Random, n: int) -> tuple[Matrix, Matrix, Matrix, Matrix, Matrix]:
    for _ in range(500):
        xs = sorted(rng.sample(range(-3, 5), n))
        ys = [rng.randint(-4, 6) for _ in range(n)]
        x = Matrix([[1, v] for v in xs])
        y = Matrix(ys)
        g = x.T * x
        if g.det() == 0:
            continue
        w = g.inv() * (x.T * y)
        if max(Rational(v).q for v in w) <= 12:
            return x, y, g, x.T * y, w
    raise RuntimeError("normal-eq")


def gen_normal_eq(rng: random.Random, level: int, seed: int) -> Exercise:
    n = 3 if level == 3 else 4
    x, y, g, xty, w = _fit(rng, n)
    check(g * w == xty, "(XᵀX)w = Xᵀy")
    check((x.T * (y - x * w)) == Matrix.zeros(2, 1), "phần dư vuông góc với cột của X")
    np_close(w, np.linalg.lstsq(to_np(x), to_np(y), rcond=None)[0])
    pts = ", ".join(f"({int(x[i, 1])}, {int(y[i])})" for i in range(n))
    steps = [Step(text="Ma trận thiết kế $\\mathbf{X}$ (cột $1$ cho hệ số chặn) và vector $\\mathbf{y}$:", latex=f"{L.mat(x, chr(92) + 'mathbf{X}')},\\quad {L.mat(y, chr(92) + 'mathbf{y}')}"),
             Step(text="$\\mathbf{X}\\T\\mathbf{X}$ và $\\mathbf{X}\\T\\mathbf{y}$:", latex=f"{L.mat(g)},\\quad {L.mat(xty)}"),
             Step(text="Nghịch đảo $2\\times 2$:", latex=L.mat(g.inv(), "(\\mathbf{X}\\T\\mathbf{X})^{-1}")),
             Step(text="$\\mathbf{w} = (\\mathbf{X}\\T\\mathbf{X})^{-1}\\mathbf{X}\\T\\mathbf{y}$:", latex=L.mat(w, "\\mathbf{w}"))]
    common = (f"Hồi quy tuyến tính $y = w_0 + w_1 x$ trên các điểm {pts}. Mô hình tìm $\\mathbf{{w}} = (w_0, w_1)\\T$ bằng phương trình chuẩn "
              f"$\\mathbf{{w}} = (\\mathbf{{X}}\\T\\mathbf{{X}})^{{-1}}\\mathbf{{X}}\\T\\mathbf{{y}}$, với $\\mathbf{{X}}$ có cột đầu toàn $1$.")
    if level == 5:
        x0 = rng.randint(-2, 6)
        pred = w[0] + w[1] * x0
        return build(
            type_="normal-eq", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text=common + f" Dự đoán $\\hat{{y}}$ tại $x = {x0}$."),
            answer=num_ans(pred),
            hints=["Tìm $\\mathbf{w}$ trước.", "Dự đoán là $\\hat{y} = w_0 + w_1 x$."],
            steps=steps + [Step(text=f"Dự đoán tại $x = {x0}$:", latex=f"\\hat{{y}} = {L.num(w[0])} + {N(w[1])}\\cdot {x0} = {L.num(pred)}")],
            verified=["sympy", "numpy-lstsq", "orthogonality"], tags=["ml", f"{n}pts", "predict"],
        )
    return build(
        type_="normal-eq", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=common + " Tính $\\mathbf{w}$."),
        answer=vec_ans(w),
        hints=["Lập $\\mathbf{X}$ (cột $1$ toàn số $1$, cột $2$ là các giá trị $x$) và $\\mathbf{y}$.", "Tính $\\mathbf{X}\\T\\mathbf{X}$, $\\mathbf{X}\\T\\mathbf{y}$ rồi giải hệ $2\\times 2$ (nghịch đảo bằng công thức)."],
        steps=steps, verified=["sympy", "numpy-lstsq", "orthogonality"], tags=["ml", f"{n}pts"],
    )


# ------------------------------------------------------------ attention-shape
def gen_attention(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        n, m, dk = rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 8)
        res = (np.zeros((n, dk)) @ np.zeros((m, dk)).T).shape
        return build(
            type_="attention-shape", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text=f"Trong attention, $\\mathbf{{Q}}$ có ${n}$ truy vấn và $\\mathbf{{K}}$ có ${m}$ khóa, mỗi vector có $d_k = {dk}$ chiều "
                               f"($\\mathbf{{Q}}$ cỡ ${n}\\times {dk}$, $\\mathbf{{K}}$ cỡ ${m}\\times {dk}$). Ma trận điểm $\\mathbf{{Q}}\\mathbf{{K}}\\T$ có kích thước bao nhiêu?"),
            answer=shape_answer(res),
            hints=["$\\mathbf{K}\\T$ có kích thước đảo lại của $\\mathbf{K}$."],
            steps=[Step(text="$\\mathbf{K}\\T$ cỡ " + f"${dk}\\times {m}$:", latex=f"({n}\\times {dk})({dk}\\times {m}) \\to {n}\\times {m}")],
            verified=["numpy", "shape"], tags=["ml"],
        )
    if level in (3, 4):
        dk = 2 if level == 3 else 4
        n, m = rng.randint(2, 3), rng.randint(2, 3)
        q = nontrivial_matrix(rng, n, dk, -2, 2)
        k = nontrivial_matrix(rng, m, dk, -2, 2)
        s = q * k.T
        verify_matmul(q, k.T, s)
        if level == 3:
            return build(
                type_="attention-shape", lesson=LESSON, level=3, seed=seed,
                prompt=Prompt(text="Tính ma trận điểm attention $\\mathbf{S} = \\mathbf{Q}\\mathbf{K}\\T$ (phần tử $s_{ij}$ là tích vô hướng của truy vấn $i$ với khóa $j$).",
                              display=[L.mat(q, B("Q")), L.mat(k, B("K"))]),
                answer=mat_ans(s),
                hints=["$s_{ij} = \\mathbf{q}_i\\T\\mathbf{k}_j$: hàng $i$ của $\\mathbf{Q}$ nhân hàng $j$ của $\\mathbf{K}$."],
                steps=[Step(text="$\\mathbf{K}\\T$:", latex=L.mat(k.T))] + matmul_steps(q, k.T, "\\mathbf{Q}", "\\mathbf{K}\\T")[1:],
                verified=["sympy", "numpy"], tags=["ml", f"{n}x{m}"],
            )
        scaled = s / sqrt(dk)
        check(all(Rational(x).q <= 2 for x in scaled), "chia √d_k hữu tỉ")
        np_close(scaled, to_np(q) @ to_np(k).T / np.sqrt(dk))
        return build(
            type_="attention-shape", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=f"Tính ma trận điểm đã co giãn $\\mathbf{{S}} = \\dfrac{{\\mathbf{{Q}}\\mathbf{{K}}\\T}}{{\\sqrt{{d_k}}}}$ với $d_k = {dk}$ (số cột của $\\mathbf{{Q}}$ và $\\mathbf{{K}}$).",
                          display=[L.mat(q, B("Q")), L.mat(k, B("K"))]),
            answer=mat_ans(scaled),
            hints=[f"$\\sqrt{{d_k}} = {sqrt(dk)}$.", "Tính $\\mathbf{Q}\\mathbf{K}\\T$ trước rồi chia mọi phần tử cho $\\sqrt{d_k}$."],
            steps=[Step(text="$\\mathbf{Q}\\mathbf{K}\\T$:", latex=L.mat(s))] +
                  [Step(text=f"Chia cho $\\sqrt{{d_k}} = {sqrt(dk)}$:", latex=L.mat(scaled, "\\mathbf{S}"))],
            verified=["sympy", "numpy"], tags=["ml", f"{n}x{m}", "scaled"],
        )
    # level 5: kích thước đầu ra multi-head
    n, m, heads, dv = rng.randint(2, 5), rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 5)
    out = np.concatenate([np.zeros((n, m)) @ np.zeros((m, dv)) for _ in range(heads)], axis=1).shape
    return build(
        type_="attention-shape", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text=f"Multi-head attention có ${heads}$ đầu. Mỗi đầu tính $\\mathrm{{softmax}}(\\mathbf{{S}})\\mathbf{{V}}$ với $\\mathbf{{S}}$ cỡ ${n}\\times {m}$ "
                           f"và $\\mathbf{{V}}$ cỡ ${m}\\times {dv}$ ($\\mathrm{{softmax}}$ áp dụng theo từng hàng, không đổi kích thước). "
                           f"Kết quả của các đầu được nối lại theo chiều cột. Ma trận cuối cùng có kích thước bao nhiêu?"),
        answer=shape_answer(out),
        hints=["Một đầu cho ma trận cỡ (số truy vấn) × $d_v$.", "Nối ${h}$ ma trận như vậy theo cột.".replace("{h}", str(heads))],
        steps=[Step(text="Một đầu:", latex=f"({n}\\times {m})({m}\\times {dv}) \\to {n}\\times {dv}"),
               Step(text=f"Nối ${heads}$ đầu theo cột:", latex=f"{n}\\times ({heads}\\cdot {dv}) = {shape_tex(out)}")],
        verified=["numpy", "shape"], tags=["ml", "multihead"],
    )


SPECS = [
    TopicSpec("ml-shape", LESSON, (2, 3, 4, 5), gen_ml_shape),
    TopicSpec("normal-eq", LESSON, (3, 4, 5), gen_normal_eq),
    TopicSpec("attention-shape", LESSON, (2, 3, 4, 5), gen_attention),
]
