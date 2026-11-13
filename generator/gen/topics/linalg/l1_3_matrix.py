"""Bài 1.3 Ma trận và ký hiệu: mat-shape, mat-entry, mat-transpose."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix

from ...core import latex as L
from ...core.nice import nontrivial_matrix
from ...core.verify import check, np_close, to_np
from ...schema import Exercise, Prompt, Step
from ._base import TopicSpec, build, mat_ans, num_ans, shape_answer, shape_tex, size_tag

LESSON = "1.3"
B = L.bold


def _mat(rng: random.Random, r: int, c: int, lo: int = -9, hi: int = 9, fp: float = 0.0) -> Matrix:
    return nontrivial_matrix(rng, r, c, lo, hi, fp) if r * c > 1 else Matrix([[rng.choice([x for x in range(lo, hi + 1) if x])]])


def gen_shape(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 5:
        n, h, w = rng.randint(2, 4), rng.randint(1, 4), rng.randint(1, 4)
        d = h * w
        flat = np.zeros((n, h, w)).reshape(n, -1)
        check(flat.shape == (n, d), "flatten numpy")
        ask_t = rng.random() < 0.5
        res = (d, n) if ask_t else (n, d)
        what = "$\\mathbf{X}\\T$" if ask_t else "$\\mathbf{X}$"
        return build(
            type_="mat-shape", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text=f"Một tập dữ liệu có ${n}$ ảnh xám, mỗi ảnh là một lưới ${h}\\times {w}$ điểm ảnh. "
                               f"Mỗi ảnh được trải phẳng thành một hàng của ma trận dữ liệu $\\mathbf{{X}}$ "
                               f"(mỗi hàng là một mẫu). Ma trận {what} có kích thước bao nhiêu?"),
            answer=shape_answer(res),
            hints=["Số hàng của $\\mathbf{X}$ bằng số mẫu.", "Số cột của $\\mathbf{X}$ bằng số điểm ảnh của một ảnh (hàng nhân cột của ảnh)."]
                  + (["Chuyển vị đổi chỗ số hàng và số cột."] if ask_t else []),
            steps=[Step(text="Mỗi ảnh có số điểm ảnh bằng:", latex=f"{h}\\cdot {w} = {d}"),
                   Step(text="Số hàng của $\\mathbf{X}$ là số ảnh, số cột là số điểm ảnh:", latex=shape_tex((n, d)))]
                  + ([Step(text="Chuyển vị đảo hai số:", latex=shape_tex(res))] if ask_t else []),
            verified=["numpy", "shape"], tags=["ml"],
        )
    r, c = rng.randint(1, 4), rng.randint(1, 4)
    a = _mat(rng, r, c)
    if level == 1:
        return build(
            type_="mat-shape", lesson=LESSON, level=1, seed=seed,
            prompt=Prompt(text="Ma trận $\\mathbf{A}$ có kích thước $m \\times n$. Tìm $m$ và $n$.", display=[L.mat(a, B("A"))]),
            answer=shape_answer((r, c)),
            hints=["$m$ là số hàng, $n$ là số cột."],
            steps=[Step(text=f"Có {r} hàng và {c} cột:", latex=shape_tex((r, c)))],
            verified=["numpy", "shape"], tags=[size_tag(a)],
        )
    # level 2: số phần tử
    check(to_np(a).size == r * c, "size numpy")
    return build(
        type_="mat-shape", lesson=LESSON, level=2, seed=seed,
        prompt=Prompt(text="Ma trận $\\mathbf{A}$ có tất cả bao nhiêu phần tử?", display=[L.mat(a, B("A"))]),
        answer=num_ans(r * c),
        hints=["Số phần tử bằng số hàng nhân số cột."],
        steps=[Step(text=f"Ma trận có {r} hàng và {c} cột:", latex=f"{r} \\cdot {c} = {r * c}")],
        verified=["numpy", "shape"], tags=[size_tag(a)],
    )


def gen_entry(rng: random.Random, level: int, seed: int) -> Exercise:
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    if level == 1:
        r, c = rng.randint(2, 3), rng.randint(2, 3)
    a = _mat(rng, r, c)
    i, j = rng.randrange(r), rng.randrange(c)
    if level == 3:  # phần tử của chuyển vị
        t = a.T
        i, j = rng.randrange(t.rows), rng.randrange(t.cols)
        check(t[i, j] == a[j, i] and float(t[i, j]) == to_np(a).T[i, j], "entry T")
        return build(
            type_="mat-entry", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text=f"Tìm phần tử $(\\mathbf{{A}}\\T)_{{{i + 1}{j + 1}}}$, tức hàng {i + 1}, cột {j + 1} của $\\mathbf{{A}}\\T$.",
                          display=[L.mat(a, B("A"))]),
            answer=num_ans(t[i, j]),
            hints=["$(\\mathbf{A}\\T)_{ij} = a_{ji}$: đổi chỗ chỉ số hàng và cột."],
            steps=[Step(text=f"$(\\mathbf{{A}}\\T)_{{{i + 1}{j + 1}}} = a_{{{j + 1}{i + 1}}}$, là phần tử hàng {j + 1}, cột {i + 1} của $\\mathbf{{A}}$:",
                        latex=f"a_{{{j + 1}{i + 1}}} = {L.num(a[j, i])}")],
            verified=["sympy", "numpy"], tags=[size_tag(a)],
        )
    check(float(a[i, j]) == to_np(a)[i, j], "entry numpy")
    return build(
        type_="mat-entry", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text=f"Tìm phần tử $a_{{{i + 1}{j + 1}}}$ (hàng {i + 1}, cột {j + 1}).", display=[L.mat(a, B("A"))]),
        answer=num_ans(a[i, j]),
        hints=["Chỉ số đầu là hàng, chỉ số sau là cột. Hàng và cột đều đếm từ $1$ trong toán học."],
        steps=[Step(text=f"Hàng {i + 1}, cột {j + 1}:", latex=f"a_{{{i + 1}{j + 1}}} = {L.num(a[i, j])}")],
        verified=["sympy", "numpy"], tags=[size_tag(a)],
    )


def gen_transpose(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 1:
        r, c = rng.randint(1, 3), rng.randint(1, 3)
        lo, hi, fp = -9, 9, 0.0
    elif level == 2:
        r, c = rng.choice([(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3), (1, 4), (4, 1)]), None
        r, c = r
        lo, hi, fp = -5, 5, 0.0
    else:
        r, c = rng.randint(3, 4), rng.randint(3, 4)
        lo, hi, fp = -9, 9, 0.2
    a = _mat(rng, r, c, lo, hi, fp)
    t = a.T
    np_close(t, to_np(a).T)
    check(t.T == a, "(A^T)^T = A")
    first_row = ", ".join(L.num(x) for x in a[0, :])
    return build(
        type_="mat-transpose", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tính $\\mathbf{A}\\T$ (ma trận chuyển vị).", display=[L.mat(a, B("A"))]),
        answer=mat_ans(t),
        hints=["Hàng $i$ của $\\mathbf{A}$ trở thành cột $i$ của $\\mathbf{A}\\T$.",
               f"Kích thước mới là ${c}\\times {r}$."],
        steps=[Step(text=f"Hàng 1 của $\\mathbf{{A}}$ là $({first_row})$, nó trở thành cột 1 của $\\mathbf{{A}}\\T$."),
               Step(text="Làm tương tự với các hàng còn lại:", latex=L.mat(t, "\\mathbf{A}\\T"))],
        verified=["sympy", "numpy", "shape"], tags=[size_tag(a), "→" + size_tag(t)],
    )


SPECS = [
    TopicSpec("mat-shape", LESSON, (1, 2, 5), gen_shape),
    TopicSpec("mat-entry", LESSON, (1, 2, 3), gen_entry),
    TopicSpec("mat-transpose", LESSON, (1, 2, 3), gen_transpose),
]
