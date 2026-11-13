"""Bài 1.11 Ánh xạ tuyến tính: transform-apply, transform-find, transform-compose."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, diag

from ...core import latex as L
from ...core.nice import nontrivial_matrix, rand_vector
from ...core.steps import matmul_steps
from ...core.verify import check, np_close, to_np, verify_matmul
from ...schema import Exercise, Prompt, Step
from ._base import TopicSpec, build, mat_ans, num_ans, vec_ans

LESSON = "1.11"
B, N = L.bold, L.num_paren

# Danh mục phép biến đổi 2D: mô tả (tiếng Việt, dùng trong chữ thường) -> ma trận
_s = Rational(3, 5)
_c = Rational(4, 5)
ROT_RATIONAL = Matrix([[_c, -_s], [_s, _c]])
CATALOG_2D: dict[str, Matrix] = {
    "quay $90^\\circ$ ngược chiều kim đồng hồ quanh gốc": Matrix([[0, -1], [1, 0]]),
    "quay $180^\\circ$ quanh gốc": Matrix([[-1, 0], [0, -1]]),
    "quay $90^\\circ$ cùng chiều kim đồng hồ quanh gốc": Matrix([[0, 1], [-1, 0]]),
    "phản xạ qua trục $Ox$": Matrix([[1, 0], [0, -1]]),
    "phản xạ qua trục $Oy$": Matrix([[-1, 0], [0, 1]]),
    "phản xạ qua đường thẳng $y = x$": Matrix([[0, 1], [1, 0]]),
    "chiếu vuông góc lên trục $Ox$": Matrix([[1, 0], [0, 0]]),
    "chiếu vuông góc lên trục $Oy$": Matrix([[0, 0], [0, 1]]),
}


def _build_pool() -> list[tuple[str, Matrix]]:
    pool = [(k, m) for k, m in CATALOG_2D.items()]
    pool += [
        ("phản xạ qua đường thẳng $y = -x$", Matrix([[0, -1], [-1, 0]])),
        ("chiếu vuông góc lên đường thẳng $y = x$", Matrix([[1, 1], [1, 1]]) / 2),
        ("chiếu vuông góc lên đường thẳng $y = -x$", Matrix([[1, -1], [-1, 1]]) / 2),
        ("quay ngược chiều kim đồng hồ một góc $\\theta$ với $\\cos\\theta = \\frac{4}{5}$, $\\sin\\theta = \\frac{3}{5}$", ROT_RATIONAL),
        ("quay cùng chiều kim đồng hồ một góc $\\theta$ với $\\cos\\theta = \\frac{4}{5}$, $\\sin\\theta = \\frac{3}{5}$", ROT_RATIONAL.T),
        ("quay ngược chiều kim đồng hồ một góc $\\theta$ với $\\cos\\theta = \\frac{3}{5}$, $\\sin\\theta = \\frac{4}{5}$", Matrix([[_s, -_c], [_c, _s]])),
        ("quay cùng chiều kim đồng hồ một góc $\\theta$ với $\\cos\\theta = \\frac{3}{5}$, $\\sin\\theta = \\frac{4}{5}$", Matrix([[_s, _c], [-_c, _s]])),
    ]
    vals = [2, 3, 4, -1, -2, -3]
    for a in vals:
        for b in vals:
            pool.append((f"co giãn theo $Ox$ hệ số ${a}$ và theo $Oy$ hệ số ${b}$", diag(a, b)))
    for k in (1, 2, 3, 4, -1, -2, -3, -4):
        sg = "+" if k > 0 else "-"
        pool.append((f"phép trượt (shear) $(x, y) \\mapsto (x {sg} {abs(k)}y,\\; y)$", Matrix([[1, k], [0, 1]])))
        pool.append((f"phép trượt (shear) $(x, y) \\mapsto (x,\\; y {sg} {abs(k)}x)$", Matrix([[1, 0], [k, 1]])))
    for k in (2, 3, 4, 5, -1, -2, -3):
        pool.append((f"co giãn đều với hệ số ${k}$ (nhân mọi vector với ${k}$)", diag(k, k)))
    return pool


POOL = _build_pool()


def _named(rng: random.Random) -> tuple[str, Matrix]:
    return rng.choice(POOL)


# ------------------------------------------------------------ transform-apply
def gen_apply(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 1:
        _desc, a = _named(rng)
        v = rand_vector(rng, 2, -5, 5)
    elif level == 2:
        a = nontrivial_matrix(rng, 2, 2, -3, 3)
        v = rand_vector(rng, 2, -5, 5)
    elif level == 3:
        kind = rng.choice(["rot", "proj", "scale", "refl", "generic"])
        a = {"rot": Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
             "proj": diag(1, 1, 0),
             "scale": diag(*[rng.choice([2, 3, -1]) for _ in range(3)]),
             "refl": diag(1, 1, -1),
             "generic": nontrivial_matrix(rng, 3, 3, -2, 2)}[kind]
        v = rand_vector(rng, 3, -5, 5)
    elif level == 4:  # hệ số co giãn diện tích
        a = nontrivial_matrix(rng, 2, 2, -4, 4)
        d = a.det()
        check(np.isclose(abs(np.linalg.det(to_np(a))), float(abs(d))), "|det| numpy")
        return build(
            type_="transform-apply", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Phép biến đổi $T(\\mathbf{x}) = \\mathbf{A}\\mathbf{x}$ biến hình vuông đơn vị (cạnh nằm trên $\\mathbf{e}_1, \\mathbf{e}_2$) "
                               "thành một hình bình hành. Tính diện tích hình bình hành đó.", display=[L.mat(a, B("A"))]),
            answer=num_ans(abs(d)),
            hints=["Ảnh của $\\mathbf{e}_1, \\mathbf{e}_2$ là hai cột của $\\mathbf{A}$; chúng là hai cạnh của hình bình hành.",
                   "Diện tích bằng giá trị tuyệt đối của định thức."],
            steps=[Step(text="Định thức:", latex=f"\\det(\\mathbf{{A}}) = {N(a[0, 0])}\\cdot {N(a[1, 1])} - {N(a[0, 1])}\\cdot {N(a[1, 0])} = {L.num(d)}"),
                   Step(text="Diện tích bằng giá trị tuyệt đối:", latex=f"|\\det(\\mathbf{{A}})| = {L.num(abs(d))}")],
            verified=["sympy", "numpy"], tags=["area", "2x2"],
        )
    else:
        raise ValueError(level)
    res = a * v
    verify_matmul(a, v, res)
    n = a.rows
    return build(
        type_="transform-apply", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Phép biến đổi tuyến tính $T(\\mathbf{x}) = \\mathbf{A}\\mathbf{x}$. Tính $T(\\mathbf{v})$.",
                      display=[L.mat(a, B("A")), L.mat(v, B("v"))]),
        answer=vec_ans(res),
        hints=["$T(\\mathbf{v})$ chính là tích ma trận–vector $\\mathbf{A}\\mathbf{v}$."],
        steps=matmul_steps(a, v, "\\mathbf{A}", "\\mathbf{v}"),
        verified=["sympy", "numpy"], tags=[f"{n}x{n}"],
    )


# ------------------------------------------------------------- transform-find
def gen_find(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:  # biết T(e1), T(e2)
        a = nontrivial_matrix(rng, 2, 2, -4, 4)
        t1, t2 = a[:, 0], a[:, 1]
        return build(
            type_="transform-find", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text="$T$ là phép biến đổi tuyến tính trên $\\R^{2}$ với $T(\\mathbf{e}_1)$ và $T(\\mathbf{e}_2)$ cho dưới đây "
                               "(với $\\mathbf{e}_1 = (1, 0)\\T$, $\\mathbf{e}_2 = (0, 1)\\T$). Tìm ma trận $\\mathbf{A}$ của $T$.",
                          display=[L.mat(t1, "T(\\mathbf{e}_1)"), L.mat(t2, "T(\\mathbf{e}_2)")]),
            answer=mat_ans(a),
            hints=["Cột $j$ của ma trận chính là ảnh của vector cơ sở thứ $j$."],
            steps=[Step(text="Cột 1 là $T(\\mathbf{e}_1)$, cột 2 là $T(\\mathbf{e}_2)$:", latex=L.mat(a, "\\mathbf{A}"))],
            verified=["sympy"], tags=["2x2", "columns"],
        )
    if level == 3:  # mô tả bằng lời
        desc, a = _named(rng)
        e1, e2 = Matrix([1, 0]), Matrix([0, 1])
        return build(
            type_="transform-find", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text=f"Tìm ma trận $\\mathbf{{A}} \\in \\R^{{2\\times 2}}$ của phép biến đổi: {desc}."),
            answer=mat_ans(a),
            hints=["Tìm ảnh của $\\mathbf{e}_1 = (1, 0)\\T$ và $\\mathbf{e}_2 = (0, 1)\\T$ rồi xếp thành các cột.",
                   "Vẽ nhanh hai vector này và xem chúng đi đâu sau phép biến đổi."],
            steps=[Step(text="Ảnh của hai vector cơ sở:", latex=f"T(\\mathbf{{e}}_1) = {L.mat(a * e1)},\\quad T(\\mathbf{{e}}_2) = {L.mat(a * e2)}"),
                   Step(text="Xếp thành các cột:", latex=L.mat(a, "\\mathbf{A}"))],
            verified=["sympy"], tags=["2x2", "described"],
        )
    # level 4: biết T(u), T(v) với u, v độc lập
    for _ in range(300):
        a = nontrivial_matrix(rng, 2, 2, -3, 3)
        u, v = rand_vector(rng, 2, -3, 3), rand_vector(rng, 2, -3, 3)
        if Matrix.hstack(u, v).det() != 0:
            break
    uv = Matrix.hstack(u, v)
    tuv = a * uv
    got = tuv * uv.inv()
    check(got == a, "A = [Tu Tv][u v]^-1")
    return build(
        type_="transform-find", lesson=LESSON, level=4, seed=seed,
        prompt=Prompt(text="$T(\\mathbf{x}) = \\mathbf{A}\\mathbf{x}$ là phép biến đổi tuyến tính trên $\\R^{2}$. Biết ảnh của hai vector độc lập $\\mathbf{u}, \\mathbf{v}$. "
                           "Tìm $\\mathbf{A}$.",
                      display=[L.mat(u, B("u")), L.mat(v, B("v")), L.mat(a * u, "T(\\mathbf{u})"), L.mat(a * v, "T(\\mathbf{v})")]),
        answer=mat_ans(a),
        hints=["Gộp thành $\\mathbf{A}\\,[\\mathbf{u}\\;\\mathbf{v}] = [T(\\mathbf{u})\\;T(\\mathbf{v})]$.", "Nhân phải với nghịch đảo của $[\\mathbf{u}\\;\\mathbf{v}]$."],
        steps=[Step(text="Ma trận $[\\mathbf{u}\\;\\mathbf{v}]$ và ảnh của nó:", latex=f"{L.mat(uv)},\\quad {L.mat(tuv)}"),
               Step(text="Nghịch đảo của $[\\mathbf{u}\\;\\mathbf{v}]$:", latex=L.mat(uv.inv())),
               Step(text="$\\mathbf{A} = [T(\\mathbf{u})\\;T(\\mathbf{v})]\\,[\\mathbf{u}\\;\\mathbf{v}]^{-1}$:", latex=L.mat(a, "\\mathbf{A}"))],
        verified=["sympy", "substitution"], tags=["2x2", "solve"],
    )


# ---------------------------------------------------------- transform-compose
def gen_compose(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 2:
        a, b = nontrivial_matrix(rng, 2, 2, -3, 3), nontrivial_matrix(rng, 2, 2, -3, 3)
        c = b * a
        verify_matmul(b, a, c)
        return build(
            type_="transform-compose", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text="Áp dụng $T_1(\\mathbf{x}) = \\mathbf{A}\\mathbf{x}$ **trước**, rồi $T_2(\\mathbf{y}) = \\mathbf{B}\\mathbf{y}$. "
                               "Tìm ma trận $\\mathbf{C}$ của phép hợp $T_2 \\circ T_1$.", display=[L.mat(a, B("A")), L.mat(b, B("B"))]),
            answer=mat_ans(c),
            hints=["$T_2(T_1(\\mathbf{x})) = \\mathbf{B}(\\mathbf{A}\\mathbf{x}) = (\\mathbf{B}\\mathbf{A})\\mathbf{x}$: phép áp dụng trước đứng bên **phải**."],
            steps=[Step(text="$\\mathbf{C} = \\mathbf{B}\\mathbf{A}$ (thứ tự đảo so với thứ tự áp dụng):")] + matmul_steps(b, a, "\\mathbf{B}", "\\mathbf{A}")[1:],
            verified=["sympy", "numpy"], tags=["2x2"],
        )
    if level == 3:
        d1, m1 = _named(rng)
        d2, m2 = _named(rng)
        c = m2 * m1
        verify_matmul(m2, m1, c)
        d1, d2 = d1, d2
        return build(
            type_="transform-compose", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text=f"Áp dụng lần lượt hai phép biến đổi trên $\\R^{{2}}$: **trước** là {d1}, **sau** là {d2}. "
                               f"Tìm ma trận của phép hợp (thực hiện cả hai)."),
            answer=mat_ans(c),
            hints=["Viết ma trận của từng phép (cột là ảnh của $\\mathbf{e}_1, \\mathbf{e}_2$).", "Phép làm sau đứng bên trái trong tích."],
            steps=[Step(text="Ma trận từng phép:", latex=f"\\mathbf{{M}}_1 = {L.mat(m1)},\\quad \\mathbf{{M}}_2 = {L.mat(m2)}"),
                   Step(text="Ma trận của phép hợp là $\\mathbf{M}_2\\mathbf{M}_1$:", latex=L.mat(c, "\\mathbf{M}_2\\mathbf{M}_1"))],
            verified=["sympy", "numpy"], tags=["2x2", "described"],
        )
    if level == 4:
        a, b, c3 = (nontrivial_matrix(rng, 2, 2, -2, 2) for _ in range(3))
        res = c3 * b * a
        np_close(res, to_np(c3) @ to_np(b) @ to_np(a))
        return build(
            type_="transform-compose", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Áp dụng lần lượt $\\mathbf{A}$, rồi $\\mathbf{B}$, rồi $\\mathbf{C}$ (như các phép biến đổi tuyến tính). Tìm ma trận của cả ba phép hợp lại.",
                          display=[L.mat(a, B("A")), L.mat(b, B("B")), L.mat(c3, B("C"))]),
            answer=mat_ans(res),
            hints=["Phép áp dụng đầu tiên đứng cuối cùng bên phải: $\\mathbf{CBA}$.", "Nhân từng cặp một: dùng tính kết hợp."],
            steps=[Step(text="Tính $\\mathbf{BA}$ trước:", latex=L.mat(b * a, "\\mathbf{BA}")),
                   Step(text="Rồi $\\mathbf{C}(\\mathbf{BA})$:", latex=L.mat(res, "\\mathbf{CBA}"))],
            verified=["sympy", "numpy"], tags=["2x2", "three"],
        )
    # level 5: hai lớp tuyến tính liên tiếp
    h = rng.randint(2, 3)
    w1 = nontrivial_matrix(rng, h, 2, -2, 2)
    w2 = nontrivial_matrix(rng, 2, h, -2, 2)
    res = w2 * w1
    verify_matmul(w2, w1, res)
    return build(
        type_="transform-compose", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text="Hai lớp tuyến tính liên tiếp (không có hàm kích hoạt) biến đầu vào $\\mathbf{x}$ thành $\\mathbf{W}_2(\\mathbf{W}_1\\mathbf{x})$. "
                           "Hai lớp đó tương đương **một** lớp tuyến tính duy nhất $\\mathbf{W}\\mathbf{x}$. Tìm $\\mathbf{W}$.",
                      display=[L.mat(w1, "\\mathbf{W}_1"), L.mat(w2, "\\mathbf{W}_2")]),
        answer=mat_ans(res),
        hints=["$\\mathbf{W}_2(\\mathbf{W}_1\\mathbf{x}) = (\\mathbf{W}_2\\mathbf{W}_1)\\mathbf{x}$.",
               "Đây là lý do mạng neural cần hàm kích hoạt phi tuyến giữa các lớp."],
        steps=[Step(text="$\\mathbf{W} = \\mathbf{W}_2\\mathbf{W}_1$:")] + matmul_steps(w2, w1, "\\mathbf{W}_2", "\\mathbf{W}_1")[1:],
        verified=["sympy", "numpy"], tags=["ml", "2x2"],
    )


SPECS = [
    TopicSpec("transform-apply", LESSON, (1, 2, 3, 4), gen_apply),
    TopicSpec("transform-find", LESSON, (2, 3, 4), gen_find),
    TopicSpec("transform-compose", LESSON, (2, 3, 4, 5), gen_compose),
]
