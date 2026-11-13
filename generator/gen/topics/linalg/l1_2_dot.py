"""Bài 1.2 Tích vô hướng, góc, cosine similarity: vec-dot, vec-cosine, vec-orthogonal, vec-project."""
from __future__ import annotations

import random

import numpy as np
from sympy import Matrix, Rational, sqrt

from ...core import latex as L
from ...core.nice import int_norm_vector, level_range, rand_vector
from ...core.steps import componentwise
from ...core.verify import check, np_close, to_np
from ...schema import Answer, Exercise, Prompt, Step
from ._base import TopicSpec, bool_ans, build, num_ans, shuffled_choices, vec_ans

LESSON = "1.2"
B = L.bold
N = L.num_paren


def _disp(**vs: Matrix) -> list[str]:
    return [L.mat(v, B(k)) for k, v in vs.items()]


def _dot_expr(u: Matrix, v: Matrix) -> str:
    return " + ".join(f"{N(a)} \\cdot {N(b)}" for a, b in zip(u, v))


def _dot(u: Matrix, v: Matrix):
    d = (u.T * v)[0, 0]
    check(np.isclose(float(d), float(np.dot(to_np(u).ravel(), to_np(v).ravel()))), "dot numpy")
    return d


# ------------------------------------------------------------------ vec-dot
def gen_dot(rng: random.Random, level: int, seed: int) -> Exercise:
    n = {1: 2, 2: 3, 3: rng.choice([3, 4]), 4: rng.choice([2, 3])}[level]
    lo, hi = level_range(level)
    fp = 0.2 if level == 3 else 0.0
    u, v = rand_vector(rng, n, lo, hi, fp), rand_vector(rng, n, lo, hi, fp)
    if level == 4:  # (u+v)·(u−v)
        d = _dot(u + v, u - v)
        check(d == _dot(u, u) - _dot(v, v), "dot identity")
        return build(
            type_="vec-dot", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Tính $(\\mathbf{u}+\\mathbf{v})\\T(\\mathbf{u}-\\mathbf{v})$.", display=_disp(u=u, v=v)),
            answer=num_ans(d),
            hints=["Tính $\\mathbf{u}+\\mathbf{v}$ và $\\mathbf{u}-\\mathbf{v}$ trước.",
                   "Cũng có thể dùng $\\mathbf{u}\\T\\mathbf{u} - \\mathbf{v}\\T\\mathbf{v}$."],
            steps=[Step(text="Tính tổng và hiệu:", latex=f"\\mathbf{{u}}+\\mathbf{{v}} = {L.mat(u + v)},\\quad \\mathbf{{u}}-\\mathbf{{v}} = {L.mat(u - v)}"),
                   Step(text="Tích vô hướng của hai vector này:", latex=f"{_dot_expr(u + v, u - v)} = {L.num(d)}")],
            verified=["sympy", "numpy"], tags=["identity", f"R{n}"],
        )
    d = _dot(u, v)
    return build(
        type_="vec-dot", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tính tích vô hướng $\\mathbf{u}\\T\\mathbf{v}$.", display=_disp(u=u, v=v)),
        answer=num_ans(d),
        hints=["Nhân các thành phần cùng vị trí rồi cộng lại."],
        steps=[Step(text="Nhân từng cặp thành phần rồi cộng:", latex=f"\\mathbf{{u}}\\T\\mathbf{{v}} = {_dot_expr(u, v)} = {L.num(d)}")],
        verified=["sympy", "numpy"], tags=[f"R{n}"],
    )


# -------------------------------------------------------------- vec-cosine
def _cos(u: Matrix, v: Matrix) -> Rational:
    nu, nv = sqrt(sum(x * x for x in u)), sqrt(sum(x * x for x in v))
    c = _dot(u, v) / (nu * nv)
    ref = np.dot(to_np(u).ravel(), to_np(v).ravel()) / (np.linalg.norm(to_np(u)) * np.linalg.norm(to_np(v)))
    check(np.isclose(float(c), ref), "cosine numpy")
    check(c.is_Rational, "cosine không hữu tỉ")
    return c


def gen_cosine(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 5:
        return _cosine_retrieval(rng, seed)
    n = {2: 2, 3: rng.choice([3, 4]), 4: rng.choice([2, 3])}[level]
    u = int_norm_vector(rng, n, hi=12 if n < 4 else 8)
    if level == 4:
        k = Rational(rng.choice([-3, -2, 2, 3, 4]))
        v = k * u
        why = "Hai vector cùng phương: cosine chỉ phụ thuộc chiều (cùng chiều là $1$, ngược chiều là $-1$)."
    else:
        v = int_norm_vector(rng, n, hi=12 if n < 4 else 8)
        why = None
    c = _cos(u, v)
    nu, nv = sqrt(sum(x * x for x in u)), sqrt(sum(x * x for x in v))
    steps = [
        Step(text="Tích vô hướng:", latex=f"\\mathbf{{u}}\\T\\mathbf{{v}} = {_dot_expr(u, v)} = {L.num(_dot(u, v))}"),
        Step(text="Hai chuẩn L2 (đều là số nguyên):", latex=f"\\norm{{\\mathbf{{u}}}}_2 = {nu},\\quad \\norm{{\\mathbf{{v}}}}_2 = {nv}"),
        Step(text="Chia tích vô hướng cho tích hai chuẩn:",
             latex=f"\\cos\\theta = \\frac{{\\mathbf{{u}}\\T\\mathbf{{v}}}}{{\\norm{{\\mathbf{{u}}}}_2\\norm{{\\mathbf{{v}}}}_2}} = {L.num(c)}"),
    ]
    if why:
        steps.append(Step(text=why))
    return build(
        type_="vec-cosine", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tính cosine similarity $\\cos\\theta = \\dfrac{\\mathbf{u}\\T\\mathbf{v}}{\\norm{\\mathbf{u}}_2\\norm{\\mathbf{v}}_2}$.",
                      display=_disp(u=u, v=v)),
        answer=num_ans(c),
        hints=["Tính tích vô hướng, rồi hai chuẩn L2.", "Các chuẩn đều là số nguyên nên kết quả là phân số."],
        steps=steps, verified=["sympy", "numpy"], tags=[f"R{n}"],
    )


def _cosine_retrieval(rng: random.Random, seed: int) -> Exercise:
    """Truy vấn q và 3 tài liệu: chọn tài liệu giống q nhất theo cosine similarity."""
    n = 3
    for _ in range(500):
        q = int_norm_vector(rng, n, hi=12)
        docs = [int_norm_vector(rng, n, hi=12) for _ in range(3)]
        cs = [_cos(q, d) for d in docs]
        if len(set(cs)) == 3:
            break
    else:
        raise RuntimeError("retrieval")
    best = max(range(3), key=lambda i: cs[i])
    choices, correct = shuffled_choices(rng, [f"$\\mathbf{{d}}_{best + 1}$"],
                                        [f"$\\mathbf{{d}}_{i + 1}$" for i in range(3) if i != best])
    # lưu thứ tự d1,d2,d3 ổn định: không xáo
    choices = [f"$\\mathbf{{d}}_{i + 1}$" for i in range(3)]
    steps = [Step(text=f"Cosine với $\\mathbf{{d}}_{i + 1}$:", latex=f"\\cos\\theta_{i + 1} = {L.num(cs[i])}") for i in range(3)]
    steps.append(Step(text="Cosine lớn nhất nghĩa là hướng gần $\\mathbf{q}$ nhất."))
    return build(
        type_="vec-cosine", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text="Trong một hệ truy xuất, vector truy vấn $\\mathbf{q}$ được so với 3 vector tài liệu theo "
                           "cosine similarity. Tài liệu nào giống $\\mathbf{q}$ nhất?",
                      display=_disp(q=q, d_1=docs[0], d_2=docs[1], d_3=docs[2])),
        answer=Answer(kind="choice", correct=[best]), choices=choices,
        hints=["Tính cosine của $\\mathbf{q}$ với từng $\\mathbf{d}_i$, chuẩn đều nguyên."],
        steps=steps, verified=["sympy", "numpy"], tags=["ml", "retrieval"],
    )


# ----------------------------------------------------------- vec-orthogonal
def gen_orthogonal(rng: random.Random, level: int, seed: int) -> Exercise:
    if level in (1, 2):
        n = 2 if level == 1 else 3
        lo, hi = level_range(level)
        for _ in range(500):
            u = rand_vector(rng, n, lo, hi)
            if rng.random() < 0.5:  # dựng cặp trực giao
                v = rand_vector(rng, n, lo, hi)
                # chiếu v lên mặt phẳng vuông góc u để ra số nguyên: dùng v - (u·v/u·u)u nếu nguyên
                p = v - (_dot(u, v) / _dot(u, u)) * u
                if any(x.q != 1 for x in p) or all(x == 0 for x in p) or any(abs(x) > hi for x in p):
                    continue
                v = p
            else:
                v = rand_vector(rng, n, lo, hi)
            break
        d = _dot(u, v)
        return build(
            type_="vec-orthogonal", lesson=LESSON, level=level, seed=seed,
            prompt=Prompt(text="Khẳng định \"$\\mathbf{u}$ và $\\mathbf{v}$ trực giao (vuông góc)\" đúng hay sai?",
                          display=_disp(u=u, v=v)),
            answer=bool_ans(d == 0),
            hints=["Hai vector trực giao khi và chỉ khi tích vô hướng bằng $0$."],
            steps=[Step(text="Tính tích vô hướng:", latex=f"\\mathbf{{u}}\\T\\mathbf{{v}} = {_dot_expr(u, v)} = {L.num(d)}"),
                   Step(text="Bằng $0$ nên trực giao." if d == 0 else "Khác $0$ nên không trực giao.")],
            verified=["sympy", "numpy"], tags=["orth" if d == 0 else "not-orth", f"R{n}"],
        )
    if level == 3:  # tìm k
        n = rng.choice([2, 3])
        u = rand_vector(rng, n, -5, 5)
        slot = rng.randrange(n)
        while u[slot] == 0:
            u = rand_vector(rng, n, -5, 5)
            slot = rng.randrange(n)
        v0 = rand_vector(rng, n, -5, 5)
        known = sum(u[i] * v0[i] for i in range(n) if i != slot)
        k = -known / u[slot]
        v = v0.copy()
        v[slot] = k
        check(_dot(u, v) == 0, "orth k")
        cells = " \\\\ ".join("k" if i == slot else L.num(v0[i]) for i in range(n))
        vtex = f"\\mathbf{{v}} = \\begin{{bmatrix}} {cells} \\end{{bmatrix}}"
        eq = " + ".join(f"{N(u[i])} \\cdot " + ("k" if i == slot else N(v0[i])) for i in range(n))
        return build(
            type_="vec-orthogonal", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text="Tìm $k$ để $\\mathbf{u}$ và $\\mathbf{v}$ trực giao.", display=[L.mat(u, B("u")), vtex]),
            answer=num_ans(k),
            hints=["Điều kiện trực giao: $\\mathbf{u}\\T\\mathbf{v} = 0$.", "Phương trình này bậc nhất theo $k$."],
            steps=[Step(text="Lập phương trình tích vô hướng bằng $0$:", latex=f"{eq} = 0"),
                   Step(text="Giải ra $k$:", latex=f"k = {L.num(k)}")],
            verified=["sympy", "numpy", "substitution"], tags=["param", f"R{n}"],
        )
    # level 4: trong 3 vector, cặp nào trực giao
    for _ in range(1000):
        a = rand_vector(rng, 3, -4, 4)
        b = rand_vector(rng, 3, -4, 4)
        p = b - (_dot(a, b) / _dot(a, a)) * a
        if all(x.q == 1 for x in p) and any(x != 0 for x in p) and all(abs(x) <= 6 for x in p):
            b = p
        else:
            continue
        c = rand_vector(rng, 3, -4, 4)
        if _dot(a, c) != 0 and _dot(b, c) != 0:
            break
    else:
        raise RuntimeError("orth pair")
    vs = [a, b, c]
    perm = list(range(3))
    rng.shuffle(perm)
    vs = [vs[i] for i in perm]
    pairs = [(0, 1), (0, 2), (1, 2)]
    dots = {p: _dot(vs[p[0]], vs[p[1]]) for p in pairs}
    right = [i for i, p in enumerate(pairs) if dots[p] == 0]
    check(len(right) == 1, "đúng một cặp")
    choices = [f"$\\mathbf{{x}}_{p[0] + 1}$ và $\\mathbf{{x}}_{p[1] + 1}$" for p in pairs]
    steps = [Step(text=f"$\\mathbf{{x}}_{p[0] + 1}\\T\\mathbf{{x}}_{p[1] + 1}$:",
                  latex=f"{_dot_expr(vs[p[0]], vs[p[1]])} = {L.num(dots[p])}") for p in pairs]
    return build(
        type_="vec-orthogonal", lesson=LESSON, level=4, seed=seed,
        prompt=Prompt(text="Cặp vector nào dưới đây trực giao?", display=_disp(x_1=vs[0], x_2=vs[1], x_3=vs[2])),
        answer=Answer(kind="choice", correct=right), choices=choices,
        hints=["Tính tích vô hướng của từng cặp; cặp trực giao cho kết quả $0$."],
        steps=steps, verified=["sympy", "numpy"], tags=["pairs"],
    )


# -------------------------------------------------------------- vec-project
def gen_project(rng: random.Random, level: int, seed: int) -> Exercise:
    n = 2 if level == 2 else 3
    u, v = rand_vector(rng, n, -5, 5), rand_vector(rng, n, -5, 5)
    uv, vv = _dot(u, v), _dot(v, v)
    coeff = uv / vv
    proj = coeff * v
    np_close(proj, (np.dot(to_np(u).ravel(), to_np(v).ravel()) / np.dot(to_np(v).ravel(), to_np(v).ravel())) * to_np(v))
    resid = u - proj
    check(_dot(resid, v) == 0, "phần dư không vuông góc")
    steps = [
        Step(text="Tính $\\mathbf{u}\\T\\mathbf{v}$ và $\\mathbf{v}\\T\\mathbf{v}$:",
             latex=f"\\mathbf{{u}}\\T\\mathbf{{v}} = {L.num(uv)},\\quad \\mathbf{{v}}\\T\\mathbf{{v}} = {L.num(vv)}"),
        Step(text="Hệ số chiếu:", latex=f"\\frac{{\\mathbf{{u}}\\T\\mathbf{{v}}}}{{\\mathbf{{v}}\\T\\mathbf{{v}}}} = {L.num(coeff)}"),
        Step(text="Nhân hệ số với $\\mathbf{v}$:", latex=f"\\mathrm{{proj}}_{{\\mathbf{{v}}}}\\mathbf{{u}} = {L.mat(proj)}"),
    ]
    if level == 4:
        steps.append(Step(text="Phần dư là $\\mathbf{u}$ trừ hình chiếu:", latex=f"\\mathbf{{u}} - \\mathrm{{proj}}_{{\\mathbf{{v}}}}\\mathbf{{u}} = {L.mat(resid)}"))
        return build(
            type_="vec-project", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Tính thành phần của $\\mathbf{u}$ vuông góc với $\\mathbf{v}$: "
                               "$\\mathbf{u} - \\dfrac{\\mathbf{u}\\T\\mathbf{v}}{\\mathbf{v}\\T\\mathbf{v}}\\mathbf{v}$.",
                          display=_disp(u=u, v=v)),
            answer=vec_ans(resid),
            hints=["Tính hình chiếu của $\\mathbf{u}$ lên $\\mathbf{v}$ trước, rồi lấy $\\mathbf{u}$ trừ đi."],
            steps=steps, verified=["sympy", "numpy", "orthogonal"], tags=["residual", f"R{n}"],
        )
    return build(
        type_="vec-project", lesson=LESSON, level=level, seed=seed,
        prompt=Prompt(text="Tính hình chiếu của $\\mathbf{u}$ lên $\\mathbf{v}$: "
                           "$\\mathrm{proj}_{\\mathbf{v}}\\mathbf{u} = \\dfrac{\\mathbf{u}\\T\\mathbf{v}}{\\mathbf{v}\\T\\mathbf{v}}\\mathbf{v}$.",
                      display=_disp(u=u, v=v)),
        answer=vec_ans(proj),
        hints=["Tính $\\mathbf{u}\\T\\mathbf{v}$ và $\\mathbf{v}\\T\\mathbf{v}$.", "Hệ số là thương hai số đó; nhân với $\\mathbf{v}$."],
        steps=steps[:3], verified=["sympy", "numpy"], tags=[f"R{n}"],
    )


SPECS = [
    TopicSpec("vec-dot", LESSON, (1, 2, 3, 4), gen_dot),
    TopicSpec("vec-cosine", LESSON, (2, 3, 4, 5), gen_cosine),
    TopicSpec("vec-orthogonal", LESSON, (1, 2, 3, 4), gen_orthogonal),
    TopicSpec("vec-project", LESSON, (2, 3, 4), gen_project),
]
