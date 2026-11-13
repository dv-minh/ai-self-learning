"""Bài 2.3 (@, * và .T) và 2.4 (rút gọn theo trục)."""
from __future__ import annotations

from sympy import Matrix

from ._np import (Answer, Step, TopicSpec, array_answer, check, choice_answer, code_prompt, make_choices, nb, np, number_answer, random, rat,
                  run, tup)

L23, L24 = "2.3", "2.4"


def _mat(rng, r, c, lo=-4, hi=4):
    return np.array([[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)])


def _runs(expr: str, **ns) -> bool:
    try:
        run(expr, **ns)
        return True
    except ValueError:
        return False


def gen_matmul_vs_mul(rng: random.Random, level: int, seed: int):
    if level in (2, 3):
        r, n, c = (2, 2, 2) if level == 2 else rng.choice([(2, 3, 2), (3, 2, 3), (3, 3, 3)])
        a = _mat(rng, r, n)
        if level == 2:
            b = _mat(rng, n, c)
            op, name = rng.choice([("A @ B", "tích ma trận"), ("A * B", "nhân từng phần tử"), ("np.dot(A, B)", "tích ma trận")])
        else:
            op, name = rng.choice([("A @ B", "tích ma trận"), ("A * B", "nhân từng phần tử"), ("A @ B.T", "tích ma trận với B chuyển vị"), ("B.T @ A.T", "tích các ma trận chuyển vị")])
            b = _mat(rng, r, n) if op in ("A * B",) else (_mat(rng, c, n) if op == "A @ B.T" else _mat(rng, n, c))
            if op == "A * B":
                b = _mat(rng, r, n)
        res = run(op, A=a, B=b)
        # đối chiếu độc lập bằng SymPy
        sa, sb = Matrix(a.tolist()), Matrix(b.tolist())
        ref = {"A @ B": lambda: sa * sb, "np.dot(A, B)": lambda: sa * sb, "A * B": lambda: sa.multiply_elementwise(sb),
               "A @ B.T": lambda: sa * sb.T, "B.T @ A.T": lambda: sb.T * sa.T}[op]()
        check(Matrix(res.tolist()) == ref, "NumPy ≠ SymPy")
        code = f"import numpy as np\n\nA = np.array({a.tolist()})\nB = np.array({b.tolist()})\nprint({op})"
        return nb(type_="np-matmul-vs-mul", lesson=L23, level=level, seed=seed,
                  prompt=code_prompt(f"Điền kết quả in ra ({res.shape[0]} × {res.shape[1]}).", code), answer=array_answer(res),
                  hints=["`@` là tích ma trận (hàng nhân cột); `*` nhân từng phần tử cùng vị trí.", "`.T` chuyển vị."],
                  steps=[Step(text=f"`{op}` là {name}; kết quả có shape `{tup(res.shape)}`.")], verified=["numpy", "sympy"], tags=["value"])
    # level 4: chọn mọi biểu thức chạy được
    cands = ["A @ B", "A * B", "A @ B.T", "A.T @ B", "A + B"]
    while True:  # cần ít nhất một biểu thức chạy được và một biểu thức lỗi
        sh_a, sh_b = (rng.randint(1, 4), rng.randint(1, 4)), (rng.randint(1, 4), rng.randint(1, 4))
        a, b = np.zeros(sh_a), np.zeros(sh_b)
        ok = [e for e in cands if _runs(e, A=a, B=b)]
        if ok and len(ok) < len(cands):
            break
    choices = [f"`{e}`" for e in cands]
    idx = [i for i, e in enumerate(cands) if e in ok]
    code = f"import numpy as np\n\nA = np.zeros({tup(sh_a)})\nB = np.zeros({tup(sh_b)})"
    return nb(type_="np-matmul-vs-mul", lesson=L23, level=4, seed=seed,
              prompt=code_prompt("Chọn **mọi** biểu thức chạy được (không báo `ValueError`). Lưu ý `*` và `+` có thể broadcast.", code),
              answer=choice_answer(idx), choices=choices,
              hints=["`@` cần số cột của trái bằng số hàng của phải.", "`*` và `+` cần hai shape cùng nhau hoặc broadcast được (bài 2.5)."],
              steps=[Step(text=f"`{e}`: " + ("chạy được." if e in ok else "báo `ValueError` (shape không phù hợp).")) for e in cands],
              verified=["numpy"], tags=["error-check"])


def gen_transpose(rng: random.Random, level: int, seed: int):
    if level in (2, 3):
        r, c = (2, 3) if level == 2 else rng.choice([(3, 2), (4, 2), (2, 4), (3, 4)])
        a = _mat(rng, r, c)
        expr = rng.choice(["A.T @ A", "A @ A.T"]) if level == 3 else "A.T"
        res = run(expr, A=a)
        ref = {"A.T": Matrix(a.tolist()).T, "A.T @ A": Matrix(a.tolist()).T * Matrix(a.tolist()), "A @ A.T": Matrix(a.tolist()) * Matrix(a.tolist()).T}[expr]
        check(Matrix(res.tolist()) == ref, "NumPy ≠ SymPy")
        code = f"import numpy as np\n\nA = np.array({a.tolist()})\nprint({expr})"
        return nb(type_="np-transpose", lesson=L23, level=level, seed=seed,
                  prompt=code_prompt(f"Điền kết quả in ra ({res.shape[0]} × {res.shape[1]}).", code), answer=array_answer(res),
                  hints=["`A.T` đổi hàng thành cột.", "`A.T @ A` luôn là ma trận vuông đối xứng (ma trận Gram)."],
                  steps=[Step(text=f"`{expr}` có shape `{tup(res.shape)}`.")], verified=["numpy", "sympy"], tags=["value"])
    if level == 4:  # vector 1 chiều: .T không đổi gì
        n = rng.randint(3, 5)
        v = np.arange(1, n + 1)
        expr = rng.choice(["v.T.shape", "(v @ v)", "np.outer(v, v).shape", "v[:, None].T.shape", "(v[:, None] @ v[None, :]).shape", "(v.T @ v)"])
        res = run(expr, v=v)
        code = f"import numpy as np\n\nv = np.arange(1, {n + 1})   # shape ({n},)\nprint({expr})"
        if isinstance(res, tuple):
            cands = [res, tuple(reversed(res)), res + (1,), (1,) + res, (n * n,)]
            choices, idx = make_choices(rng, [tup(res)], [tup(c) for c in cands if c != res], 4)
            ans = choice_answer(idx)
            kind = "shape"
        else:
            choices, ans, kind = None, number_answer(res), "value"
        return nb(type_="np-transpose", lesson=L23, level=4, seed=seed,
                  prompt=code_prompt("Điều gì được in ra?" + (" Chọn shape." if choices else " Nhập một số nguyên."), code),
                  answer=ans, choices=choices,
                  hints=["Vector 1 chiều (`shape (n,)`) **không có** hàng/cột: `.T` không làm gì với nó.", "Muốn vector cột hay hàng cần thêm trục: `v[:, None]` hoặc `v[None, :]`."],
                  steps=[Step(text=f"`{expr}` = `{res}`.")], verified=["numpy"], tags=[kind])
    # level 5: shape trong ML
    n, d, h = rng.randint(2, 6), rng.randint(2, 5), rng.randint(2, 5)
    X, W = np.zeros((n, d)), np.zeros((d, h))
    expr = rng.choice(["X.T @ X", "X @ W", "(X @ W).T", "W.T @ X.T", "X.T @ (X @ W)"])
    res = run(expr, X=X, W=W).shape
    cands = [res, tuple(reversed(res)), (n, n), (d, d), (n, h), (h, d)]
    choices, idx = make_choices(rng, [tup(res)], [tup(c) for c in cands if c != res], 4)
    code = f"import numpy as np\n\nX = np.zeros(({n}, {d}))   # {n} mẫu, {d} đặc trưng\nW = np.zeros(({d}, {h}))\nprint(({expr}).shape)"
    return nb(type_="np-transpose", lesson=L23, level=5, seed=seed,
              prompt=code_prompt("`shape` nào được in ra?", code), answer=choice_answer(idx), choices=choices,
              hints=["Viết shape các thừa số cạnh nhau và kiểm tra hai số ở giữa khớp nhau."], steps=[Step(text=f"`{expr}` có shape `{tup(res)}`.")],
              verified=["numpy", "shape"], tags=["ml"])


# ---------------------------------------------------------------- 2.4 axis
AXIS_FUNCS = ["sum", "max", "min", "prod"]


def gen_axis(rng: random.Random, level: int, seed: int):
    if level in (2, 3):
        r, c = (rng.randint(2, 3), rng.randint(2, 4)) if level == 2 else (rng.randint(3, 4), rng.randint(3, 4))
        a = _mat(rng, r, c, 1 if level == 2 else -5, 6 if level == 2 else 7)
        fn = rng.choice(["sum", "max", "min"] if level == 2 else ["sum", "max", "min", "argmax"])
        ax = rng.randint(0, 1)
        kd = level == 3 and rng.random() < 0.4
        expr = f"a.{fn}(axis={ax}" + (", keepdims=True" if kd else "") + ")"
        res = run(expr, a=a)
        ref = {"sum": np.add.reduce, "max": np.maximum.reduce, "min": np.minimum.reduce}.get(fn)
        if ref is not None:
            check(np.array_equal(res.reshape(-1), ref(a, axis=ax)), "reduce độc lập")
        code = f"import numpy as np\n\na = np.array({a.tolist()})\nprint({expr})"
        return nb(type_="np-axis", lesson=L24, level=level, seed=seed,
                  prompt=code_prompt(f"Điền kết quả in ra ({res.shape[0]}" + (f" × {res.shape[1]}" if res.ndim == 2 else " phần tử") + ").", code),
                  answer=array_answer(res),
                  hints=["`axis=0` rút gọn **theo hàng đi xuống** (kết quả có một giá trị cho mỗi cột); `axis=1` rút gọn dọc theo mỗi hàng.", "`keepdims=True` giữ trục đã rút gọn với độ dài $1$."],
                  steps=[Step(text=f"`{expr}` cho shape `{tup(res.shape)}`.")], verified=["numpy"], tags=[f"axis{ax}", fn])
    if level == 4:  # trung bình, cần giá trị nguyên/nửa
        for _ in range(100):
            r, c = rng.randint(2, 3), rng.randint(2, 4)
            a = _mat(rng, r, c, 0, 9)
            ax = rng.randint(0, 1)
            m = a.mean(axis=ax)
            if all(abs(x * 2 - round(x * 2)) < 1e-9 for x in m.reshape(-1)):
                break
        expr = f"a.mean(axis={ax})"
        res = run(expr, a=a)
        code = f"import numpy as np\n\na = np.array({a.tolist()})\nprint({expr})"
        return nb(type_="np-axis", lesson=L24, level=4, seed=seed,
                  prompt=code_prompt("Điền kết quả in ra (số thực; nhập số nguyên hoặc dạng $p/q$ hoặc thập phân).", code),
                  answer=array_answer(res), hints=["`mean` = tổng chia số phần tử dọc theo trục."], steps=[Step(text=f"`{expr}` = `{res.tolist()}`.")],
                  verified=["numpy"], tags=["mean"])
    # level 5: shape trong ML
    n, d = rng.randint(2, 6), rng.randint(2, 5)
    z = np.zeros((n, d))
    expr, why = rng.choice([("z.mean(axis=0)", "trung bình theo batch cho mỗi đặc trưng"), ("z.sum(axis=1)", "tổng theo từng mẫu"),
                            ("np.exp(z).sum(axis=1, keepdims=True)", "mẫu số của softmax theo hàng"), ("z.max(axis=1, keepdims=True)", "giá trị lớn nhất của từng mẫu"),
                            ("z.std(axis=0, keepdims=True)", "độ lệch chuẩn từng đặc trưng")])
    res = run(expr, z=z).shape
    cands = [res, (n,), (d,), (n, 1), (1, d), (n, d)]
    choices, idx = make_choices(rng, [tup(res)], [tup(c) for c in cands if c != res], 4)
    code = f"import numpy as np\n\nz = np.zeros(({n}, {d}))   # {n} mẫu, {d} lớp/đặc trưng\nprint(({expr}).shape)"
    return nb(type_="np-axis", lesson=L24, level=5, seed=seed,
              prompt=code_prompt("`shape` nào được in ra?", code), answer=choice_answer(idx), choices=choices,
              hints=["Trục bị rút gọn biến mất, trừ khi dùng `keepdims=True`."], steps=[Step(text=f"`{expr}` ({why}): `{tup(res)}`.")],
              verified=["numpy", "shape"], tags=["ml"])


SPECS = [
    TopicSpec("np-matmul-vs-mul", L23, (2, 3, 4), gen_matmul_vs_mul),
    TopicSpec("np-transpose", L23, (2, 3, 4, 5), gen_transpose),
    TopicSpec("np-axis", L24, (2, 3, 4, 5), gen_axis),
]
