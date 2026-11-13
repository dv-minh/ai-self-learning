"""Bài 2.1 (ndarray, shape, dtype) và 2.2 (chỉ số, cắt lát)."""
from __future__ import annotations

from ._np import (Answer, Step, TopicSpec, array_answer, check, choice_answer, code_prompt, make_choices, nb, np, number_answer, random, rat,
                  run, tup)

# =========================================================================== 2.1
L21 = "2.1"

# (biểu thức trên x, mô tả hành động) áp dụng lên mảng x
OPS_2D = [
    ("x.T", "chuyển vị"), ("x[0]", "lấy hàng đầu"), ("x[:, 0]", "lấy cột đầu"), ("x[:, None]", "thêm trục ở giữa"),
    ("x[None, :]", "thêm trục đầu"), ("x.flatten()", "trải phẳng"), ("x[:2]", "lấy 2 hàng đầu"), ("x.sum(axis=0)", "tổng theo cột"),
    ("x.sum(axis=1)", "tổng theo hàng"), ("x.sum(axis=1, keepdims=True)", "tổng theo hàng, giữ trục"), ("x[1:, :2]", "cắt lát"),
]
OPS_3D = [
    ("x.sum(axis=0)", "tổng theo trục 0"), ("x.sum(axis=1)", "tổng theo trục 1"), ("x.sum(axis=2)", "tổng theo trục 2"),
    ("x.transpose(2, 0, 1)", "hoán vị trục"), ("x[0]", "lấy lát đầu"), ("x[:, 0, :]", "cố định trục 1"), ("x.reshape(-1)", "trải phẳng"),
    ("x.reshape(x.shape[0], -1)", "gộp các trục sau"), ("x.mean(axis=(0, 1))", "trung bình theo hai trục"), ("x[..., 0]", "lấy kênh đầu"),
]
OPS_ML = [
    ("X.reshape(X.shape[0], -1)", "trải phẳng từng ảnh"), ("X.mean(axis=(1, 2))", "trung bình theo không gian"),
    ("X.transpose(0, 3, 1, 2)", "đưa kênh lên trước"), ("X.mean(axis=0)", "trung bình theo batch"), ("X[:, :, :, 0]", "lấy kênh đầu"),
    ("X.max(axis=3)", "max theo kênh"), ("X.reshape(-1, X.shape[3])", "mỗi điểm ảnh một hàng"),
]


def _shape_choices(rng, correct_shape, other_shapes):
    wrong = [tup(s) for s in other_shapes if s != correct_shape]
    wrong += [tup(tuple(reversed(correct_shape))), tup(correct_shape + (1,)), tup(correct_shape[:-1] + (correct_shape[-1] + 1,))]
    wrong = [w for w in wrong if w != tup(correct_shape)]
    return make_choices(rng, [tup(correct_shape)], wrong, 4)


def gen_np_shape(rng: random.Random, level: int, seed: int):
    if level == 1:
        nd = rng.choice([1, 2, 2])
        shape = (rng.randint(2, 5),) if nd == 1 else (rng.randint(2, 4), rng.randint(2, 5))
        a = np.arange(int(np.prod(shape))).reshape(shape) + rng.randint(0, 3)
        code = f"import numpy as np\n\nx = {np.array2string(a, separator=', ').replace(chr(10), '')}\nx = np.array(x)\nprint(x.shape)"
        code = f"import numpy as np\n\nx = np.array({a.tolist()})\nprint(x.shape)"
        res = a.shape
        others = [tuple(reversed(res)), (int(np.prod(res)),), res + (1,)]
        steps = [Step(text=f"Mảng có {a.ndim} chiều. `shape` là tuple độ dài của từng chiều; với mảng 1 chiều dài ${res[0]}$, tuple có một phần tử và viết thêm dấu phẩy: `{tup(res)}`.")]
        tags = [f"{a.ndim}d"]
    else:
        if level == 2:
            shape, ops = (rng.randint(2, 5), rng.randint(2, 5)), OPS_2D
            if shape[0] == shape[1]:
                shape = (shape[0], shape[1] + 1)
            name, line = "x", "x"
        elif level == 3:
            shape, ops = (rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 5)), OPS_3D
            name = "x"
        else:  # level 5
            shape, ops = (rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 4), rng.choice([1, 3, 3])), OPS_ML
            name = "X"
        expr, why = rng.choice(ops)
        x = np.zeros(shape, dtype=np.int64)
        res = run(expr, **{name: x, "x": x, "X": x}).shape
        decl = f"{name} = np.zeros({tup(shape)})"
        code = f"import numpy as np\n\n{decl}\nprint({expr}.shape)"
        others = [run(e, **{name: x, "x": x, "X": x}).shape for e, _ in ops]
        steps = [Step(text=f"`{name}` có shape `{tup(shape)}`.")]
        steps.append(Step(text=f"`{expr}` ({why}) cho shape `{tup(res)}`."))
        tags = [f"{len(shape)}d"] + (["ml"] if level == 5 else [])
    choices, idx = _shape_choices(rng, res, others)
    text = "Điều gì được in ra? Chọn đáp án đúng."
    return nb(type_="np-shape", lesson=L21, level=level, seed=seed, prompt=code_prompt(text, code), answer=choice_answer(idx), choices=choices,
              hints=["Nhớ `shape` là tuple; mảng một chiều có shape dạng `(n,)`.", "Đếm số chiều trước, rồi độ dài từng chiều."],
              steps=steps, verified=["numpy"], tags=tags)


# dtype: (biểu thức, giải thích)
DTYPES = [
    ("np.array([1, 2, 3]).dtype", "toàn số nguyên nên là kiểu nguyên mặc định (`int64` với NumPy 2)"),
    ("np.array([1, 2.5, 3]).dtype", "có một số thực nên cả mảng nâng lên số thực"),
    ("np.array([1, 2, 3]).astype(float).dtype", "`astype(float)` chuyển sang `float64`"),
    ("(np.array([1, 2, 3]) / 2).dtype", "phép chia `/` luôn cho số thực"),
    ("(np.array([1, 2, 3]) // 2).dtype", "phép chia nguyên `//` giữ kiểu nguyên"),
    ("(np.array([1, 2, 3]) * 2.0).dtype", "nhân với số thực cho số thực"),
    ("np.array([True, False, True]).dtype", "mảng giá trị đúng/sai là `bool`"),
    ("(np.array([1, 2, 3]) > 1).dtype", "phép so sánh cho mảng `bool`"),
    ("np.array([True, False, True]).sum().dtype", "cộng các `bool` cho số nguyên"),
    ("np.arange(5).dtype", "`arange` với đối số nguyên cho số nguyên"),
    ("np.arange(0, 1, 0.25).dtype", "`arange` với bước thực cho số thực"),
    ("np.zeros(3).dtype", "`zeros` mặc định là số thực"),
    ("np.ones(3, dtype=int).dtype", "đã chỉ định `dtype=int`"),
    ("np.array([1, 2, 3], dtype=np.float32).dtype", "đã chỉ định `float32`"),
    ("np.linspace(0, 1, 5).dtype", "`linspace` cho số thực"),
    ("np.eye(2).dtype", "`eye` mặc định là số thực"),
    ("np.array([[1, 2], [3, 4]]).mean(axis=0).dtype", "trung bình luôn là số thực"),
    ("np.array([1, 2, 3]).sum().dtype", "tổng các số nguyên giữ kiểu nguyên"),
    ("np.sqrt(np.array([1, 4, 9])).dtype", "`sqrt` trả số thực"),
    ("(np.array([1, 2]) + np.array([0.5, 0.5])).dtype", "cộng nguyên với thực cho thực"),
]
DT_POOL = ["int64", "float64", "bool", "float32", "int32", "complex128"]


def gen_np_dtype(rng: random.Random, level: int, seed: int):
    pool = {1: DTYPES[:10], 2: DTYPES[5:15], 3: DTYPES[8:]}[level]
    expr, why = rng.choice(pool)
    res = str(run(expr))
    check(res in DT_POOL, f"dtype lạ {res}")
    code = f"import numpy as np\n\nprint({expr})"
    choices, idx = make_choices(rng, [res], [d for d in DT_POOL if d != res], 4)
    return nb(type_="np-dtype", lesson=L21, level=level, seed=seed,
              prompt=code_prompt("Đoạn code in ra kiểu dữ liệu nào? (NumPy 2, Python 64 bit)", code),
              answer=choice_answer(idx), choices=choices,
              hints=["Số nguyên thuần cho `int64`; xuất hiện một số thực thì cả mảng thành `float64`; so sánh cho `bool`."],
              steps=[Step(text=f"`{res}`: {why}.")], verified=["numpy"], tags=["dtype"])


# =========================================================================== 2.2
L22 = "2.2"


def _mat(rng, r, c, lo=-9, hi=9):
    return np.array([[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)])


def gen_np_index(rng: random.Random, level: int, seed: int):
    if level == 5:  # đổi chỉ số toán (từ 1) sang NumPy (từ 0)
        r, c = rng.randint(2, 4), rng.randint(2, 4)
        i, j = rng.randint(1, r), rng.randint(1, c)
        correct = f"A[{i - 1}, {j - 1}]"
        wrong = [f"A[{i}, {j}]", f"A[{j - 1}, {i - 1}]", f"A[{i - 1}][{j}]" if j < c else f"A[{i}, {j - 1}]", f"A[{i - 1}, {j}]" if j < c else f"A[{i}, {j - 1}]"]
        choices, idx = make_choices(rng, [correct], wrong, 4)
        a = _mat(rng, r, c, 1, 9)
        v = run(f"A[{i - 1}, {j - 1}]", A=a)
        code = f"import numpy as np\n\nA = np.array({a.tolist()})   # ma trận A cỡ {r}x{c} trong toán"
        return nb(type_="np-index", lesson=L22, level=5, seed=seed,
                  prompt=code_prompt(f"Trong toán, $a_{{{i}{j}}}$ là phần tử hàng {i}, cột {j} (đếm từ $1$). Biểu thức NumPy nào lấy đúng phần tử này?", code),
                  answer=choice_answer(idx), choices=choices,
                  hints=["Toán đếm từ $1$, NumPy đếm từ $0$: trừ $1$ ở cả chỉ số hàng và cột.", "Chỉ số hàng đứng trước."],
                  steps=[Step(text=f"Hàng {i} trong toán là chỉ số `{i - 1}`, cột {j} là chỉ số `{j - 1}`: `{correct}` (giá trị {v}).")],
                  verified=["numpy"], tags=["ml", "index-base"])
    r, c = (rng.randint(2, 3), rng.randint(2, 4)) if level == 1 else (rng.randint(3, 4), rng.randint(3, 5))
    a = _mat(rng, r, c, -9 if level > 1 else 0, 9)
    if level == 1:
        i, j = rng.randrange(r), rng.randrange(c)
        expr = f"a[{i}, {j}]"
    elif level == 2:
        kind = rng.choice(["neg", "neg-col", "chain", "last"])
        i, j = rng.randrange(r), rng.randrange(c)
        expr = {"neg": f"a[-{rng.randint(1, r)}, {j}]", "neg-col": f"a[{i}, -{rng.randint(1, c)}]", "chain": f"a[{i}][{j}]", "last": "a[-1, -1]"}[kind]
    else:
        kind = rng.choice(["mask", "fancy", "rowsum", "colmax"])
        if kind == "mask":
            t = rng.randint(-3, 5)
            expr = f"a[a > {t}].sum()"
        elif kind == "fancy":
            ii = rng.sample(range(r), 2)
            jj = [rng.randrange(c), rng.randrange(c)]
            expr = f"a[{ii}, {jj}].sum()"
        elif kind == "rowsum":
            expr = f"a[{rng.randrange(r)}].sum()"
        else:
            expr = f"a[:, {rng.randrange(c)}].max()"
    v = run(expr, a=a)
    code = f"import numpy as np\n\na = np.array({a.tolist()})\nprint({expr})"
    return nb(type_="np-index", lesson=L22, level=level, seed=seed,
              prompt=code_prompt("Đoạn code in ra giá trị nào? Nhập một số nguyên.", code), answer=number_answer(v),
              hints=["Chỉ số bắt đầu từ $0$; chỉ số âm đếm từ cuối (`-1` là phần tử cuối).", "`a[i, j]` và `a[i][j]` cho cùng kết quả với mảng 2 chiều."],
              steps=[Step(text=f"Đánh giá `{expr}` trên ma trận đã cho: kết quả {rat(v)}.")], verified=["numpy"], tags=[f"{r}x{c}"])


def gen_np_slice(rng: random.Random, level: int, seed: int):
    if level == 2:  # 1 chiều
        n = rng.randint(5, 8)
        v = np.array([rng.randint(-9, 9) for _ in range(n)])
        s = rng.choice([f"{rng.randint(0, 2)}:{rng.randint(3, n)}", f"::{rng.randint(2, 3)}", "::-1", f"{rng.randint(1, 3)}:", f":-{rng.randint(1, 3)}", f"-{rng.randint(2, 4)}:"])
        expr = f"v[{s}]"
        res = run(expr, v=v)
        code = f"import numpy as np\n\nv = np.array({v.tolist()})\nprint({expr})"
        tags = ["1d"]
    else:
        r, c = rng.randint(3, 4), rng.randint(3, 4)
        a = _mat(rng, r, c, -9, 9)
        if level == 3:
            expr = rng.choice([f"a[{rng.randint(0, 1)}:{rng.randint(2, r)}, {rng.randint(0, 1)}:{rng.randint(2, c)}]", f"a[:, {rng.randrange(c)}]", f"a[{rng.randrange(r)}, :]", f"a[1:, :2]", f"a[:2, 1:]"])
        else:
            expr = rng.choice(["a[::2, ::2]", "a[1:, ::-1]", "a[:-1, 1:]", "a[::-1, :2]", f"a[1:, :-1][::2]", f"a[:, ::2]", "a[-2:, -2:]", "a[::-1, ::-1]"])
        res = run(expr, a=a)
        code = f"import numpy as np\n\na = np.array({a.tolist()})\nprint({expr})"
        tags = [f"{r}x{c}"]
    check(np.asarray(res).size > 0, "lát cắt rỗng")
    res = np.atleast_1d(res)
    shape_txt = "vector" if res.ndim == 1 else "ma trận"
    return nb(type_="np-slice", lesson=L22, level=level, seed=seed,
              prompt=code_prompt(f"Điền kết quả in ra ({shape_txt} {res.shape[0]}" + (f" × {res.shape[1]}" if res.ndim == 2 else " phần tử") + ").", code),
              answer=array_answer(res),
              hints=["Cú pháp `start:stop:step`: lấy từ `start` đến **trước** `stop`.", "Với mảng 2 chiều, dấu phẩy ngăn lát cắt của hàng và của cột."],
              steps=[Step(text=f"`{expr}` cho mảng có shape `{tup(res.shape)}`:", latex=None)], verified=["numpy"], tags=tags)


SPECS = [
    TopicSpec("np-shape", L21, (1, 2, 3, 5), gen_np_shape),
    TopicSpec("np-dtype", L21, (1, 2, 3), gen_np_dtype),
    TopicSpec("np-index", L22, (1, 2, 3, 5), gen_np_index),
    TopicSpec("np-slice", L22, (2, 3, 4), gen_np_slice),
]
