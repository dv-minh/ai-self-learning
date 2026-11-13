"""Bài 2.5 (broadcasting) và 2.6 (reshape, stack, concatenate)."""
from __future__ import annotations

from ._np import (Answer, Step, TopicSpec, array_answer, check, choice_answer, code_prompt, make_choices, nb, np, number_answer, random, rat,
                  run, tup)

L25, L26 = "2.5", "2.6"
ERR = "ValueError: không broadcast được"


def _mat(rng, r, c, lo=-5, hi=9):
    return np.array([[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)])


def _bshape(sa, sb):
    try:
        return np.broadcast_shapes(sa, sb)
    except ValueError:
        return None


def gen_broadcast_shape(rng: random.Random, level: int, seed: int):
    for _ in range(500):
        if level == 2:
            n, m = rng.randint(2, 5), rng.randint(2, 5)
            sa, sb = rng.choice([((n, m), (m,)), ((n, m), (n,)), ((n, m), (1, m)), ((n, m), (n, 1)), ((n,), (n,)), ((n, 1), (1, m))])
        elif level == 3:
            n, m, k = rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)
            sa, sb = rng.choice([((n, 1), (m,)), ((n, m), (k,)), ((1, m), (n, 1)), ((n, m, 1), (m, k)), ((n,), (m, 1)), ((n, m), (n, m))])
        else:
            n, m, k = rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)
            sa, sb = rng.choice([((n, 1, k), (m, 1)), ((n, m, k), (m, 1)), ((n, m, k), (k,)), ((1, m, 1), (n, 1, k)), ((n, m, k), (n, 1)), ((n, 1, 1), (1, m, k))])
        res = _bshape(sa, sb)
        if (res is None) == (rng.random() < 0.45 or level == 2 and rng.random() < 0.3):
            break
    code = f"import numpy as np\n\na = np.zeros({tup(sa)})\nb = np.zeros({tup(sb)})\nprint((a + b).shape)"
    if res is None:
        # xác nhận bằng cách chạy thật
        try:
            np.zeros(sa) + np.zeros(sb)
            raise AssertionError("đáng lẽ phải lỗi")
        except ValueError:
            pass
        wrong = [tup(sa), tup(sb), tup(tuple(max(x, y) for x, y in zip(sa[::-1], sb[::-1]))[::-1])]
        correct = ERR
    else:
        check((np.zeros(sa) + np.zeros(sb)).shape == res, "broadcast thật")
        wrong = [ERR, tup(sa), tup(sb), tup(tuple(reversed(res))), tup(res + (1,))]
        correct = tup(res)
    wrong = [w for w in wrong if w != correct]
    choices, idx = make_choices(rng, [correct], wrong, 4)
    if ERR not in choices and res is not None and rng.random() < 0.6:
        pass
    steps = [Step(text="Quy tắc: căn phải các shape; mỗi cặp chiều phải bằng nhau hoặc một trong hai bằng $1$ (chiều $1$ được kéo giãn); shape ngắn hơn được thêm $1$ ở bên trái.")]
    steps.append(Step(text=("Có cặp chiều khác nhau và không bằng $1$: lỗi." if res is None else f"Kết quả: `{tup(res)}`.")))
    return nb(type_="np-broadcast-shape", lesson=L25, level=level, seed=seed,
              prompt=code_prompt("Điều gì được in ra? Chọn đáp án đúng (nếu không broadcast được, NumPy báo `ValueError`).", code),
              answer=choice_answer(idx), choices=choices,
              hints=["Viết hai shape thẳng hàng theo bên **phải**, bổ sung `1` bên trái cho shape ngắn hơn.", "Từng cặp chiều: bằng nhau, hoặc có một chiều bằng $1$."],
              steps=steps, verified=["numpy"], tags=["error" if res is None else "ok"])


def gen_broadcast_value(rng: random.Random, level: int, seed: int):
    if level == 3:
        r, c = rng.randint(2, 3), rng.randint(2, 4)
        a = _mat(rng, r, c)
        which = rng.choice(["row", "col", "scalar"])
        if which == "row":
            v = np.array([rng.randint(-3, 5) for _ in range(c)])
            expr, vsrc = "a + v", f"v = np.array({v.tolist()})"
        elif which == "col":
            v = np.array([[rng.randint(-3, 5)] for _ in range(r)])
            expr, vsrc = "a * v", f"v = np.array({v.tolist()})"
        else:
            v = rng.randint(2, 5)
            expr, vsrc = "a - v", f"v = {v}"
        res = run(expr, a=a, v=v)
        code = f"import numpy as np\n\na = np.array({a.tolist()})\n{vsrc}\nprint({expr})"
        return nb(type_="np-broadcast-value", lesson=L25, level=3, seed=seed,
                  prompt=code_prompt(f"Điền kết quả in ra ({res.shape[0]} × {res.shape[1]}).", code), answer=array_answer(res),
                  hints=["Vector được \"kéo giãn\" thành mọi hàng (hoặc mọi cột) rồi phép toán thực hiện từng phần tử."],
                  steps=[Step(text=f"`{expr}` broadcast `v` (shape `{tup(np.shape(v))}`) lên shape `{tup(a.shape)}`.")], verified=["numpy"], tags=[which])
    if level == 4:  # cột + hàng -> ma trận
        n, m = rng.randint(2, 3), rng.randint(2, 4)
        col = np.array([[rng.randint(0, 4)] for _ in range(n)])
        row = np.array([rng.randint(0, 4) for _ in range(m)])
        op = rng.choice(["+", "*"])
        expr = f"col {op} row"
        res = run(expr, col=col, row=row)
        code = f"import numpy as np\n\ncol = np.array({col.tolist()})   # shape {tup(col.shape)}\nrow = np.array({row.tolist()})    # shape {tup(row.shape)}\nprint({expr})"
        return nb(type_="np-broadcast-value", lesson=L25, level=4, seed=seed,
                  prompt=code_prompt(f"Điền kết quả in ra ({n} × {m}). Mỗi phần tử $(i, j)$ kết hợp `col[i]` với `row[j]`.", code), answer=array_answer(res),
                  hints=["Cột $(n, 1)$ và hàng $(m,)$ broadcast thành $(n, m)$: phần tử $(i, j)$ là `col[i] op row[j]`."],
                  steps=[Step(text=f"Kết quả có shape `{tup(res.shape)}`.")], verified=["numpy"], tags=["outer"])
    # level 5: căn giữa dữ liệu theo cột
    for _ in range(200):
        n, d = rng.randint(2, 4), rng.randint(2, 4)
        x = _mat(rng, n, d, 0, 9)
        if np.all((x.mean(axis=0) * 2) % 1 == 0):
            break
    expr = "X - X.mean(axis=0)"
    res = run(expr, X=x)
    code = f"import numpy as np\n\nX = np.array({x.tolist()})   # {n} mẫu, {d} đặc trưng\nprint({expr})"
    return nb(type_="np-broadcast-value", lesson=L25, level=5, seed=seed,
              prompt=code_prompt("Căn giữa dữ liệu (trừ trung bình từng cột, bước đầu của PCA). Điền kết quả in ra; số thực nhập dạng $p/q$ hoặc thập phân.", code),
              answer=array_answer(res),
              hints=["`X.mean(axis=0)` là trung bình từng cột, shape `(d,)`; phép trừ broadcast nó lên mọi hàng."],
              steps=[Step(text=f"Trung bình các cột: `{x.mean(axis=0).tolist()}`.")], verified=["numpy"], tags=["ml", "centering"])


# --------------------------------------------------------------------- 2.6
def gen_reshape(rng: random.Random, level: int, seed: int):
    if level == 2:
        r, c = rng.choice([(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (2, 6)])
        n = r * c
        new = rng.choice([(c, r), (n // 2, 2), (2, n // 2), (n,), (1, n)]) if n % 2 == 0 else (c, r)
        a = np.arange(n) + rng.randint(0, 40)
        expr = f"a.reshape({new})" if len(new) > 1 else f"a.reshape({new[0]})"
        res = run(expr, a=a)
        if res.ndim == 2 and max(res.shape) <= 6 and min(res.shape) >= 1:
            code = f"import numpy as np\n\na = np.arange({int(a[0])}, {int(a[0]) + n})\nprint(a.reshape{tup(new) if len(new) > 1 else '(' + str(new[0]) + ')'})"
            return nb(type_="np-reshape", lesson=L26, level=2, seed=seed,
                      prompt=code_prompt(f"Điền kết quả in ra ({res.shape[0]} × {res.shape[1]}). `reshape` đọc các phần tử theo thứ tự hàng trước (row-major).", code),
                      answer=array_answer(res), hints=["Các phần tử giữ nguyên thứ tự, chỉ đổi cách chia thành hàng và cột."],
                      steps=[Step(text=f"Mảng 1 chiều {n} phần tử xếp thành `{tup(res.shape)}` theo từng hàng.")], verified=["numpy"], tags=["value"])
        return gen_reshape(random.Random(rng.random()), level, seed)
    if level == 3:  # -1 và shape
        a, b, c = rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)
        x = np.zeros((a, b, c))
        expr, _ = rng.choice([("x.reshape(-1, c)".replace("c", str(c)), ""), (f"x.reshape({a}, -1)", ""), (f"x.reshape(-1, {b * c})", ""), (f"x.reshape({a * b}, -1)", ""), ("x.reshape(-1)", ""), (f"x.reshape({b}, -1, {c})" if (a * b * c) % (b * c) == 0 and a == b else f"x.reshape(-1, {a})", "")])
        try:
            res = run(expr, x=x).shape
        except ValueError:
            return gen_reshape(random.Random(rng.random()), level, seed)
        cands = [res, (a, b * c), (a * b, c), (a * b * c,), (c, b, a), (a, b, c)]
        choices, idx = make_choices(rng, [tup(res)], [tup(s) for s in cands if s != res], 4)
        code = f"import numpy as np\n\nx = np.zeros({tup((a, b, c))})\nprint({expr}.shape)"
        return nb(type_="np-reshape", lesson=L26, level=3, seed=seed, prompt=code_prompt("`shape` nào được in ra? (`-1` nghĩa là \"tự suy ra\")", code),
                  answer=choice_answer(idx), choices=choices,
                  hints=["Tổng số phần tử không đổi; `-1` được tính từ phần còn lại."], steps=[Step(text=f"Tổng {a * b * c} phần tử, kết quả `{tup(res)}`.")],
                  verified=["numpy"], tags=["shape"])
    if level == 4:  # stack / concatenate
        r, c = rng.randint(2, 4), rng.randint(2, 4)
        kind = rng.choice(["concatenate0", "concatenate1", "vstack", "hstack", "stack0", "stack1", "stack2"])
        a, b = np.zeros((r, c)), np.zeros((r, c))
        expr = {"concatenate0": "np.concatenate([a, b], axis=0)", "concatenate1": "np.concatenate([a, b], axis=1)", "vstack": "np.vstack([a, b])",
                "hstack": "np.hstack([a, b])", "stack0": "np.stack([a, b], axis=0)", "stack1": "np.stack([a, b], axis=1)", "stack2": "np.stack([a, b], axis=2)"}[kind]
        res = run(expr, a=a, b=b).shape
        cands = [res, (2 * r, c), (r, 2 * c), (2, r, c), (r, 2, c), (r, c, 2), (r, c)]
        choices, idx = make_choices(rng, [tup(res)], [tup(s) for s in cands if s != res], 4)
        code = f"import numpy as np\n\na = np.zeros({tup((r, c))})\nb = np.zeros({tup((r, c))})\nprint({expr}.shape)"
        return nb(type_="np-reshape", lesson=L26, level=4, seed=seed, prompt=code_prompt("`shape` nào được in ra?", code),
                  answer=choice_answer(idx), choices=choices,
                  hints=["`concatenate` nối dọc theo một trục **đã có**; `stack` tạo trục **mới**."], steps=[Step(text=f"`{expr}` cho `{tup(res)}`.")],
                  verified=["numpy"], tags=[kind])
    # level 5: ảnh -> vector đặc trưng
    n, h, w = rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 4)
    expr, why = rng.choice([("imgs.reshape(imgs.shape[0], -1)", "trải phẳng mỗi ảnh thành một hàng"), ("imgs.reshape(-1, imgs.shape[2])", "mỗi hàng ảnh thành một mẫu"),
                            ("np.concatenate([imgs, imgs], axis=0)", "nối thêm một bản sao của batch"), ("np.stack([imgs, imgs], axis=1)", "xếp hai bản sao thành trục mới")])
    imgs = np.zeros((n, h, w))
    res = run(expr, imgs=imgs, np=np).shape
    cands = [res, (n, h * w), (n * h, w), (2 * n, h, w), (n, 2, h, w), (n, h, w)]
    choices, idx = make_choices(rng, [tup(res)], [tup(s) for s in cands if s != res], 4)
    code = f"import numpy as np\n\nimgs = np.zeros({tup((n, h, w))})   # {n} ảnh xám {h}x{w}\nprint(({expr}).shape)"
    return nb(type_="np-reshape", lesson=L26, level=5, seed=seed, prompt=code_prompt("`shape` nào được in ra?", code),
              answer=choice_answer(idx), choices=choices, hints=["Theo dõi tổng số phần tử và trục nào được gộp hoặc thêm."],
              steps=[Step(text=f"`{expr}` ({why}): `{tup(res)}`.")], verified=["numpy"], tags=["ml"])


SPECS = [
    TopicSpec("np-broadcast-shape", L25, (2, 3, 4), gen_broadcast_shape),
    TopicSpec("np-broadcast-value", L25, (3, 4, 5), gen_broadcast_value),
    TopicSpec("np-reshape", L26, (2, 3, 4, 5), gen_reshape),
]
