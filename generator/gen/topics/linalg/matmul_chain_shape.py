"""1.5 matmul-chain-shape: kích thước của tích nhiều ma trận (mức 4) và shape trong ML (mức 5)."""
from __future__ import annotations

import random

import numpy as np

from ...core.verify import check
from ...schema import Exercise, Prompt, Step
from ._base import TopicSpec, build, shape_answer, shape_tex

TYPE, LESSON, LEVELS = "matmul-chain-shape", "1.5", (4, 5)


def _chain(tokens: list[tuple[str, bool, tuple[int, int]]]) -> tuple[int, int] | None:
    """Kích thước tích theo NumPy; token = (tên, có chuyển vị, kích thước lưu trữ)."""
    acc = None
    try:
        for _, t, shp in tokens:
            z = np.zeros(shp)
            z = z.T if t else z
            acc = z if acc is None else acc @ z
    except ValueError:
        return None
    return acc.shape


def _expected(tokens) -> tuple[int, int] | None:
    eff = [(s[::-1] if t else s) for _, t, s in tokens]
    for x, y in zip(eff, eff[1:]):
        if x[1] != y[0]:
            return None
    return (eff[0][0], eff[-1][1])


def _name(n: str, t: bool) -> str:
    return f"\\mathbf{{{n}}}" + ("\\T" if t else "")


def _level4(rng: random.Random, seed: int) -> Exercise:
    names = ["A", "B", "C"]
    flags = [rng.random() < 0.3 for _ in names]
    d = [rng.randint(1, 4) for _ in range(4)]
    eff = [(d[i], d[i + 1]) for i in range(3)]
    if rng.random() < 0.3:  # cố ý làm lệch một kích thước
        k = rng.randrange(2)  # đổi số cột của thừa số k để lệch với số hàng thừa số k+1
        eff[k] = (eff[k][0], rng.choice([x for x in range(1, 5) if x != eff[k][1]]))
    tokens = [(n, f, (e[::-1] if f else e)) for n, f, e in zip(names, flags, eff)]
    result = _chain(tokens)
    check(result == _expected(tokens), "chuỗi nhân: NumPy khác dự kiến")
    expr = "".join(_name(n, f) for n, f, _ in tokens)
    dims = "; ".join(f"${_name(n, False)}$ là ${shape_tex(s)}$" for n, _, s in tokens)
    steps = []
    eff_shapes = [(s[::-1] if t else s) for _, t, s in tokens]
    for (n1, t1, _), (n2, t2, _), e1, e2 in zip(tokens, tokens[1:], eff_shapes, eff_shapes[1:]):
        steps.append(Step(
            text=f"Ghép ${_name(n1, t1)}$ với ${_name(n2, t2)}$: số cột {e1[1]} so với số hàng {e2[0]}.",
            latex=f"({shape_tex(e1)})({shape_tex(e2)})",
        ))
        if e1[1] != e2[0]:
            steps.append(Step(text="Không bằng nhau nên tích không xác định."))
            break
    else:
        steps.append(Step(text="Mọi cặp liền kề khớp. Kích thước là (hàng của thừa số đầu) × (cột của thừa số cuối).",
                          latex=shape_tex(result)))
    return build(
        type_=TYPE, lesson=LESSON, level=4, seed=seed,
        prompt=Prompt(text=f"Biết {dims}. Tích ${expr}$ có kích thước bao nhiêu (hoặc không xác định)?"),
        answer=shape_answer(result),
        hints=["Kích thước của $\\mathbf{M}\\T$ là kích thước của $\\mathbf{M}$ đảo hàng và cột.",
               "Kiểm tra từng cặp liền kề: số cột bên trái phải bằng số hàng bên phải."],
        steps=steps, verified=["sympy", "numpy", "shape"], tags=["chain"],
    )


def _level5(rng: random.Random, seed: int) -> Exercise:
    kind = rng.choice(["linear", "linear-T", "attention", "two-layer"])
    n, d, h, k = (rng.randint(1, 4) for _ in range(4))
    if kind == "linear":
        text = (f"Một lớp tuyến tính nhận ma trận dữ liệu $\\mathbf{{X}}$ gồm ${n}$ mẫu, mỗi mẫu ${d}$ đặc trưng "
                f"(mỗi hàng là một mẫu), và ma trận trọng số $\\mathbf{{W}}$ kích thước ${d}\\times {h}$. "
                f"Kích thước của $\\mathbf{{XW}}$ là gì?")
        toks = [("X", False, (n, d)), ("W", False, (d, h))]
        why = "Mỗi hàng của $\\mathbf{X}$ là một mẫu; nhân với $\\mathbf{W}$ cho đầu ra ứng với từng mẫu."
    elif kind == "linear-T":
        text = (f"Trọng số được lưu dạng $\\mathbf{{W}}$ kích thước ${h}\\times {d}$ (mỗi hàng là một nơ-ron). "
                f"Dữ liệu $\\mathbf{{X}}$ có kích thước ${n}\\times {d}$. Kích thước của $\\mathbf{{X}}\\mathbf{{W}}\\T$ là gì?")
        toks = [("X", False, (n, d)), ("W", True, (h, d))]
        why = "$\\mathbf{W}\\T$ có kích thước $d\\times h$ nên nhân được với $\\mathbf{X}$."
    elif kind == "attention":
        text = (f"Trong attention, ma trận truy vấn $\\mathbf{{Q}}$ có kích thước ${n}\\times {d}$ và ma trận khóa "
                f"$\\mathbf{{K}}$ có kích thước ${k}\\times {d}$. Kích thước của $\\mathbf{{Q}}\\mathbf{{K}}\\T$ là gì?")
        toks = [("Q", False, (n, d)), ("K", True, (k, d))]
        why = "Điểm attention giữa mỗi truy vấn và mỗi khóa tạo thành ma trận có số hàng bằng số truy vấn, số cột bằng số khóa."
    else:
        text = (f"Mạng hai lớp: $\\mathbf{{X}}$ có kích thước ${n}\\times {d}$, $\\mathbf{{W}}_1$ có kích thước "
                f"${d}\\times {h}$, $\\mathbf{{W}}_2$ có kích thước ${h}\\times {k}$. Kích thước của "
                f"$\\mathbf{{X}}\\mathbf{{W}}_1\\mathbf{{W}}_2$ là gì? (bỏ qua hàm kích hoạt)")
        toks = [("X", False, (n, d)), ("W1", False, (d, h)), ("W2", False, (h, k))]
        why = "Mỗi lớp đổi số cột: $d\\to h\\to k$, số hàng (số mẫu) giữ nguyên."
    result = _chain(toks)
    check(result == _expected(toks), "L5: NumPy khác dự kiến")
    eff = [(s[::-1] if t else s) for _, t, s in toks]
    steps = [Step(text=f"Kích thước hiệu dụng các thừa số theo thứ tự: " + ", ".join(f"${shape_tex(e)}$" for e in eff) + "."),
             Step(text=why, latex=shape_tex(result))]
    return build(
        type_=TYPE, lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text=text),
        answer=shape_answer(result),
        hints=["Viết kích thước từng thừa số cạnh nhau và kiểm tra các số ở giữa có khớp không.",
               "Kích thước kết quả lấy từ số hàng thừa số đầu và số cột thừa số cuối."],
        steps=steps, verified=["sympy", "numpy", "shape"], tags=["ml", kind],
    )


def generate(rng: random.Random, level: int, seed: int) -> Exercise:
    return _level4(rng, seed) if level == 4 else _level5(rng, seed)


SPEC = TopicSpec(TYPE, LESSON, LEVELS, generate)
