"""Tiện ích cho Phần 5 (Tối ưu): mọi phép tính bằng Fraction chính xác; kiểm chứng độc lập bằng SymPy, NumPy hoặc liệt kê."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import Matrix, Rational

from ...core import latex as L
from ...core.verify import check
from ...schema import Answer, Check, Prompt, Step
from ..linalg._base import TopicSpec, build, shuffled_choices

PART = "optimization"
F = Fraction

NUM_HINT = "Nhập phân số $p/q$ hoặc số thập phân."
VEC_HINT = "Nhập từng thành phần (phân số $p/q$ hoặc số thập phân)."


def ob(**kw):
    return build(part=PART, **kw)


def R(x) -> Rational:
    x = F(x)
    return Rational(x.numerator, x.denominator)


def ft(x) -> str:
    """LaTeX của Fraction."""
    return L.num(R(x))


def fp(x) -> str:
    """LaTeX của Fraction, bọc ngoặc khi âm."""
    return L.num_paren(R(x))


def fs(x) -> str:
    return L.rat_str(R(x))


def vt(vals) -> str:
    """Vector cột viết ngang: (a, b)^T."""
    return "(" + ", ".join(ft(v) for v in vals) + ")\\T"


def num_ans(x) -> Answer:
    return Answer(kind="number", value=fs(x))


def vec_ans(vals) -> Answer:
    return Answer(kind="vector", rows=len(vals), cols=1, entries=[[fs(v)] for v in vals])


def mat_tex(rows) -> str:
    body = " \\\\ ".join(" & ".join(ft(v) for v in r) for r in rows)
    return f"\\begin{{bmatrix}} {body} \\end{{bmatrix}}"


def nice(vals, max_den: int = 120, max_abs: int = 400) -> bool:
    """Giá trị đủ gọn để người học tính tay: mẫu số và độ lớn bị chặn."""
    return all(F(v).denominator <= max_den and abs(F(v)) <= max_abs for v in vals)


def pick(rng: random.Random, lo: int, hi: int, nonzero: bool = True) -> int:
    while True:
        v = rng.randint(lo, hi)
        if v or not nonzero:
            return v


def choice_exercise(rng, correct: str, wrong: list[str]):
    """(choices, answer) từ một đáp án đúng và các đáp án sai."""
    choices, idx = shuffled_choices(rng, [correct], wrong)
    return choices, Answer(kind="choice", correct=idx)


def lf(coeffs, names) -> str:
    """Tổ hợp tuyến tính a1 n1 + a2 n2 ...; tên "1" nghĩa là số hạng hằng."""
    out = ""
    for c, n in zip(coeffs, names):
        c = Rational(c)
        if c == 0:
            continue
        mag = abs(c)
        body = L.num(mag) if n == "1" else (n if mag == 1 else f"{L.num(mag)}{n}")
        out += (("-" if c < 0 else "") + body) if not out else ((" - " if c < 0 else " + ") + body)
    return out or "0"


def mat_vec(m, v):
    return [sum(F(m[i][j]) * v[j] for j in range(len(v))) for i in range(len(m))]


__all__ = ["lf", "ob", "R", "ft", "fp", "fs", "vt", "num_ans", "vec_ans", "mat_tex", "nice", "pick", "choice_exercise", "mat_vec", "F", "NUM_HINT", "VEC_HINT",
           "TopicSpec", "Answer", "Check", "Prompt", "Step", "check", "random", "Matrix", "Rational", "L"]
