"""Tiện ích cho Phần 2 (NumPy): chạy code thật, dựng đáp án từ kết quả NumPy, tạo lựa chọn nhiễu."""
from __future__ import annotations

import random
from fractions import Fraction

import numpy as np

from ...core.verify import check
from ...schema import Answer, Check, Exercise, Prompt, Step
from ..linalg._base import TopicSpec, build

PART = "numpy"


def nb(**kw) -> Exercise:
    return build(part=PART, **kw)


def run(expr: str, **ns):
    """Tính một biểu thức NumPy thật (mã do generator tự viết, không nhận từ bên ngoài)."""
    return eval(expr, {"np": np, "__builtins__": __builtins__}, ns)


def exec_code(code: str, **ns) -> dict:
    env = {"np": np, **ns}
    exec(code, {"__builtins__": __builtins__}, env)
    return env


def tup(shape: tuple) -> str:
    return "()" if shape == () else (f"({shape[0]},)" if len(shape) == 1 else "(" + ", ".join(str(s) for s in shape) + ")")


def arr_src(a: np.ndarray) -> str:
    return f"np.array({a.tolist()})"


def rat(x) -> str:
    """float/int của NumPy -> chuỗi "p/q" chính xác (kiểm tra không mất độ chính xác)."""
    f = Fraction(float(x)).limit_denominator(1000)
    check(abs(float(f) - float(x)) < 1e-12, f"không biểu diễn chính xác {x}")
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def array_answer(a: np.ndarray) -> Answer:
    """Mảng 1 chiều -> vector cột; 2 chiều -> ma trận. Phần tử là số hữu tỉ chính xác."""
    a = np.asarray(a)
    if a.ndim == 1:
        return Answer(kind="vector", rows=a.shape[0], cols=1, entries=[[rat(x)] for x in a])
    check(a.ndim == 2, "chỉ hỗ trợ mảng 1-2 chiều")
    return Answer(kind="matrix", rows=a.shape[0], cols=a.shape[1], entries=[[rat(x) for x in row] for row in a])


def number_answer(x) -> Answer:
    return Answer(kind="number", value=rat(x))


def choice_answer(correct: list[int]) -> Answer:
    return Answer(kind="choice" if len(correct) == 1 else "multi", correct=correct)


def make_choices(rng: random.Random, correct: list[str], wrong: list[str], n: int = 4) -> tuple[list[str], list[int]]:
    """Trộn lựa chọn: tất cả đáp án đúng + đủ nhiễu để có tối đa n lựa chọn; loại trùng."""
    wrong = [w for i, w in enumerate(wrong) if w not in correct and w not in wrong[:i]]
    rng.shuffle(wrong)
    items = [(c, True) for c in correct] + [(w, False) for w in wrong[: max(n - len(correct), 1)]]
    rng.shuffle(items)
    return [c for c, _ in items], [i for i, (_, ok) in enumerate(items) if ok]


def code_prompt(text: str, code: str, display: list[str] | None = None) -> Prompt:
    return Prompt(text=text, display=display or [], code=code)


__all__ = ["nb", "run", "exec_code", "tup", "arr_src", "rat", "array_answer", "number_answer", "choice_answer", "make_choices",
           "code_prompt", "TopicSpec", "Answer", "Check", "Step", "check", "np", "random"]
