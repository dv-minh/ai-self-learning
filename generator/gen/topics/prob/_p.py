"""Tiện ích cho Phần 4 (Xác suất – thống kê): mọi tính toán bằng phân số chính xác, kiểm chứng bằng liệt kê."""
from __future__ import annotations

import random
from itertools import product

from sympy import Matrix, Rational

from ...core import latex as L
from ...core.verify import check
from ...schema import Answer, Check, Prompt, Step
from ..linalg._base import TopicSpec, build

PART = "probability"


def pb(**kw):
    return build(part=PART, **kw)


def R(a, b=1) -> Rational:
    return Rational(a, b)


def num_ans(x) -> Answer:
    return Answer(kind="number", value=str(Rational(x)))


def vec_ans(vals) -> Answer:
    return Answer(kind="vector", rows=len(vals), cols=1, entries=[[str(Rational(v))] for v in vals])


def mat_ans(m: Matrix) -> Answer:
    return Answer(kind="matrix", rows=m.rows, cols=m.cols, entries=[[str(Rational(m[i, j])) for j in range(m.cols)] for i in range(m.rows)])


def choice_ans(idx: int) -> Answer:
    return Answer(kind="choice", correct=[idx])


def frac(x) -> str:
    """LaTeX của phân số $p/q$."""
    return L.num(Rational(x))


def pmf_array(xs, ps) -> str:
    """Bảng phân phối dạng mảng LaTeX: dòng $x$ và dòng $P(X=x)$."""
    cols = "c" * (len(xs) + 1)
    top = " & ".join(["x"] + [str(v) for v in xs])
    bot = " & ".join(["P(X=x)"] + [frac(p) for p in ps])
    return f"\\begin{{array}}{{{cols}}} {top} \\\\ \\hline {bot} \\end{{array}}"


def table_array(header: list[str], rows: list[list]) -> str:
    cols = "c" * len(header)
    head = " & ".join(header)
    body = " \\\\ ".join(" & ".join(str(c) for c in r) for r in rows)
    return f"\\begin{{array}}{{{cols}}} {head} \\\\ \\hline {body} \\end{{array}}"


def shuffled(rng: random.Random, correct: list[str], wrong: list[str], n: int = 4) -> tuple[list[str], list[int]]:
    wrong = [w for i, w in enumerate(wrong) if w not in correct and w not in wrong[:i]]
    rng.shuffle(wrong)
    items = [(c, True) for c in correct] + [(w, False) for w in wrong[: max(n - len(correct), 1)]]
    rng.shuffle(items)
    return [c for c, _ in items], [i for i, (_, ok) in enumerate(items) if ok]


__all__ = ["pb", "R", "num_ans", "vec_ans", "mat_ans", "choice_ans", "frac", "pmf_array", "table_array", "shuffled", "TopicSpec", "Answer", "Check", "Prompt",
           "Step", "check", "Rational", "Matrix", "product", "random", "L"]
