"""Tiện ích cho Phần 6 (ML cổ điển): phân số chính xác, kiểm chứng độc lập bằng NumPy, SymPy hoặc liệt kê."""
from __future__ import annotations

from fractions import Fraction as F

from ...core import latex as L
from ...core.verify import check
from ...schema import Answer, Check, Prompt, Step
from ..linalg._base import TopicSpec, build, shuffled_choices
from ..opt._o import NUM_HINT, VEC_HINT, choice_exercise, fp, fs, ft, lf, mat_tex, nice, num_ans, pick, vec_ans, vt

PART = "classical-ml"


def cb(**kw):
    return build(part=PART, **kw)


def mat_ans(rows) -> Answer:
    return Answer(kind="matrix", rows=len(rows), cols=len(rows[0]), entries=[[fs(v) for v in r] for r in rows])


def data_table(names: list[str], cols: list[list]) -> str:
    """Bảng dữ liệu theo hàng: mỗi phần tử của `names` là một dòng (nhãn ASCII), `cols` là các giá trị tương ứng."""
    n = len(cols[0])
    head = "i & " + " & ".join(str(i + 1) for i in range(n))
    rows = " \\\\ ".join(f"{nm} & " + " & ".join(ft(v) if not isinstance(v, str) else v for v in col) for nm, col in zip(names, cols))
    return "\\begin{array}{c|" + "c" * n + "} " + head + " \\\\ \\hline " + rows + " \\end{array}"


def points_table(pts, labels=None, names=("x_1", "x_2")) -> str:
    n = len(pts)
    cols = [[p[j] for p in pts] for j in range(len(pts[0]))]
    nms = list(names[: len(cols)])
    if labels is not None:
        cols.append(labels)
        nms.append("y")
    return data_table(nms, cols)


__all__ = ["cb", "mat_ans", "data_table", "points_table", "F", "L", "check", "Answer", "Check", "Prompt", "Step", "TopicSpec", "shuffled_choices", "NUM_HINT",
           "VEC_HINT", "choice_exercise", "fp", "fs", "ft", "lf", "mat_tex", "nice", "num_ans", "pick", "vec_ans", "vt"]
