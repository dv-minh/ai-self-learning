r"""Khử Gauss có ghi từng bước (docs/spec.md mục 5, "Lời giải từng bước").

Phép biến đổi sơ cấp:
  ("swap", i, j)        R_i <-> R_j
  ("scale", i, k)       R_i <- k R_i
  ("add", i, j, k)      R_i <- R_i + k R_j
Chỉ số hàng bắt đầu từ 0 trong mã, từ 1 khi hiển thị.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sympy import Matrix, Rational

from ..schema import Step
from . import latex as L


def coef(k) -> str:
    k = Rational(k)
    if k == 1:
        return ""
    if k == -1:
        return "-"
    return L.num(k)


def op_tex(op: tuple) -> str:
    kind = op[0]
    if kind == "swap":
        return f"R_{{{op[1] + 1}}} \\leftrightarrow R_{{{op[2] + 1}}}"
    if kind == "scale":
        return f"R_{{{op[1] + 1}}} \\leftarrow {coef(op[2])}R_{{{op[1] + 1}}}"
    _, i, j, k = op
    k = Rational(k)
    sign = "-" if k < 0 else "+"
    return f"R_{{{i + 1}}} \\leftarrow R_{{{i + 1}}} {sign} {coef(abs(k))}R_{{{j + 1}}}"


def apply_op(m: Matrix, op: tuple) -> Matrix:
    m = m.copy()
    kind = op[0]
    if kind == "swap":
        m.row_swap(op[1], op[2])
    elif kind == "scale":
        m[op[1], :] = m[op[1], :] * op[2]
    else:
        _, i, j, k = op
        m[i, :] = m[i, :] + k * m[j, :]
    return m


def show(m: Matrix, aug: int = 0) -> str:
    return L.aug(m, aug) if aug else L.mat(m)


@dataclass
class Elimination:
    matrix: Matrix
    pivots: list[int]
    ops: list[tuple]
    swaps: int
    steps: list[Step] = field(default_factory=list)


def eliminate(m: Matrix, aug: int = 0, mode: str = "ref") -> Elimination:
    """mode="ref": khử xuôi, không chuẩn hóa pivot (chỉ swap khi pivot = 0, và R_i <- R_i + kR_j).
    mode="rref": Gauss–Jordan, pivot chuẩn hóa về 1 và khử cả phía trên."""
    cur = m.copy()
    ncols = m.cols - aug
    steps = [Step(text="Ma trận ban đầu:", latex=show(cur, aug))]
    ops: list[tuple] = []
    pivots: list[int] = []
    swaps = 0
    row = 0

    def do(op):
        nonlocal cur, swaps
        cur = apply_op(cur, op)
        ops.append(op)
        if op[0] == "swap":
            swaps += 1
        steps.append(Step(text=f"${op_tex(op)}$:", latex=show(cur, aug)))

    for col in range(ncols):
        if row >= m.rows:
            break
        piv = next((r for r in range(row, m.rows) if cur[r, col] != 0), None)
        if piv is None:
            continue
        if piv != row:
            do(("swap", row, piv))
        if mode == "rref" and cur[row, col] != 1:
            do(("scale", row, 1 / cur[row, col]))
        targets = range(row + 1, m.rows) if mode == "ref" else [r for r in range(m.rows) if r != row]
        for r in targets:
            if cur[r, col] != 0:
                do(("add", r, row, -cur[r, col] / cur[row, col]))
        pivots.append(col)
        row += 1
    return Elimination(cur, pivots, ops, swaps, steps)
