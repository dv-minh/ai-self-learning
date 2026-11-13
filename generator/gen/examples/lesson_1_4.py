r"""Ví dụ số bài 1.4 Cộng ma trận, nhân vô hướng."""
from __future__ import annotations

from sympy import Matrix, Rational

from ..core import latex as L
from ..core.verify import check, np_close, to_np
from .common import M, display, write_examples

LESSON = "1.4"
B, N = L.bold, L.num_paren


def _cells(m: Matrix, f) -> str:
    return L.expr_mat([[f(i, j) for j in range(m.cols)] for i in range(m.rows)])


def add_small() -> str:
    a, b = M(11, 2, 2), M(12, 2, 2)
    r = a + b
    np_close(r, to_np(a) + to_np(b))
    return display(L.mat(a, B("A")), L.mat(b, B("B")),
                   f"\\mathbf{{A}}+\\mathbf{{B}} = {_cells(a, lambda i, j: f'{N(a[i, j])} + {N(b[i, j])}')} = {L.mat(r)}")


def add_large() -> str:
    a, b = M(21, 3, 4, -9, 9), M(22, 3, 4, -9, 9)
    r = a - b
    np_close(r, to_np(a) - to_np(b))
    return display(L.mat(a, B("A")), L.mat(b, B("B")),
                   f"\\mathbf{{A}}-\\mathbf{{B}} = {_cells(a, lambda i, j: f'{N(a[i, j])} - {N(b[i, j])}')} = {L.mat(r)}")


def scale() -> str:
    a = M(31, 2, 3, -6, 6)
    k = Rational(-3)
    r = k * a
    np_close(r, float(k) * to_np(a))
    return display(L.mat(a, B("A")), f"{L.num(k)}\\mathbf{{A}} = {_cells(a, lambda i, j: f'{N(k)}\\cdot {N(a[i, j])}')} = {L.mat(r)}")


def lincomb() -> str:
    a, b = M(41, 2, 3), M(42, 2, 3)
    r = 2 * a - 3 * b
    np_close(r, 2 * to_np(a) - 3 * to_np(b))
    return display(L.mat(a, B("A")), L.mat(b, B("B")),
                   f"2\\mathbf{{A}} - 3\\mathbf{{B}} = {_cells(a, lambda i, j: f'2\\cdot {N(a[i, j])} - 3\\cdot {N(b[i, j])}')} = {L.mat(r)}")


def mixup() -> str:
    a, b = M(51, 2, 2, 0, 9), M(52, 2, 2, 0, 9)
    lam = Rational(1, 4)
    r = lam * a + (1 - lam) * b
    np_close(r, float(lam) * to_np(a) + (1 - float(lam)) * to_np(b))
    return display(L.mat(a, B("A")), L.mat(b, B("B")),
                   f"{L.num(lam)}\\mathbf{{A}} + {L.num(1 - lam)}\\mathbf{{B}} = "
                   f"{_cells(a, lambda i, j: f'{N(lam)}\\cdot {N(a[i, j])} + {N(1 - lam)}\\cdot {N(b[i, j])}')} = {L.mat(r)}")


def mistake_shape() -> str:
    a, b = M(61, 2, 3), M(62, 3, 2)
    return (
        display(L.mat(a, B("A")), L.mat(b, B("B")))
        + f"\n\n**Sai:** cộng $\\mathbf{{A}}+\\mathbf{{B}}$ bằng cách ghép các phần tử có mặt ở cả hai ma trận. "
        f"$\\mathbf{{A}}$ là ${a.rows}\\times {a.cols}$ còn $\\mathbf{{B}}$ là ${b.rows}\\times {b.cols}$ nên **không cùng kích thước**: phép cộng không xác định.\n\n"
        f"**Đúng:** muốn cộng được ta cần $\\mathbf{{B}}\\T$ (cỡ ${b.cols}\\times {b.rows}$), khi đó:\n\n"
        + display(f"\\mathbf{{A}}+\\mathbf{{B}}\\T = {L.mat(a + b.T)}")
    )


EXAMPLES = {"add-small": add_small, "add-large": add_large, "scale": scale, "lincomb": lincomb, "mixup": mixup, "mistake-shape": mistake_shape}


def prop_commute() -> str:
    a, b = M(71, 2, 3), M(72, 2, 3)
    check(a + b == b + a, "commute")
    return display(f"\\mathbf{{A}}+\\mathbf{{B}} = {L.mat(a + b)},\\qquad \\mathbf{{B}}+\\mathbf{{A}} = {L.mat(b + a)}") \
        + f"\n\nvới $\\mathbf{{A}} = {L.mat(a)}$ và $\\mathbf{{B}} = {L.mat(b)}$."


def prop_distribute() -> str:
    a, b = M(73, 2, 2), M(74, 2, 2)
    k = Rational(-2)
    check(k * (a + b) == k * a + k * b, "distribute")
    return f"Với $k = {L.num(k)}$, $\\mathbf{{A}} = {L.mat(a)}$, $\\mathbf{{B}} = {L.mat(b)}$:\n\n" + display(
        f"k(\\mathbf{{A}}+\\mathbf{{B}}) = {L.mat(k * (a + b))},\\qquad k\\mathbf{{A}}+k\\mathbf{{B}} = {L.mat(k * a + k * b)}")


def prop_transpose_sum() -> str:
    a, b = M(75, 2, 3), M(76, 2, 3)
    check((a + b).T == a.T + b.T, "transpose sum")
    return f"Với $\\mathbf{{A}} = {L.mat(a)}$ và $\\mathbf{{B}} = {L.mat(b)}$:\n\n" + display(
        f"(\\mathbf{{A}}+\\mathbf{{B}})\\T = {L.mat((a + b).T)},\\qquad \\mathbf{{A}}\\T+\\mathbf{{B}}\\T = {L.mat(a.T + b.T)}")


def prop_associative() -> str:
    a, b, c = M(77, 2, 2), M(78, 2, 2), M(79, 2, 2)
    check((a + b) + c == a + (b + c), "associative")
    return f"Với $\\mathbf{{A}} = {L.mat(a)}$, $\\mathbf{{B}} = {L.mat(b)}$, $\\mathbf{{C}} = {L.mat(c)}$:\n\n" + display(
        f"(\\mathbf{{A}}+\\mathbf{{B}})+\\mathbf{{C}} = {L.mat((a + b) + c)},\\qquad \\mathbf{{A}}+(\\mathbf{{B}}+\\mathbf{{C}}) = {L.mat(a + (b + c))}")


EXAMPLES.update({"prop-commute": prop_commute, "prop-distribute": prop_distribute, "prop-transpose-sum": prop_transpose_sum,
                 "prop-associative": prop_associative})


def build():
    return write_examples(LESSON, EXAMPLES)
