r"""Ví dụ số bài 1.8 Định thức."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational

from ..core import latex as L
from ..core.gauss import eliminate
from ..core.verify import check, verify_det
from ..topics.linalg.l1_8_det import det2_expr, laplace_steps, sarrus_expr
from ..topics.linalg.l1_7_gauss import _lu
from .common import M, V, display, rng, steps_md, write_examples

LESSON = "1.8"
B, N = L.bold, L.num_paren


def det2() -> str:
    a = M(11, 2, 2, -6, 6)
    d = a.det()
    verify_det(a, d)
    return display(L.mat(a, B("A")), f"\\det(\\mathbf{{A}}) = {det2_expr(a)} = {L.num(d)}")


def sarrus() -> str:
    a = M(21, 3, 3, -5, 5)
    d = a.det()
    verify_det(a, d)
    return display(L.mat(a, B("A")), f"\\det(\\mathbf{{A}}) = {sarrus_expr(a)} = {L.num(d)}")


def laplace4() -> str:
    from ..topics.linalg.l1_8_det import _sparse4

    a = _sparse4(rng(31), -5, 5)
    d = a.det()
    verify_det(a, d)
    counts = [sum(1 for j in range(4) if a[i, j] == 0) for i in range(4)]
    row = counts.index(max(counts))
    return display(L.mat(a, B("A"))) + "\n\n" + steps_md(laplace_steps(a, d, row=row))


def rowops_det() -> str:
    a, _u = _lu(rng(41), 3, 3, dmax=3, off=3, lmax=2)
    d = a.det()
    verify_det(a, d)
    el = eliminate(a, mode="ref")
    prod = 1
    for i in range(3):
        prod *= el.matrix[i, i]
    check((-1) ** el.swaps * prod == d, "det rowops")
    return (
        steps_md(el.steps)
        + f"\n\nMa trận cuối là tam giác trên, và ta không đổi hàng lần nào nên định thức bằng tích đường chéo:\n\n"
        + display(f"\\det(\\mathbf{{A}}) = " + " \\cdot ".join(N(el.matrix[i, i]) for i in range(3)) + f" = {L.num(d)}")
        if el.swaps == 0
        else steps_md(el.steps) + "\n\n" + display(f"\\det(\\mathbf{{A}}) = (-1)^{{{el.swaps}}}\\cdot " + " \\cdot ".join(N(el.matrix[i, i]) for i in range(3)) + f" = {L.num(d)}")
    )


def area() -> str:
    u, v = V(51, 2, -5, 5), V(52, 2, -5, 5)
    a = u.row_join(v)
    d = a.det()
    return (
        f"Hai vector làm hai cạnh của một hình bình hành:\n\n"
        + display(L.mat(u, B("u")), L.mat(v, B("v")), f"\\det\\begin{{bmatrix}} \\mathbf{{u}} & \\mathbf{{v}} \\end{{bmatrix}} = {det2_expr(a)} = {L.num(d)}")
        + f"\n\nDiện tích hình bình hành là $|\\det| = {L.num(abs(d))}$."
    )


def mistake_add() -> str:
    a, b = M(61, 2, 2, -4, 4), M(62, 2, 2, -4, 4)
    while (a + b).det() == a.det() + b.det():
        b = M(62 + 1000 + int(abs(b[0, 0])), 2, 2, -4, 4)
    return (
        display(L.mat(a, B("A")), L.mat(b, B("B")))
        + "\n\n**Sai:** cho rằng định thức cộng được: "
        + f"$\\det(\\mathbf{{A}}) + \\det(\\mathbf{{B}}) = {L.num(a.det())} + {N(b.det())} = {L.num(a.det() + b.det())}$ "
        + "rồi coi đó là $\\det(\\mathbf{A}+\\mathbf{B})$.\n\n"
        + "**Đúng:** tính trực tiếp:\n\n"
        + display(f"\\mathbf{{A}}+\\mathbf{{B}} = {L.mat(a + b)},\\quad \\det(\\mathbf{{A}}+\\mathbf{{B}}) = {L.num((a + b).det())}")
        + "\n\nĐịnh thức **không** tuyến tính theo phép cộng ma trận. Chỉ có tính chất nhân: $\\det(\\mathbf{AB}) = \\det(\\mathbf{A})\\det(\\mathbf{B})$."
    )


def mistake_scalar() -> str:
    a = M(71, 3, 3, -3, 3)
    k = 2
    d = a.det()
    return (
        display(L.mat(a, B("A")))
        + f"\n\n**Sai:** $\\det({k}\\mathbf{{A}}) = {k}\\det(\\mathbf{{A}}) = {L.num(k * d)}$.\n\n"
        + f"**Đúng:** nhân cả $\\mathbf{{A}}$ với ${k}$ nghĩa là nhân **mỗi một trong {a.rows} hàng** với ${k}$, nên định thức nhân với ${k}^{{{a.rows}}}$:\n\n"
        + display(f"\\det({k}\\mathbf{{A}}) = {k}^{{{a.rows}}}\\det(\\mathbf{{A}}) = {k ** a.rows}\\cdot {N(d)} = {L.num(k ** a.rows * d)}",
                  f"\\det\\!\\left({L.mat(k * a)}\\right) = {L.num((k * a).det())}")
    )


EXAMPLES = {"det2": det2, "sarrus": sarrus, "laplace4": laplace4, "rowops-det": rowops_det, "area": area,
            "mistake-add": mistake_add, "mistake-scalar": mistake_scalar}


def prop_transpose() -> str:
    a = M(81, 3, 3, -4, 4)
    check(a.T.det() == a.det(), "det A^T")
    return display(L.mat(a, B("A")), f"\\det(\\mathbf{{A}}) = {L.num(a.det())},\\qquad \\det(\\mathbf{{A}}\\T) = {L.num(a.T.det())}")


def prop_product() -> str:
    a, b = M(82, 2, 2, -4, 4), M(83, 2, 2, -4, 4)
    check((a * b).det() == a.det() * b.det(), "det AB")
    return display(L.mat(a, B("A")), L.mat(b, B("B")),
                   f"\\det(\\mathbf{{AB}}) = {L.num((a * b).det())},\\qquad \\det(\\mathbf{{A}})\\det(\\mathbf{{B}}) = {N(a.det())}\\cdot {N(b.det())} = {L.num(a.det() * b.det())}")


def prop_swap() -> str:
    a = M(84, 3, 3, -4, 4)
    s = a.copy()
    s.row_swap(0, 1)
    check(s.det() == -a.det(), "swap sign")
    return display(L.mat(a, B("A")), f"R_1 \\leftrightarrow R_2:\\quad {L.mat(s)}", f"\\det(\\mathbf{{A}}) = {L.num(a.det())},\\qquad \\det(\\text{{ma trận sau đổi}}) = {L.num(s.det())}".replace("\\text{ma trận sau đổi}", "\\mathbf{A}'")) \
        + "\n\nĐổi chỗ hai hàng làm định thức đổi dấu."


def prop_equal_rows() -> str:
    a = M(85, 3, 3, -4, 4)
    a[2, :] = a[0, :]
    check(a.det() == 0, "equal rows")
    return display(L.mat(a, B("A")), f"\\det(\\mathbf{{A}}) = {L.num(a.det())}") + "\n\nHai hàng bằng nhau (hoặc tỉ lệ) thì định thức bằng $0$."


EXAMPLES.update({"prop-transpose": prop_transpose, "prop-product": prop_product, "prop-swap": prop_swap, "prop-equal-rows": prop_equal_rows})


def build():
    return write_examples(LESSON, EXAMPLES)
