r"""Ví dụ số bài 1.6 Ma trận đặc biệt."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational, diag, eye

from ..core import latex as L
from ..core.verify import check, np_close, to_np
from .common import M, display, write_examples

LESSON = "1.6"
B, N = L.bold, L.num_paren


def special_gallery() -> str:
    d = diag(2, -1, 3)
    up = Matrix([[1, 4, -2], [0, 3, 5], [0, 0, -6]])
    lo = up.T
    sym = Matrix([[2, -1, 4], [-1, 5, 0], [4, 0, -3]])
    q = Matrix([[3, -4], [4, 3]]) / 5
    check(sym == sym.T and q.T * q == eye(2), "gallery")
    return (
        f"- **Ma trận đơn vị:** ${L.mat(eye(3), chr(92) + 'mathbf{I}_3')}$\n"
        f"- **Ma trận đường chéo:** ${L.mat(d, B('D'))}$\n"
        f"- **Tam giác trên:** ${L.mat(up, B('U'))}$ và **tam giác dưới** là chuyển vị của nó: ${L.mat(lo, B('L'))}$\n"
        f"- **Đối xứng** ($\\mathbf{{S}} = \\mathbf{{S}}\\T$): ${L.mat(sym, B('S'))}$\n"
        f"- **Trực giao** ($\\mathbf{{Q}}\\T\\mathbf{{Q}} = \\mathbf{{I}}$): ${L.mat(q, B('Q'))}$, kiểm tra: ${L.mat(q.T * q)}$"
    )


def power_diag() -> str:
    d = diag(2, -1, 3)
    k = 4
    r = d ** k
    np_close(r, np.linalg.matrix_power(to_np(d), k))
    return display(L.mat(d, B("D")), f"\\mathbf{{D}}^{{{k}}} = \\diag(" + ", ".join(f"{N(x)}^{{{k}}}" for x in [d[0, 0], d[1, 1], d[2, 2]]) + f") = {L.mat(r)}")


def orthogonal_check() -> str:
    q = Matrix([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
    qq = q.T * q
    check(qq == eye(3), "orth")
    np_close(qq, to_np(q).T @ to_np(q))
    return display(L.mat(q, B("Q")), f"\\mathbf{{Q}}\\T\\mathbf{{Q}} = {L.mat(q.T)}{L.mat(q)} = {L.mat(qq)} = \\mathbf{{I}}_3")


def mistake_power() -> str:
    a = M(71, 2, 2, -3, 3)
    while a[0, 1] == 0 or a[1, 0] == 0:
        a = M(71 + 1000 + int(abs(a[0, 0])), 2, 2, -3, 3)
    sq = a * a
    wrong = a.applyfunc(lambda x: x * x)
    return (
        "**Sai:** với ma trận không phải đường chéo, lũy thừa từng phần tử:\n\n"
        + display(L.mat(a, B("A")), f"\\mathbf{{A}}^2 \\overset{{?}}{{=}} {L.mat(wrong)}")
        + "\n\n**Đúng:** $\\mathbf{A}^2 = \\mathbf{A}\\mathbf{A}$ là tích ma trận:\n\n"
        + display(f"\\mathbf{{A}}^2 = {L.mat(a)}{L.mat(a)} = {L.mat(sq)}")
        + "\n\nCách lũy thừa từng phần tử chỉ đúng khi ma trận là ma trận đường chéo."
    )


EXAMPLES = {"gallery": special_gallery, "power-diag": power_diag, "orthogonal-check": orthogonal_check, "mistake-power": mistake_power}


def prop_diag_commute() -> str:
    d1, d2 = diag(2, -1, 3), diag(-3, 4, 1)
    check(d1 * d2 == d2 * d1, "diag commute")
    return display(f"\\mathbf{{D}}_1\\mathbf{{D}}_2 = {L.mat(d1 * d2)} = \\mathbf{{D}}_2\\mathbf{{D}}_1") \
        + f"\n\nvới $\\mathbf{{D}}_1 = {L.mat(d1)}$ và $\\mathbf{{D}}_2 = {L.mat(d2)}$."


def prop_orth_inverse() -> str:
    q = Matrix([[3, -4], [4, 3]]) / 5
    check(q.T * q == eye(2) and q.inv() == q.T, "orth inverse")
    return display(L.mat(q, B("Q")), f"\\mathbf{{Q}}^{{-1}} = {L.mat(q.inv())} = \\mathbf{{Q}}\\T = {L.mat(q.T)}") \
        + "\n\nVới ma trận trực giao, nghịch đảo chính là chuyển vị: không cần tính nghịch đảo."


def prop_orth_norm() -> str:
    from ..core.nice import int_norm_vector

    q = Matrix([[3, -4], [4, 3]]) / 5
    x = int_norm_vector(__import__("random").Random(5), 2, 12)
    nx = (x.T * x)[0, 0] ** Rational(1, 2)
    qx = q * x
    nq = (qx.T * qx)[0, 0] ** Rational(1, 2)
    check(nx == nq, "‖Qx‖ = ‖x‖")
    return display(L.mat(x, B("x")), f"\\mathbf{{Qx}} = {L.mat(qx)}", f"\\norm{{\\mathbf{{x}}}}_2 = {nx},\\qquad \\norm{{\\mathbf{{Qx}}}}_2 = {nq}") \
        + "\n\nMa trận trực giao giữ nguyên độ dài vector (nó chỉ quay hoặc phản xạ)."


def prop_sym_product() -> str:
    a = M(81, 3, 2, -4, 4)
    g = a.T * a
    check(g == g.T, "A^T A symmetric")
    return display(L.mat(a, B("A")), f"\\mathbf{{A}}\\T\\mathbf{{A}} = {L.mat(g)}") \
        + "\n\nMa trận này đối xứng với mọi $\\mathbf{A}$, vì $(\\mathbf{A}\\T\\mathbf{A})\\T = \\mathbf{A}\\T\\mathbf{A}$ (xem tính chất chuyển vị của tích ở bài 1.5)."


EXAMPLES.update({"prop-diag-commute": prop_diag_commute, "prop-orth-inverse": prop_orth_inverse,
                 "prop-orth-norm": prop_orth_norm, "prop-sym-product": prop_sym_product})


def build():
    return write_examples(LESSON, EXAMPLES)
