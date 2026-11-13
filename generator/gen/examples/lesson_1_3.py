r"""Ví dụ số bài 1.3 Ma trận và ký hiệu."""
from __future__ import annotations

import numpy as np

from ..core import latex as L
from ..core.verify import check, np_close, to_np
from .common import M, display, write_examples

LESSON = "1.3"
B = L.bold


def shape_entry() -> str:
    a = M(11, 3, 4, -9, 9)
    i, j = 1, 2
    check(float(a[i, j]) == to_np(a)[i, j], "entry")
    return (
        display(L.mat(a, B("A")))
        + f"\n\n$\\mathbf{{A}}$ có {a.rows} hàng và {a.cols} cột nên kích thước là ${a.rows}\\times {a.cols}$. "
        f"Phần tử ở hàng {i + 1}, cột {j + 1} là $a_{{{i + 1}{j + 1}}} = {L.num(a[i, j])}$. "
        f"Trong NumPy (Phần 2) cùng phần tử đó có chỉ số bắt đầu từ $0$, tức là `A[{i}, {j}]`."
    )


def transpose_small() -> str:
    a = M(21, 2, 3, -9, 9)
    t = a.T
    np_close(t, to_np(a).T)
    return display(L.mat(a, B("A")), L.mat(t, "\\mathbf{A}\\T")) + (
        f"\n\n$\\mathbf{{A}}$ là ${a.rows}\\times {a.cols}$ còn $\\mathbf{{A}}\\T$ là ${t.rows}\\times {t.cols}$. "
        f"Hàng 1 của $\\mathbf{{A}}$ trở thành cột 1 của $\\mathbf{{A}}\\T$."
    )


def transpose_large() -> str:
    a = M(31, 4, 3, -9, 9)
    t = a.T
    check(t.T == a, "(A^T)^T = A")
    return display(L.mat(a, B("A")), L.mat(t, "\\mathbf{A}\\T"), f"(\\mathbf{{A}}\\T)\\T = {L.mat(t.T)} = \\mathbf{{A}}")


def mistake_index() -> str:
    a = M(41, 3, 3, -9, 9)
    while a[0, 1] == a[1, 0]:
        a = M(41 + 1000 + int(abs(a[0, 0])), 3, 3, -9, 9)
    return (
        display(L.mat(a, B("A")))
        + f"\n\n**Sai:** đọc $a_{{12}}$ là phần tử hàng 2, cột 1 (đảo thứ tự chỉ số), tức là ${L.num(a[1, 0])}$.\n\n"
        + f"**Đúng:** chỉ số đầu là hàng, chỉ số sau là cột, nên $a_{{12}}$ là hàng 1, cột 2: ${L.num(a[0, 1])}$. "
        + f"Còn ${L.num(a[1, 0])}$ là $a_{{21}}$."
    )


EXAMPLES = {"shape-entry": shape_entry, "transpose-small": transpose_small, "transpose-large": transpose_large, "mistake-index": mistake_index}


def prop_entry_t() -> str:
    a = M(51, 2, 3, -9, 9)
    i, j = 2, 1  # (A^T)_{ij} với chỉ số 0-based: hàng 2, cột 1 của A^T
    check(a.T[i, j] == a[j, i], "(A^T)_ij = a_ji")
    return (
        display(L.mat(a, B("A")), L.mat(a.T, "\\mathbf{A}\\T"))
        + f"\n\n$(\\mathbf{{A}}\\T)_{{{i + 1}{j + 1}}} = {L.num(a.T[i, j])}$ và $a_{{{j + 1}{i + 1}}} = {L.num(a[j, i])}$: hai giá trị bằng nhau."
    )


def prop_symmetric() -> str:
    from sympy import Matrix

    s = Matrix([[2, -3, 5], [-3, 0, 7], [5, 7, -1]])
    check(s == s.T, "symmetric")
    return display(L.mat(s, B("S")), L.mat(s.T, "\\mathbf{S}\\T")) + "\n\nHai ma trận trùng nhau nên $\\mathbf{S} = \\mathbf{S}\\T$: $\\mathbf{S}$ là ma trận đối xứng."


EXAMPLES.update({"prop-entry-t": prop_entry_t, "prop-symmetric": prop_symmetric})


def build():
    return write_examples(LESSON, EXAMPLES)
