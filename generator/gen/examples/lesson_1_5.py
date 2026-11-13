r"""Ví dụ số cho bài 1.5 (Nhân ma trận). Mọi con số do SymPy tính và NumPy kiểm lại.

Xuất các partial MDX vào site/src/generated/lesson-1-5/ để bài học import.
Không có số nào trong bài học được gõ tay: đổi seed ở đây là đổi toàn bộ ví dụ.
"""
from __future__ import annotations

import random
from pathlib import Path

from sympy import Matrix

from ..core import latex as L
from ..core.steps import matmul_steps
from ..core.verify import verify_hadamard, verify_matmul
from ..core.nice import nontrivial_matrix

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "site" / "src" / "generated" / "lesson-1-5"


def _m(seed: int, r: int, c: int, lo: int = -4, hi: int = 4) -> Matrix:
    return nontrivial_matrix(random.Random(seed), r, c, lo, hi)


def _display(*lines: str) -> str:
    return "\n\n".join(f"$$\n{ln}\n$$" for ln in lines)


def _steps_md(steps) -> str:
    out = []
    for i, s in enumerate(steps, 1):
        out.append(f"{i}. {s.text}")
        if s.latex:
            out.append(f"\n   $$\n   {s.latex}\n   $$\n")
    return "\n".join(out)


def example_small() -> str:
    a, b = _m(11, 2, 2), _m(12, 2, 2)
    c = a * b
    verify_matmul(a, b, c)
    return (
        _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")))
        + "\n\nTính $\\mathbf{AB}$.\n\n"
        + _steps_md(matmul_steps(a, b, L.bold("A"), L.bold("B")))
    )


def example_large() -> str:
    a, b = _m(21, 2, 3), _m(22, 3, 2)
    c = a * b
    verify_matmul(a, b, c)
    return (
        "Ví dụ này dùng ma trận không vuông: "
        f"$\\mathbf{{A}}$ là ${a.rows}\\times {a.cols}$ và $\\mathbf{{B}}$ là ${b.rows}\\times {b.cols}$.\n\n"
        + _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")))
        + "\n\n"
        + _steps_md(matmul_steps(a, b, L.bold("A"), L.bold("B")))
    )


def example_matvec_columns() -> str:
    a = _m(31, 3, 3)
    x = _m(32, 3, 1)
    y = a * x
    verify_matmul(a, x, y)
    cols = [a[:, j] for j in range(a.cols)]
    comb = " + ".join(f"{L.num_paren(x[j, 0])}{L.vec(cols[j])}" for j in range(a.cols))
    return (
        _display(L.mat(a, L.bold("A")), L.mat(x, L.bold("x")))
        + "\n\nTheo cách nhìn tổ hợp cột, $\\mathbf{Ax}$ là tổ hợp tuyến tính của các cột của $\\mathbf{A}$ "
        "với hệ số là các phần tử của $\\mathbf{x}$:\n\n"
        + _display(f"\\mathbf{{Ax}} = {comb} = {L.vec(y)}")
    )


def example_noncommute() -> str:
    rng = random.Random(0)
    for seed in range(100):
        a, b = _m(41 + seed, 2, 2, -3, 3), _m(141 + seed, 2, 2, -3, 3)
        if a * b != b * a:
            break
    ab, ba = a * b, b * a
    verify_matmul(a, b, ab)
    verify_matmul(b, a, ba)
    return (
        _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")))
        + "\n\nTa có hai kết quả khác nhau:\n\n"
        + _display(f"\\mathbf{{AB}} = {L.mat(ab)}", f"\\mathbf{{BA}} = {L.mat(ba)}")
        + f"\n\nVậy $\\mathbf{{AB}} \\neq \\mathbf{{BA}}$: nhân ma trận **không giao hoán**."
    )


def example_hadamard() -> str:
    a, b = _m(51, 2, 2, -4, 4), _m(52, 2, 2, -4, 4)
    ab, h = a * b, a.multiply_elementwise(b)
    verify_matmul(a, b, ab)
    verify_hadamard(a, b, h)
    return (
        _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")))
        + "\n\nHai phép này cho kết quả khác nhau:\n\n"
        + _display(f"\\mathbf{{AB}} = {L.mat(ab)}", f"\\mathbf{{A}}\\odot\\mathbf{{B}} = {L.mat(h)}")
    )


def example_transpose_product() -> str:
    a, b = _m(61, 2, 3), _m(62, 3, 2)
    ab = a * b
    verify_matmul(a, b, ab)
    lhs = ab.T
    rhs = b.T * a.T
    assert lhs == rhs
    return (
        _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")))
        + "\n\n"
        + _display(f"(\\mathbf{{AB}})\\T = {L.mat(lhs)}", f"\\mathbf{{B}}\\T\\mathbf{{A}}\\T = {L.mat(rhs)}")
        + "\n\nHai vế bằng nhau, đúng như tính chất $(\\mathbf{AB})\\T = \\mathbf{B}\\T\\mathbf{A}\\T$."
    )


def example_distributive() -> str:
    a, b, c = _m(71, 2, 2), _m(72, 2, 2), _m(73, 2, 2)
    lhs = a * (b + c)
    rhs = a * b + a * c
    assert lhs == rhs
    verify_matmul(a, b + c, lhs)
    return (
        _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")), L.mat(c, L.bold("C")))
        + "\n\n"
        + _display(f"\\mathbf{{A}}(\\mathbf{{B}}+\\mathbf{{C}}) = {L.mat(lhs)}",
                   f"\\mathbf{{AB}}+\\mathbf{{AC}} = {L.mat(rhs)}")
    )


def example_associative() -> str:
    a, b, c = _m(91, 2, 2, -3, 3), _m(92, 2, 2, -3, 3), _m(93, 2, 2, -3, 3)
    lhs, rhs = (a * b) * c, a * (b * c)
    assert lhs == rhs
    verify_matmul(a * b, c, lhs)
    verify_matmul(a, b * c, rhs)
    return (
        _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")), L.mat(c, L.bold("C")))
        + "\n\n"
        + _display(f"(\\mathbf{{AB}})\\mathbf{{C}} = {L.mat(lhs)}", f"\\mathbf{{A}}(\\mathbf{{BC}}) = {L.mat(rhs)}")
    )


def example_shape_rule() -> str:
    from ..core.verify import matmul_shape
    cases = [((2, 3), (3, 4)), ((3, 2), (3, 4)), ((1, 4), (4, 1)), ((4, 1), (1, 4))]
    rows = []
    for x, y in cases:
        r = matmul_shape(x, y)
        res = f"${r[0]}\\times {r[1]}$" if r else "không xác định"
        rows.append(f"| ${x[0]}\\times {x[1]}$ | ${y[0]}\\times {y[1]}$ | {res} |")
    return "| Kích thước $\\mathbf{A}$ | Kích thước $\\mathbf{B}$ | Kích thước $\\mathbf{AB}$ |\n| --- | --- | --- |\n" + "\n".join(rows)


def example_common_mistake_order() -> str:
    a, b = _m(81, 2, 2, -3, 3), _m(82, 2, 2, -3, 3)
    ab, ba = a * b, b * a
    return (
        _display(L.mat(a, L.bold("A")), L.mat(b, L.bold("B")))
        + "\n\n**Sai:** tính $\\mathbf{BA}$ khi đề yêu cầu $\\mathbf{AB}$:\n\n"
        + _display(f"\\mathbf{{BA}} = {L.mat(ba)}")
        + "\n\n**Đúng:** $\\mathbf{AB}$ lấy hàng của $\\mathbf{A}$ nhân cột của $\\mathbf{B}$:\n\n"
        + _display(f"\\mathbf{{AB}} = {L.mat(ab)}")
    )


EXAMPLES = {
    "small": example_small,
    "large": example_large,
    "matvec-columns": example_matvec_columns,
    "noncommute": example_noncommute,
    "hadamard": example_hadamard,
    "transpose-product": example_transpose_product,
    "distributive": example_distributive,
    "associative": example_associative,
    "shape-rule": example_shape_rule,
    "mistake-order": example_common_mistake_order,
}


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in EXAMPLES.items():
        p = OUT / f"{name}.mdx"
        p.write_text(fn().strip() + "\n", encoding="utf-8")
        paths.append(p)
    return paths
