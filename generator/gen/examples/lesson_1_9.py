r"""Ví dụ số bài 1.9 Ma trận nghịch đảo."""
from __future__ import annotations

from sympy import Matrix, eye

from ..core import latex as L
from ..core.gauss import eliminate
from ..core.verify import check, verify_inverse, verify_solution
from ..topics.linalg.l1_9_inverse import _invertible, _singular, _unimodular
from .common import M, display, rng, steps_md, write_examples

LESSON = "1.9"
B, N = L.bold, L.num_paren


def _swap_expr(a: Matrix) -> str:
    return L.expr_mat([[L.num(a[1, 1]), L.num(-a[0, 1])], [L.num(-a[1, 0]), L.num(a[0, 0])]])


def inv2() -> str:
    a = _invertible(rng(11), 2, -6, 6)
    d = a.det()
    inv = a.inv()
    verify_inverse(a, inv)
    return display(
        L.mat(a, B("A")),
        f"\\det(\\mathbf{{A}}) = {N(a[0, 0])}\\cdot {N(a[1, 1])} - {N(a[0, 1])}\\cdot {N(a[1, 0])} = {L.num(d)}",
        f"\\mathbf{{A}}^{{-1}} = \\frac{{1}}{{{N(d)}}}{_swap_expr(a)} = {L.mat(inv)}",
        f"\\mathbf{{A}}\\mathbf{{A}}^{{-1}} = {L.mat(a)}{L.mat(inv)} = {L.mat(a * inv)}",
    )


def gauss_jordan() -> str:
    a = _unimodular(rng(21), 3, 8, 4)
    inv = a.inv()
    el = eliminate(a.row_join(eye(3)), aug=3, mode="rref")
    check(el.matrix[:, 3:] == inv, "Gauss–Jordan")
    verify_inverse(a, inv)
    return display(L.mat(a, B("A"))) + "\n\n" + steps_md(el.steps) + "\n\nVế phải là nghịch đảo:\n\n" + display(L.mat(inv, "\\mathbf{A}^{-1}"))


def singular() -> str:
    a = _singular(rng(31), 3, -5, 5)
    el = eliminate(a, mode="ref")
    check(a.det() == 0, "singular")
    return (
        display(L.mat(a, B("A")))
        + "\n\nKhử Gauss:\n\n" + steps_md(el.steps)
        + f"\n\nXuất hiện một hàng toàn $0$ (hạng của $\\mathbf{{A}}$ là {a.rank()}, nhỏ hơn {a.rows}), và $\\det(\\mathbf{{A}}) = {L.num(a.det())}$. "
        "Ma trận này **không khả nghịch**."
    )


def solve() -> str:
    a = _invertible(rng(41), 2, -5, 5)
    x = Matrix([3, -2])
    b = a * x
    inv = a.inv()
    verify_solution(a, b, x)
    check(inv * b == x, "x = A^-1 b")
    return display(L.system(a, b), L.mat(inv, "\\mathbf{A}^{-1}"), f"\\mathbf{{x}} = \\mathbf{{A}}^{{-1}}\\mathbf{{b}} = {L.mat(inv)}{L.mat(b)} = {L.mat(x)}")


def product_inverse() -> str:
    a, b = _unimodular(rng(51), 2, 6, 3), _unimodular(rng(52), 2, 6, 3)
    c = a * b
    right = b.inv() * a.inv()
    check(c.inv() == right, "(AB)^-1")
    return display(
        L.mat(a, B("A")), L.mat(b, B("B")),
        f"\\mathbf{{AB}} = {L.mat(c)},\\quad (\\mathbf{{AB}})^{{-1}} = {L.mat(c.inv())}",
        f"\\mathbf{{B}}^{{-1}}\\mathbf{{A}}^{{-1}} = {L.mat(b.inv())}{L.mat(a.inv())} = {L.mat(right)}",
    ) + "\n\nHai kết quả trùng nhau: $(\\mathbf{AB})^{-1} = \\mathbf{B}^{-1}\\mathbf{A}^{-1}$."


def mistake_order() -> str:
    for seed in range(100):
        a, b = _unimodular(rng(61 + seed), 2, 6, 3), _unimodular(rng(161 + seed), 2, 6, 3)
        if a.inv() * b.inv() != b.inv() * a.inv():
            break
    c = a * b
    wrong = a.inv() * b.inv()
    return (
        display(L.mat(a, B("A")), L.mat(b, B("B")))
        + "\n\n**Sai:** giữ nguyên thứ tự: $(\\mathbf{AB})^{-1} = \\mathbf{A}^{-1}\\mathbf{B}^{-1}$:\n\n"
        + display(f"\\mathbf{{A}}^{{-1}}\\mathbf{{B}}^{{-1}} = {L.mat(wrong)}", f"\\mathbf{{AB}}\\cdot\\mathbf{{A}}^{{-1}}\\mathbf{{B}}^{{-1}} = {L.mat(c * wrong)} \\neq \\mathbf{{I}}")
        + "\n\n**Đúng:** đảo thứ tự: $(\\mathbf{AB})^{-1} = \\mathbf{B}^{-1}\\mathbf{A}^{-1}$:\n\n"
        + display(f"\\mathbf{{AB}}\\cdot\\mathbf{{B}}^{{-1}}\\mathbf{{A}}^{{-1}} = {L.mat(c * b.inv() * a.inv())} = \\mathbf{{I}}")
    )


EXAMPLES = {"inv2": inv2, "gauss-jordan": gauss_jordan, "singular": singular, "solve": solve,
            "product-inverse": product_inverse, "mistake-order": mistake_order}


def prop_inverse_inverse() -> str:
    a = _invertible(rng(71), 2, -6, 6)
    check(a.inv().inv() == a, "(A^-1)^-1")
    return display(L.mat(a, B("A")), f"\\mathbf{{A}}^{{-1}} = {L.mat(a.inv())},\\qquad (\\mathbf{{A}}^{{-1}})^{{-1}} = {L.mat(a.inv().inv())} = \\mathbf{{A}}")


def prop_transpose_inverse() -> str:
    a = _invertible(rng(72), 3, -3, 3)
    check(a.T.inv() == a.inv().T, "(A^T)^-1 = (A^-1)^T")
    return display(L.mat(a, B("A")), f"(\\mathbf{{A}}\\T)^{{-1}} = {L.mat(a.T.inv())},\\qquad (\\mathbf{{A}}^{{-1}})\\T = {L.mat(a.inv().T)}")


def prop_det_inverse() -> str:
    a = _invertible(rng(73), 3, -3, 3)
    check(a.inv().det() == 1 / a.det(), "det A^-1")
    return display(L.mat(a, B("A")), f"\\det(\\mathbf{{A}}) = {L.num(a.det())},\\qquad \\det(\\mathbf{{A}}^{{-1}}) = {L.num(a.inv().det())} = \\frac{{1}}{{\\det(\\mathbf{{A}})}}")


EXAMPLES.update({"prop-inverse-inverse": prop_inverse_inverse, "prop-transpose-inverse": prop_transpose_inverse, "prop-det-inverse": prop_det_inverse})


def build():
    return write_examples(LESSON, EXAMPLES)
