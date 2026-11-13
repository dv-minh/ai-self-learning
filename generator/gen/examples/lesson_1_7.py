r"""Ví dụ số bài 1.7 Hệ phương trình và khử Gauss."""
from __future__ import annotations

import random

from sympy import Matrix, Rational

from ..core import latex as L
from ..core.gauss import apply_op, eliminate, op_tex, show
from ..core.verify import check, verify_rref, verify_solution, verify_solution_set
from ..topics.linalg.l1_7_gauss import _lu, _nice_from_rref, _solution_set_from_rref
from .common import M, display, rng, steps_md, write_examples

LESSON = "1.7"
B = L.bold


def rowops() -> str:
    m = M(11, 3, 3, -6, 6)
    ops = [("swap", 0, 1), ("scale", 2, Rational(-2)), ("add", 1, 0, Rational(3))]
    out = [display(L.mat(m, B("M")))]
    cur = m
    for op in ops:
        nxt = apply_op(cur, op)
        out.append(f"Áp dụng ${op_tex(op)}$:\n\n" + display(f"{L.mat(cur)} \\;\\longrightarrow\\; {L.mat(nxt)}"))
        cur = nxt
    return "\n\n".join(out)


def unique() -> str:
    a, _u = _lu(rng(21), 3, 3, dmax=3, off=3, lmax=2)
    x = Matrix([2, -1, 3])
    b = a * x
    verify_solution(a, b, x)
    el = eliminate(a.row_join(b), aug=1, mode="rref")
    check(el.matrix[:, 3] == x, "RREF cho nghiệm")
    return (
        display(L.system(a, b))
        + "\n\nMa trận mở rộng và khử Gauss–Jordan:\n\n"
        + steps_md(el.steps)
        + "\n\nĐọc nghiệm ở cột cuối: " + f"${L.mat(x, chr(92) + 'mathbf{x}')}$. "
        + "Thế lại vào hệ: " + f"${L.mat(a * x)} = {L.mat(b)}$."
    )


def infinite() -> str:
    a, rr, piv = _nice_from_rref(rng(31), 3, 4, 2, bound=10, free_max=2)
    x0 = Matrix([1, -2, 2, 1])
    b = a * x0
    el = eliminate(a.row_join(b), aug=1, mode="rref")
    xp, basis = _solution_set_from_rref(el.matrix, 4, el.pivots)
    verify_solution_set(a, b, xp, basis)
    gen = L.mat(xp, "\\mathbf{x}") + "".join(f" + t_{{{k + 1}}}{L.mat(v)}" for k, v in enumerate(basis))
    free = [j for j in range(4) if j not in el.pivots]
    return (
        display(L.system(a, b))
        + "\n\n" + steps_md(el.steps)
        + f"\n\nCột pivot: " + ", ".join(str(c + 1) for c in el.pivots) + ". Ẩn tự do: " + ", ".join(f"$x_{{{j + 1}}}$" for j in free)
        + ". Nghiệm tổng quát:\n\n" + display(gen)
    )


def inconsistent() -> str:
    a, rr, piv = _nice_from_rref(rng(41), 3, 3, 2, bound=8, free_max=2)
    b = a * Matrix([1, 2, -1]) + Matrix([0, 0, 3])
    aug = a.row_join(b)
    from ..core.verify import verify_rank

    check(a.rank() < aug.rank(), "vô nghiệm")
    verify_rank(a, a.rank())
    verify_rank(aug, aug.rank())
    el = eliminate(aug, aug=1, mode="ref")
    return display(L.system(a, b)) + "\n\n" + steps_md(el.steps) + (
        f"\n\nSau khử xuất hiện một hàng có dạng $0 = c$ với $c \\neq 0$ (hạng của ma trận mở rộng là {aug.rank()}, "
        f"lớn hơn hạng của $\\mathbf{{A}}$ là {a.rank()}), nên hệ **vô nghiệm**."
    )


def ref_vs_rref() -> str:
    a, rr, piv = _nice_from_rref(rng(51), 3, 4, 3, bound=10, free_max=2)
    ref = eliminate(a, mode="ref")
    rref = eliminate(a, mode="rref")
    verify_rref(a, rref.matrix, rref.pivots)
    return (
        display(L.mat(a, B("A")))
        + "\n\nDạng bậc thang (chỉ khử xuôi, không chuẩn hóa pivot):\n\n" + display(L.mat(ref.matrix))
        + "\n\nDạng bậc thang rút gọn (pivot bằng $1$, cột pivot chỉ có một số khác $0$):\n\n" + display(L.mat(rref.matrix))
    )


def mistake_aug() -> str:
    a, _u = _lu(rng(61), 2, 2, dmax=3, off=3, lmax=2)
    b = Matrix([3, -2])
    k = Rational(-2)
    wrong = a.copy()
    wrong[1, :] = k * a[1, :]
    wr = wrong.row_join(b)
    right = a.row_join(b)
    right[1, :] = k * right[1, :]
    return (
        display(L.aug(a.row_join(b), 2))
        + f"\n\n**Sai:** thực hiện $R_2 \\leftarrow {L.num(k)}R_2$ nhưng quên nhân vế phải:\n\n"
        + display(L.aug(wr, 2))
        + "\n\n**Đúng:** một phép biến đổi hàng tác động lên cả hàng của ma trận mở rộng, gồm cả cột vế phải:\n\n"
        + display(L.aug(right, 2))
    )


EXAMPLES = {"rowops": rowops, "unique": unique, "infinite": infinite, "inconsistent": inconsistent, "ref-vs-rref": ref_vs_rref, "mistake-aug": mistake_aug}


def prop_same_solution() -> str:
    a, _u = _lu(rng(71), 3, 3, dmax=3, off=3, lmax=2)
    x = Matrix([1, -2, 2])
    b = a * x
    aug = a.row_join(b)
    after = apply_op(apply_op(aug, ("add", 1, 0, Rational(2))), ("swap", 0, 2))
    a2, b2 = after[:, :3], after[:, 3]
    check(a2 * x == b2, "nghiệm giữ nguyên")
    return (
        f"Hệ gốc có nghiệm $\\mathbf{{x}} = {L.vec_inline(x)}$. Áp dụng $R_2 \\leftarrow R_2 + 2R_1$ rồi $R_1 \\leftrightarrow R_3$:\n\n"
        + display(f"{L.aug(aug, 3)} \\;\\longrightarrow\\; {L.aug(after, 3)}")
        + f"\n\nThế $\\mathbf{{x}}$ vào hệ mới: ${L.mat(a2 * x)} = {L.mat(b2)}$. Nghiệm không đổi."
    )


EXAMPLES.update({"prop-same-solution": prop_same_solution})


def build():
    return write_examples(LESSON, EXAMPLES)
