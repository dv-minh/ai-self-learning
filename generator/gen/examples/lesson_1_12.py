r"""Ví dụ số bài 1.12 Trị riêng, vector riêng."""
from __future__ import annotations

from sympy import Matrix, Rational, diag, eye, factor, latex, zeros

from ..core import latex as L
from ..core.gauss import eliminate
from ..core.verify import check, np_close, to_np
from ..topics.linalg.l1_12_eigen import LAM, _charpoly_steps, _make, _verify_eigs
from ..topics.linalg.l1_9_inverse import _unimodular
from .common import M, display, rng, steps_md, write_examples

import numpy as np

LESSON = "1.12"
B, N = L.bold, L.num_paren


def eig2() -> str:
    a, p, d = _make(rng(11), 2, [-1, 3])
    roots = sorted(Rational(x) for x in (-1, 3))
    _verify_eigs(a, roots)
    out = [display(L.mat(a, B("A")))] + [steps_md(_charpoly_steps(a, roots))]
    for j, lam in enumerate([-1, 3]):
        m = a - lam * eye(2)
        v = p[:, j]
        check(a * v == lam * v, "Av = λv")
        out.append(f"Với $\\lambda = {lam}$: giải $(\\mathbf{{A}} - \\lambda\\mathbf{{I}})\\mathbf{{v}} = \\mathbf{{0}}$ với $\\mathbf{{A}} - \\lambda\\mathbf{{I}} = {L.mat(m)}$. "
                   f"Một nghiệm khác $\\mathbf{{0}}$ là $\\mathbf{{v}} = {L.vec_inline(v)}$, và kiểm tra:\n\n"
                   + display(f"\\mathbf{{A}}\\mathbf{{v}} = {L.mat(a * v)} = {N(lam)}{L.mat(v)}"))
    return "\n\n".join(out)


def eig3() -> str:
    a, p, d = _make(rng(21), 3, [-2, 1, 2])
    roots = [Rational(-2), Rational(1), Rational(2)]
    _verify_eigs(a, roots)
    j = 1
    m = a - 1 * eye(3)
    el = eliminate(m, mode="rref")
    v = p[:, j]
    check(a * v == v, "Av = v")
    return (display(L.mat(a, B("A"))) + "\n\n" + steps_md(_charpoly_steps(a, roots))
            + "\n\nVector riêng ứng với $\\lambda = 1$: khử Gauss–Jordan $\\mathbf{A} - \\mathbf{I}$:\n\n" + steps_md(el.steps)
            + f"\n\nMột nghiệm khác $\\mathbf{{0}}$ là $\\mathbf{{v}} = {L.vec_inline(v)}$.")


def diagonalize() -> str:
    a, p, d = _make(rng(31), 2, [-2, 3])
    inv = p.inv()
    check(inv * a * p == d, "P^-1 A P = D")
    return display(L.mat(a, B("A")), L.mat(p, B("P")) + ",\\quad " + L.mat(inv, "\\mathbf{P}^{-1}"),
                   f"\\mathbf{{A}}\\mathbf{{P}} = {L.mat(a * p)},\\qquad \\mathbf{{P}}^{{-1}}\\mathbf{{A}}\\mathbf{{P}} = {L.mat(d, chr(92) + 'mathbf{D}')}") + \
        "\n\nCác cột của $\\mathbf{P}$ là vector riêng; $\\mathbf{D}$ chứa các trị riêng đúng theo thứ tự các cột đó."


def power() -> str:
    a, p, d = _make(rng(41), 2, [-1, 2], pbound=3)
    k = 5
    inv = p.inv()
    dk = diag(*[d[i, i] ** k for i in range(2)])
    res = p * dk * inv
    check(res == a ** k, "A^k")
    np_close(res, np.linalg.matrix_power(to_np(a), k))
    return display(L.mat(a, B("A")), f"\\mathbf{{D}}^{{{k}}} = {L.mat(dk)}", f"\\mathbf{{A}}^{{{k}}} = \\mathbf{{P}}\\mathbf{{D}}^{{{k}}}\\mathbf{{P}}^{{-1}} = {L.mat(p)}{L.mat(dk)}{L.mat(inv)} = {L.mat(res)}") + \
        f"\n\nChỉ cần lũy thừa ${k}$ hai số trên đường chéo, thay vì nhân $\\mathbf{{A}}$ với chính nó ${k - 1}$ lần."


def defective() -> str:
    for seed in range(100):
        p = _unimodular(rng(51 + seed), 2, 4, 3)
        a = p * Matrix([[2, 1], [0, 2]]) * p.inv()
        if max(abs(x) for x in a) <= 12 and a != Matrix([[2, 1], [0, 2]]):
            break
    check(not a.is_diagonalizable(), "defective")
    m = a - 2 * eye(2)
    return (display(L.mat(a, B("A")), f"\\det(\\lambda\\mathbf{{I}} - \\mathbf{{A}}) = {latex(factor((LAM * eye(2) - a).det().expand()))}")
            + f"\n\nTrị riêng duy nhất là $\\lambda = 2$ (bội đại số $2$). Tính $\\mathbf{{A}} - 2\\mathbf{{I}} = {L.mat(m)}$, hạng bằng ${m.rank()}$, nên chỉ có ${2 - m.rank()}$ vector riêng độc lập (bội hình học $1$). "
              "Không đủ hai vector riêng độc lập để lập $\\mathbf{P}$ khả nghịch: $\\mathbf{A}$ **không chéo hóa được**.")


def prop_trace_det() -> str:
    a, p, d = _make(rng(61), 3, [-1, 2, 3])
    vals = [d[i, i] for i in range(3)]
    check(sum(vals) == a.trace() and vals[0] * vals[1] * vals[2] == a.det(), "tổng, tích")
    return display(L.mat(a, B("A")), f"\\tr(\\mathbf{{A}}) = {L.num(a.trace())} = " + " + ".join(N(x) for x in vals),
                   f"\\det(\\mathbf{{A}}) = {L.num(a.det())} = " + " \\cdot ".join(N(x) for x in vals)) + \
        "\n\nTổng các trị riêng bằng vết, tích các trị riêng bằng định thức."


def prop_triangular() -> str:
    a = Matrix([[3, 5, -2], [0, -1, 4], [0, 0, 2]])
    return display(L.mat(a, B("A"))) + "\n\nMa trận tam giác: trị riêng là các phần tử trên đường chéo, tức " + ", ".join(f"${a[i, i]}$" for i in range(3)) + "."


def prop_scale() -> str:
    a, p, d = _make(rng(71), 2, [1, 3])
    v = p[:, 1]
    k = -2
    check(a * (k * v) == 3 * (k * v), "bội vô hướng")
    return f"Với $\\mathbf{{v}} = {L.vec_inline(v)}$ là vector riêng ứng với $\\lambda = 3$:\n\n" + display(
        f"\\mathbf{{A}}(k\\mathbf{{v}}) = {L.mat(a * (k * v))} = 3\\cdot{L.mat(k * v)},\\qquad k = {k}") + "\n\nMọi bội vô hướng khác $0$ của vector riêng vẫn là vector riêng của cùng trị riêng."


def prop_power_eig() -> str:
    a, p, d = _make(rng(81), 2, [2, -1])
    v = p[:, 0]
    k = 4
    check((a ** k) * v == 2 ** k * v, "A^k v = λ^k v")
    return f"Với $\\mathbf{{v}} = {L.vec_inline(v)}$, $\\lambda = 2$ và $k = {k}$:\n\n" + display(
        f"\\mathbf{{A}}^{{{k}}}\\mathbf{{v}} = {L.mat(a ** k * v)},\\qquad 2^{{{k}}}\\mathbf{{v}} = {L.mat(2 ** k * v)}")


def mistake_zero() -> str:
    a, p, d = _make(rng(91), 2, [1, 2])
    z = zeros(2, 1)
    return display(L.mat(a, B("A")), f"\\mathbf{{A}}\\mathbf{{0}} = {L.mat(a * z)}") + \
        "\n\n**Sai:** coi $\\mathbf{0}$ là vector riêng vì $\\mathbf{A}\\mathbf{0} = \\lambda\\mathbf{0}$ đúng với mọi $\\lambda$. " \
        "**Đúng:** theo định nghĩa, vector riêng phải **khác $\\mathbf{0}$**; nếu không thì mọi số đều là trị riêng và khái niệm mất ý nghĩa."


def mistake_order() -> str:
    a, p, d = _make(rng(95), 2, [-2, 3])
    swapped = p[:, [1, 0]]
    check(swapped * d * swapped.inv() != a and swapped * diag(d[1, 1], d[0, 0]) * swapped.inv() == a, "thứ tự D theo P")
    return (display(L.mat(a, B("A")), L.mat(swapped, B("P")))
            + f"\n\n$\\mathbf{{P}}$ có cột 1 là vector riêng của $\\lambda = {d[1, 1]}$ và cột 2 là của $\\lambda = {d[0, 0]}$.\n\n"
            + f"**Sai:** viết $\\mathbf{{D}} = \\diag({d[0, 0]}, {d[1, 1]})$ theo thứ tự trị riêng đã tìm, rồi kiểm tra $\\mathbf{{P}}\\mathbf{{D}}\\mathbf{{P}}^{{-1}}$ thấy khác $\\mathbf{{A}}$:\n\n"
            + display(f"\\mathbf{{P}}\\mathbf{{D}}\\mathbf{{P}}^{{-1}} = {L.mat(swapped * d * swapped.inv())} \\neq \\mathbf{{A}}")
            + "\n\n**Đúng:** thứ tự trên đường chéo của $\\mathbf{D}$ phải khớp thứ tự các cột của $\\mathbf{P}$:\n\n"
            + display(f"\\mathbf{{D}} = \\diag({d[1, 1]}, {d[0, 0]}),\\qquad \\mathbf{{P}}\\mathbf{{D}}\\mathbf{{P}}^{{-1}} = {L.mat(swapped * diag(d[1, 1], d[0, 0]) * swapped.inv())} = \\mathbf{{A}}"))


EXAMPLES = {"eig2": eig2, "eig3": eig3, "diagonalize": diagonalize, "power": power, "defective": defective,
            "prop-trace-det": prop_trace_det, "prop-triangular": prop_triangular, "prop-scale": prop_scale, "prop-power-eig": prop_power_eig,
            "mistake-zero": mistake_zero, "mistake-order": mistake_order}


def build():
    return write_examples(LESSON, EXAMPLES)
