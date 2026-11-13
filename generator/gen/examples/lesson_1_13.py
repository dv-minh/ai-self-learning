r"""Ví dụ số bài 1.13 Ma trận đối xứng và xác định dương."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational, diag, eye

from ..core import latex as L
from ..core.verify import check, np_close, to_np
from ..topics.linalg.l1_12_eigen import _charpoly_steps, _verify_eigs
from ..topics.linalg.nice_sym import pick_sym
from .common import M, V, display, rng, steps_md, write_examples

LESSON = "1.13"
B, N = L.bold, L.num_paren


def _sym(seed: int, n: int, r: int, want=None):
    return pick_sym(rng(seed), n, r, want)


def sym_eig() -> str:
    a, eig = _sym(11, 2, 8)
    roots = sorted(Rational(x) for x in eig)
    _verify_eigs(a, roots)
    ns = [(a - r * eye(2)).nullspace()[0] for r in roots]
    d = (ns[0].T * ns[1])[0, 0]
    check(d == 0, "vector riêng trực giao")
    return (display(L.mat(a, B("A"))) + "\n\n" + steps_md(_charpoly_steps(a, roots))
            + f"\n\nVector riêng ứng với $\\lambda_1 = {L.num(roots[0])}$ và $\\lambda_2 = {L.num(roots[1])}$ lần lượt là $\\mathbf{{v}}_1 = {L.vec_inline(ns[0])}$ và $\\mathbf{{v}}_2 = {L.vec_inline(ns[1])}$. "
              + "Tích vô hướng của chúng:\n\n" + display(f"\\mathbf{{v}}_1\\T\\mathbf{{v}}_2 = {L.num(d)}") + "\n\nHai vector riêng của hai trị riêng khác nhau của ma trận đối xứng **trực giao**.")


def spectral() -> str:
    a, eig = _sym(21, 2, 8, lambda e: len(set(e)) == 2 and all(x > 0 for x in e))
    roots = sorted(Rational(x) for x in eig)
    vs = [(a - r * eye(2)).nullspace()[0] for r in roots]
    # chuẩn hóa: dùng tổng hạng λ u uᵀ với u = v / (v^T v) (không cần căn): A = Σ λ_i v_i v_iᵀ / (v_iᵀ v_i)
    terms = [r * (v * v.T) / (v.T * v)[0, 0] for r, v in zip(roots, vs)]
    check(sum(terms, Matrix.zeros(2, 2)) == a, "phân rã phổ")
    return (display(L.mat(a, B("A")))
            + "\n\nVới ma trận đối xứng, $\\mathbf{A} = \\sum_i \\lambda_i\\,\\dfrac{\\mathbf{v}_i\\mathbf{v}_i\\T}{\\mathbf{v}_i\\T\\mathbf{v}_i}$ (tổng các phép chiếu lên hướng riêng, nhân trị riêng):\n\n"
            + display(*[f"\\lambda_{{{i + 1}}}\\dfrac{{\\mathbf{{v}}_{{{i + 1}}}\\mathbf{{v}}_{{{i + 1}}}\\T}}{{\\mathbf{{v}}_{{{i + 1}}}\\T\\mathbf{{v}}_{{{i + 1}}}}} = {L.mat(t)}" for i, t in enumerate(terms)]
                       + [f"\\mathbf{{A}} = {L.mat(a)}"]))


def quad() -> str:
    a, _e = _sym(31, 2, 5)
    x = V(32, 2, -3, 3)
    v = (x.T * a * x)[0, 0]
    check(np.isclose(float(v), float((to_np(x).T @ to_np(a) @ to_np(x))[0, 0])), "numpy")
    ax = a * x
    return display(L.mat(a, B("A")), L.mat(x, B("x")), f"\\mathbf{{A}}\\mathbf{{x}} = {L.mat(ax)}",
                   f"q(\\mathbf{{x}}) = \\mathbf{{x}}\\T\\mathbf{{A}}\\mathbf{{x}} = " + " + ".join(f"{N(p)}\\cdot {N(q)}" for p, q in zip(x, ax)) + f" = {L.num(v)}")


def posdef_pd() -> str:
    a, eig = _sym(41, 3, 3, lambda e: all(x > 0 for x in e))
    minors = [a[:k, :k].det() for k in (1, 2, 3)]
    check(all(m > 0 for m in minors), "Sylvester")
    return (display(L.mat(a, B("A")))
            + "\n\nCác định thức con chính (góc trên trái):\n\n"
            + display(", \\quad ".join(f"\\Delta_{{{k + 1}}} = {L.num(m)}" for k, m in enumerate(minors)))
            + f"\n\nTất cả đều dương nên $\\mathbf{{A}}$ **xác định dương**. Kiểm tra bằng trị riêng: " + ", ".join(L.num(Rational(x)) for x in sorted(eig)) + " (đều dương).")


def posdef_indef() -> str:
    a, eig = _sym(51, 2, 8, lambda e: min(e) < 0 < max(e))
    x1 = Matrix([1, 0])
    lam_pos = max(eig)
    v_pos = (a - lam_pos * eye(2)).nullspace()[0]
    lam_neg = min(eig)
    v_neg = (a - lam_neg * eye(2)).nullspace()[0]
    qp, qn = (v_pos.T * a * v_pos)[0, 0], (v_neg.T * a * v_neg)[0, 0]
    check(qp > 0 > qn, "indefinite")
    return (display(L.mat(a, B("A")))
            + f"\n\nTrị riêng: ${eig[0]}$ và ${eig[1]}$, trái dấu. Dạng toàn phương nhận cả giá trị dương lẫn âm: với $\\mathbf{{v}}_1 = {L.vec_inline(v_pos)}$ ta có "
              f"$q(\\mathbf{{v}}_1) = {L.num(qp)} > 0$, còn với $\\mathbf{{v}}_2 = {L.vec_inline(v_neg)}$ ta có $q(\\mathbf{{v}}_2) = {L.num(qn)} < 0$. Vậy $\\mathbf{{A}}$ **không xác định**.")


def prop_gram() -> str:
    for seed in range(2000):  # chọn X để trị riêng của Gram là số hữu tỉ, tính tay được
        x = M(61 + seed, 3, 2, -3, 3)
        g = x.T * x
        eig = g.eigenvals()
        if all(e.is_rational for e in eig) and len(eig) == 2:
            break
    check(all(e >= 0 for e in eig), "Gram PSD")
    return display(L.mat(x, B("X")), f"\\mathbf{{X}}\\T\\mathbf{{X}} = {L.mat(g)}") + \
        "\n\nTrị riêng: " + ", ".join(f"${L.num(e)}$" for e in sorted(eig, key=float)) + " (không âm). Ma trận Gram $\\mathbf{X}\\T\\mathbf{X}$ luôn **bán xác định dương**, vì $\\mathbf{v}\\T\\mathbf{X}\\T\\mathbf{X}\\mathbf{v} = \\norm{\\mathbf{X}\\mathbf{v}}_2^2 \\ge 0$."


def prop_real() -> str:
    a, eig = _sym(71, 3, 3)
    np_close(Matrix(sorted(eig)), np.sort(np.linalg.eigvalsh(to_np(a))))
    return display(L.mat(a, B("A"))) + "\n\nTrị riêng của ma trận đối xứng thực luôn là số thực. Ở đây: " + ", ".join(f"${x}$" for x in sorted(eig)) + "."


def prop_diag_pd() -> str:
    d = diag(2, 5, 1)
    return display(L.mat(d, B("D"))) + "\n\nMa trận đường chéo có trị riêng là các phần tử đường chéo, nên nó xác định dương khi và chỉ khi mọi phần tử đường chéo dương."


def mistake_diag_positive() -> str:
    for seed in range(300):
        a, eig = _sym(81 + seed, 2, 8, lambda e: min(e) < 0 < max(e))
        if a[0, 0] > 0 and a[1, 1] > 0:
            break
    return (display(L.mat(a, B("A")))
            + f"\n\n**Sai:** thấy mọi phần tử đường chéo dương ({L.num(a[0, 0])} và {L.num(a[1, 1])}) rồi kết luận $\\mathbf{{A}}$ xác định dương.\n\n"
            + f"**Đúng:** phải xét định thức con chính hoặc trị riêng. Ở đây $\\det(\\mathbf{{A}}) = {L.num(a.det())} < 0$ nên hai trị riêng trái dấu (${eig[0]}$ và ${eig[1]}$): $\\mathbf{{A}}$ không xác định.")


def mistake_nonsym() -> str:
    a = Matrix([[0, -1], [1, 0]])
    return (display(L.mat(a, B("A")), f"\\det(\\lambda\\mathbf{{I}} - \\mathbf{{A}}) = \\lambda^2 + 1")
            + "\n\n**Sai:** cho rằng mọi ma trận thực đều có trị riêng thực. Ma trận quay này không đối xứng và phương trình $\\lambda^2 + 1 = 0$ vô nghiệm thực. "
              "Tính chất \"trị riêng thực\" đúng với ma trận **đối xứng**, chứ không đúng với ma trận bất kỳ.")


EXAMPLES = {"sym-eig": sym_eig, "spectral": spectral, "quad": quad, "posdef-pd": posdef_pd, "posdef-indef": posdef_indef,
            "prop-gram": prop_gram, "prop-real": prop_real, "prop-diag-pd": prop_diag_pd,
            "mistake-diag-positive": mistake_diag_positive, "mistake-nonsym": mistake_nonsym}


def build():
    return write_examples(LESSON, EXAMPLES)
