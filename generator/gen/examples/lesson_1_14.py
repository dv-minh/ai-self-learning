r"""Ví dụ số bài 1.14 SVD, xấp xỉ hạng thấp, PCA."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational, sqrt

from ..core import latex as L
from ..core.verify import check, np_close, to_np
from ..topics.linalg.l1_14_svd import _build_svd, _pca_data, _verify_svd
from .common import M, display, rng, steps_md, write_examples

LESSON = "1.14"
B, N = L.bold, L.num_paren


def svd() -> str:
    a, u, sg, v, s = _build_svd(rng(11), 2, 2, 9)
    _verify_svd(a, s)
    g = a.T * a
    eig = sorted(g.eigenvals(), reverse=True)
    check([sqrt(e) for e in eig] == [Rational(x) for x in s], "σ = √λ")
    check(u * sg * v.T == a, "A = UΣVᵀ")
    return (display(L.mat(a, B("A")), f"\\mathbf{{A}}\\T\\mathbf{{A}} = {L.mat(g)}")
            + f"\n\nTrị riêng của $\\mathbf{{A}}\\T\\mathbf{{A}}$ là " + ", ".join(f"${L.num(e)}$" for e in eig) + ", nên các giá trị suy biến là "
            + ", ".join(f"$\\sigma_{{{i + 1}}} = {x}$" for i, x in enumerate(s)) + ". Một phân tích SVD đầy đủ:\n\n"
            + display(f"\\mathbf{{A}} = \\mathbf{{U}}\\boldsymbol{{\\Sigma}}\\mathbf{{V}}\\T = {L.mat(u)}{L.mat(sg)}{L.mat(v.T)}"))


def svd_rect() -> str:
    a, u, sg, v, s = _build_svd(rng(21), 2, 3, 9)
    _verify_svd(a, s)
    return (display(L.mat(a, B("A")), f"\\mathbf{{A}}\\mathbf{{A}}\\T = {L.mat(a * a.T)}")
            + f"\n\nMa trận $\\mathbf{{A}}$ cỡ $2\\times 3$ nên dùng $\\mathbf{{A}}\\mathbf{{A}}\\T$ (cỡ $2\\times 2$) cho gọn. Trị riêng: "
            + ", ".join(f"${L.num(e)}$" for e in sorted((a * a.T).eigenvals(), reverse=True)) + ". Các giá trị suy biến khác $0$: " + ", ".join(f"${x}$" for x in s) + ".")


def low_rank() -> str:
    a, u, sg, v, s = _build_svd(rng(31), 3, 3, 9)
    _verify_svd(a, s)
    a1 = s[0] * u[:, 0] * v[:, 0].T
    a2 = a1 + s[1] * u[:, 1] * v[:, 1].T
    err1 = sum(Rational(x) ** 2 for x in s[1:])
    fro = sum(x * x for x in (a - a1))
    check(fro == err1 and a1.rank() == 1 and a2.rank() == 2, "Eckart–Young")
    return (display(L.mat(a, B("A")), "\\sigma = " + ", ".join(str(x) for x in s))
            + "\n\nXấp xỉ hạng $1$ giữ lại giá trị suy biến lớn nhất:\n\n"
            + display(f"\\mathbf{{A}}_1 = \\sigma_1\\mathbf{{u}}_1\\mathbf{{v}}_1\\T = {L.mat(a1)}")
            + f"\n\nSai số bình phương chuẩn Frobenius bằng tổng bình phương các giá trị bị bỏ: ${' + '.join(f'{x}^2' for x in s[1:])} = {L.num(err1)}$."
            + " Giữ thêm một giá trị nữa cho xấp xỉ hạng $2$:\n\n" + display(f"\\mathbf{{A}}_2 = {L.mat(a2)},\\qquad \\norm{{\\mathbf{{A}} - \\mathbf{{A}}_2}}_F^2 = {s[2]}^2 = {s[2] ** 2}"))


def pca() -> str:
    x, mu, xc, cov, lam = _pca_data(rng(41))
    lam = sorted(lam, reverse=True)
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    check(np.allclose(to_np(cov), np.cov(to_np(x).T)), "cov numpy")
    v1 = (cov - lam[0] * Matrix.eye(2)).nullspace()[0]
    return (display(L.mat(x, B("X")))
            + "\n\n**Bước 1: trừ trung bình.** Vector trung bình và dữ liệu đã căn giữa:\n\n"
            + display(L.mat(mu, "\\bar{\\mathbf{x}}"), L.mat(xc, "\\mathbf{X}_c"))
            + "\n\n**Bước 2: ma trận hiệp phương sai** $\\mathbf{C} = \\frac{1}{n-1}\\mathbf{X}_c\\T\\mathbf{X}_c$:\n\n"
            + display(L.mat(cov, "\\mathbf{C}"))
            + "\n\n**Bước 3: trị riêng.** Phương trình đặc trưng:\n\n"
            + display(f"\\lambda^2 - {N(a + c)}\\lambda + {N(a * c - b * b)} = 0 \\;\\Rightarrow\\; \\lambda_1 = {L.num(lam[0])},\\; \\lambda_2 = {L.num(lam[1])}")
            + f"\n\n**Bước 4: thành phần chính.** Vector riêng ứng với $\\lambda_1$ (hướng PC1) là $\\mathbf{{v}}_1 = {L.vec_inline(v1)}$ (chưa chuẩn hóa). "
            + f"PC1 giải thích tỉ lệ ${L.num(lam[0] / (lam[0] + lam[1]))}$ tổng phương sai.")


def lora_count() -> str:
    d, k, r = 8, 6, 2
    full, lora = d * k, d * r + r * k
    return f"Với $\\mathbf{{W}} \\in \\R^{{{d}\\times {k}}}$: huấn luyện cả $\\mathbf{{W}}$ cần ${d}\\cdot {k} = {full}$ tham số. LoRA hạng ${r}$ chỉ huấn luyện $\\mathbf{{B}}$ cỡ ${d}\\times {r}$ và $\\mathbf{{A}}$ cỡ ${r}\\times {k}$: ${d}\\cdot {r} + {r}\\cdot {k} = {lora}$ tham số, giảm ${full - lora}$."


def prop_frobenius() -> str:
    a, u, sg, v, s = _build_svd(rng(51), 2, 3, 9)
    fro = sum(x * x for x in a)
    check(fro == sum(Rational(x) ** 2 for x in s), "‖A‖_F² = Σσ²")
    return display(L.mat(a, B("A")), f"\\norm{{\\mathbf{{A}}}}_F^2 = \\sum_{{ij}} a_{{ij}}^2 = {L.num(fro)},\\qquad \\sum_i \\sigma_i^2 = " + " + ".join(f"{x}^2" for x in s) + f" = {L.num(fro)}")


def prop_rank() -> str:
    a, u, sg, v, s = _build_svd(rng(61), 3, 3, 9)
    # làm suy biến: đặt σ_3 = 0
    s2 = [s[0], s[1], 0]
    sg2 = sg.copy()
    sg2[2, 2] = 0
    b = u * sg2 * v.T
    check(b.rank() == 2, "rank = số σ khác 0")
    np_close(Matrix(s2), np.linalg.svd(to_np(b), compute_uv=False))
    return display(L.mat(b, B("B")), "\\sigma = " + ", ".join(str(x) for x in s2)) + f"\n\nCó {sum(1 for x in s2 if x != 0)} giá trị suy biến khác $0$ nên $\\rank(\\mathbf{{B}}) = {b.rank()}$."


def prop_order() -> str:
    a, u, sg, v, s = _build_svd(rng(71), 2, 2, 12)
    return display(L.mat(a, B("A")), "\\sigma_1 = " + str(s[0]) + " \\ge \\sigma_2 = " + str(s[1]) + " > 0") + "\n\nCác giá trị suy biến luôn không âm và được xếp giảm dần."


def mistake_eig() -> str:
    from ..topics.linalg.l1_12_eigen import _make

    a, _p, _d = _make(rng(81), 2, [2, 3], pbound=3)  # không đối xứng, trị riêng nguyên
    ev = a.eigenvals()
    sv = np.linalg.svd(to_np(a), compute_uv=False)
    check(abs(float(min(ev)) - sv[1]) > 1e-6, "trị riêng khác giá trị suy biến")
    g = a.T * a
    return (display(L.mat(a, B("A")))
            + "\n\n**Sai:** lấy trực tiếp các trị riêng của $\\mathbf{A}$ làm giá trị suy biến: " + ", ".join(f"${L.num(e)}$" for e in sorted(ev, reverse=True)) + ".\n\n"
            + f"**Đúng:** giá trị suy biến là căn bậc hai của trị riêng của $\\mathbf{{A}}\\T\\mathbf{{A}} = {L.mat(g)}$, tức $\\sigma_1 \\approx {sv[0]:.3f}$ và $\\sigma_2 \\approx {sv[1]:.3f}$. "
              "Hai khái niệm chỉ trùng nhau khi $\\mathbf{A}$ đối xứng và xác định dương.")


def mistake_center() -> str:
    x, mu, xc, cov, lam = _pca_data(rng(91))
    wrong = x.T * x / 3
    check(wrong != cov, "không căn giữa khác")
    return (display(L.mat(x, B("X")))
            + "\n\n**Sai:** tính $\\frac{1}{n-1}\\mathbf{X}\\T\\mathbf{X}$ mà quên trừ trung bình:\n\n" + display(L.mat(wrong))
            + "\n\n**Đúng:** phải căn giữa dữ liệu trước (trừ trung bình từng cột):\n\n" + display(L.mat(cov, "\\mathbf{C}")))


EXAMPLES = {"svd": svd, "svd-rect": svd_rect, "low-rank": low_rank, "pca": pca, "lora-count": lora_count,
            "prop-frobenius": prop_frobenius, "prop-rank": prop_rank, "prop-order": prop_order,
            "mistake-eig": mistake_eig, "mistake-center": mistake_center}


def build():
    return write_examples(LESSON, EXAMPLES)
