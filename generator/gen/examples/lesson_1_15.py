r"""Ví dụ số bài 1.15 Ma trận trong ML."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational, sqrt

from ..core import latex as L
from ..core.verify import check, np_close, to_np
from ..topics.linalg.l1_15_ml import _fit
from .common import M, display, rng, steps_md, write_examples

LESSON = "1.15"
B, N = L.bold, L.num_paren


def data_matrix() -> str:
    x = M(11, 4, 3, 0, 9)
    w = Matrix([2, -1, 1])
    b0 = 3
    pred = x * w + Matrix.ones(4, 1) * b0
    np_close(pred, to_np(x) @ to_np(w) + b0)
    return display(L.mat(x, B("X")), L.mat(w, B("w")), f"\\hat{{\\mathbf{{y}}}} = \\mathbf{{X}}\\mathbf{{w}} + b = {L.mat(x * w)} + {b0} = {L.mat(pred)}") + \
        f"\n\n$\\mathbf{{X}}$ cỡ ${x.rows}\\times {x.cols}$ ({x.rows} mẫu, {x.cols} đặc trưng) nhân $\\mathbf{{w}}$ cỡ ${w.rows}\\times 1$ cho vector dự đoán {x.rows} phần tử. Một phép nhân ma trận–vector tính dự đoán cho **cả batch**."


def normal_eq() -> str:
    x, y, g, xty, w = _fit(rng(21), 3)
    pts = ", ".join(f"({int(x[i, 1])}, {int(y[i])})" for i in range(3))
    check(g * w == xty, "phương trình chuẩn")
    resid = y - x * w
    check(x.T * resid == Matrix.zeros(2, 1), "phần dư vuông góc")
    return (f"Khớp đường thẳng $y = w_0 + w_1 x$ với các điểm {pts}.\n\n"
            + display(L.mat(x, B("X")) + ",\\quad " + L.mat(y, B("y")), f"\\mathbf{{X}}\\T\\mathbf{{X}} = {L.mat(g)},\\qquad \\mathbf{{X}}\\T\\mathbf{{y}} = {L.mat(xty)}",
                      f"\\mathbf{{w}} = (\\mathbf{{X}}\\T\\mathbf{{X}})^{{-1}}\\mathbf{{X}}\\T\\mathbf{{y}} = {L.mat(g.inv())}{L.mat(xty)} = {L.mat(w)}")
            + f"\n\nPhần dư $\\mathbf{{y}} - \\mathbf{{X}}\\mathbf{{w}} = {L.vec_inline(resid)}$ và $\\mathbf{{X}}\\T(\\mathbf{{y}} - \\mathbf{{X}}\\mathbf{{w}}) = {L.vec_inline(x.T * resid)}$: "
              "phần dư vuông góc với mọi cột của $\\mathbf{X}$, đúng như phương trình chuẩn đòi hỏi.")


def attention_scores() -> str:
    from ..topics.linalg.l1_15_ml import nontrivial_matrix

    r = rng(31)
    dk = 4
    q = nontrivial_matrix(r, 2, dk, -2, 2)
    k = nontrivial_matrix(r, 3, dk, -2, 2)
    s = q * k.T
    scaled = s / sqrt(dk)
    np_close(scaled, to_np(q) @ to_np(k).T / np.sqrt(dk))
    return display(L.mat(q, B("Q")), L.mat(k, B("K")), f"\\mathbf{{Q}}\\mathbf{{K}}\\T = {L.mat(s)}",
                   f"\\frac{{\\mathbf{{Q}}\\mathbf{{K}}\\T}}{{\\sqrt{{d_k}}}} = \\frac{{1}}{{{sqrt(dk)}}}{L.mat(s)} = {L.mat(scaled)}") + \
        f"\n\n$\\mathbf{{Q}}$ có {q.rows} truy vấn và $\\mathbf{{K}}$ có {k.rows} khóa (cùng $d_k = {dk}$), nên ma trận điểm cỡ ${q.rows}\\times {k.rows}$: mỗi hàng chứa điểm của một truy vấn với mọi khóa. Chia cho $\\sqrt{{d_k}}$ giữ cho điểm không quá lớn trước khi qua softmax."


def parts_sum(layers: list[int]) -> str:
    return " + ".join(str(layers[i] * layers[i + 1] + layers[i + 1]) for i in range(len(layers) - 1))


def param_count() -> str:
    layers = [4, 6, 3]
    parts = [f"{layers[i]}\\cdot {layers[i + 1]} + {layers[i + 1]} = {layers[i] * layers[i + 1] + layers[i + 1]}" for i in range(2)]
    total = sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(2))
    return "Mạng ${} \\to {} \\to {}$:\n\n".format(*layers) + display(*parts) + f"\n\nTổng số tham số học được là ${parts_sum(layers)} = {total}$."


def prop_gram_sym() -> str:
    x = M(41, 4, 2, -3, 3)
    g = x.T * x
    check(g == g.T, "Gram đối xứng")
    return display(L.mat(x, B("X")), f"\\mathbf{{X}}\\T\\mathbf{{X}} = {L.mat(g)}") + "\n\nMa trận Gram đối xứng, và cỡ $d\\times d$ với $d$ là số đặc trưng (bất kể số mẫu)."


def prop_residual() -> str:
    x, y, g, xty, w = _fit(rng(51), 4)
    resid = y - x * w
    check(x.T * resid == Matrix.zeros(2, 1), "phần dư vuông góc")
    return f"Với $\\mathbf{{w}} = {L.vec_inline(w)}$:\n\n" + display(f"\\mathbf{{X}}\\T(\\mathbf{{y}} - \\mathbf{{X}}\\mathbf{{w}}) = {L.mat(x.T * resid)}") + \
        "\n\nNghiệm bình phương tối thiểu làm phần dư vuông góc với không gian cột của $\\mathbf{X}$: $\\mathbf{X}\\mathbf{w}$ là hình chiếu của $\\mathbf{y}$ lên không gian cột."


def mistake_singular() -> str:
    x = M(61, 4, 2, 1, 4)
    x[:, 1] = 2 * x[:, 0]
    g = x.T * x
    check(g.det() == 0, "Gram suy biến")
    return (display(L.mat(x, B("X")), f"\\mathbf{{X}}\\T\\mathbf{{X}} = {L.mat(g)},\\qquad \\det(\\mathbf{{X}}\\T\\mathbf{{X}}) = {L.num(g.det())}")
            + "\n\n**Sai:** vẫn áp dụng $\\mathbf{w} = (\\mathbf{X}\\T\\mathbf{X})^{-1}\\mathbf{X}\\T\\mathbf{y}$. "
              "Cột thứ hai gấp đôi cột thứ nhất (đặc trưng thừa), nên $\\mathbf{X}\\T\\mathbf{X}$ suy biến và không có nghịch đảo: phương trình chuẩn có vô số nghiệm. "
              "Cách xử lý: bỏ đặc trưng thừa hoặc thêm regularization (ridge).")


def mistake_shape() -> str:
    x = M(71, 4, 3, -2, 2)
    w = Matrix([1, -1, 2])
    return (display(L.mat(x, B("X")), L.mat(w, B("w")))
            + f"\n\n**Sai:** viết $\\mathbf{{w}}\\mathbf{{X}}$. $\\mathbf{{w}}$ cỡ ${w.rows}\\times 1$ và $\\mathbf{{X}}$ cỡ ${x.rows}\\times {x.cols}$: "
              f"số cột của $\\mathbf{{w}}$ là $1$, khác số hàng của $\\mathbf{{X}}$ là ${x.rows}$, nên phép nhân không xác định.\n\n"
            + f"**Đúng:** $\\mathbf{{X}}\\mathbf{{w}}$ cỡ $({x.rows}\\times {x.cols})({w.rows}\\times 1) \\to {x.rows}\\times 1$. Luôn kiểm tra kích thước trước khi viết công thức.")


EXAMPLES = {"data-matrix": data_matrix, "normal-eq": normal_eq, "attention-scores": attention_scores, "param-count": param_count,
            "prop-gram-sym": prop_gram_sym, "prop-residual": prop_residual, "mistake-singular": mistake_singular, "mistake-shape": mistake_shape}


def build():
    return write_examples(LESSON, EXAMPLES)
