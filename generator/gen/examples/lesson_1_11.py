r"""Ví dụ số bài 1.11 Ánh xạ tuyến tính."""
from __future__ import annotations

from sympy import Matrix, Rational

from ..core import latex as L
from ..core.verify import check
from .common import M, V, display, write_examples

LESSON = "1.11"
B, N = L.bold, L.num_paren
E1, E2 = Matrix([1, 0]), Matrix([0, 1])


def rotate() -> str:
    a = Matrix([[0, -1], [1, 0]])
    v = V(11, 2, -4, 4)
    return display(L.mat(a, B("A")), f"\\mathbf{{A}}\\mathbf{{e}}_1 = {L.mat(a * E1)},\\quad \\mathbf{{A}}\\mathbf{{e}}_2 = {L.mat(a * E2)}",
                   L.mat(v, B("v")), f"\\mathbf{{A}}\\mathbf{{v}} = {L.mat(a * v)}") + \
        "\n\nCác cột của $\\mathbf{A}$ chính là ảnh của $\\mathbf{e}_1$ và $\\mathbf{e}_2$. Ma trận này quay mặt phẳng $90^\\circ$ ngược chiều kim đồng hồ."


def shear_3d() -> str:
    a = Matrix([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
    v = V(21, 3, -4, 4)
    return display(L.mat(a, B("A")), L.mat(v, B("v")), f"\\mathbf{{A}}\\mathbf{{v}} = {L.mat(a * v)}") + \
        "\n\nPhép trượt này giữ nguyên thành phần thứ hai và thứ ba, chỉ cộng một bội của thành phần thứ hai vào thành phần thứ nhất."


def area_scale() -> str:
    a = M(31, 2, 2, -4, 4)
    return display(L.mat(a, B("A")), f"\\det(\\mathbf{{A}}) = {L.num(a.det())}") + \
        f"\n\nẢnh của hình vuông đơn vị là hình bình hành có hai cạnh là hai cột của $\\mathbf{{A}}$, với diện tích $|\\det(\\mathbf{{A}})| = {L.num(abs(a.det()))}$."


def compose() -> str:
    a, b = M(41, 2, 2, -3, 3), M(42, 2, 2, -3, 3)
    while a * b == b * a:
        b = M(42 + 1000 + int(abs(b[0, 0])), 2, 2, -3, 3)
    v = V(43, 2, -3, 3)
    check(b * (a * v) == (b * a) * v, "hợp ánh xạ")
    return (
        f"Áp dụng $T_1(\\mathbf{{x}}) = \\mathbf{{A}}\\mathbf{{x}}$ trước, rồi $T_2(\\mathbf{{y}}) = \\mathbf{{B}}\\mathbf{{y}}$:\n\n"
        + display(L.mat(a, B("A")), L.mat(b, B("B")), L.mat(v, B("v")),
                  f"T_2(T_1(\\mathbf{{v}})) = \\mathbf{{B}}(\\mathbf{{A}}\\mathbf{{v}}) = \\mathbf{{B}}{L.mat(a * v)} = {L.mat(b * (a * v))}",
                  f"\\mathbf{{B}}\\mathbf{{A}} = {L.mat(b * a)},\\qquad (\\mathbf{{B}}\\mathbf{{A}})\\mathbf{{v}} = {L.mat((b * a) * v)}")
        + "\n\nHai cách cho cùng kết quả: ma trận của phép hợp là tích $\\mathbf{B}\\mathbf{A}$, với phép làm trước đứng bên phải."
    )


def prop_linearity() -> str:
    a = M(51, 2, 2, -3, 3)
    u, v = V(52, 2, -3, 3), V(53, 2, -3, 3)
    k = Rational(3)
    check(a * (u + v) == a * u + a * v and a * (k * u) == k * (a * u), "tuyến tính")
    return f"Với $k = {L.num(k)}$, $\\mathbf{{u}} = {L.vec_inline(u)}$, $\\mathbf{{v}} = {L.vec_inline(v)}$ và $\\mathbf{{A}} = {L.mat(a)}$:\n\n" + display(
        f"T(\\mathbf{{u}}+\\mathbf{{v}}) = {L.mat(a * (u + v))},\\qquad T(\\mathbf{{u}})+T(\\mathbf{{v}}) = {L.mat(a * u + a * v)}",
        f"T(k\\mathbf{{u}}) = {L.mat(a * (k * u))},\\qquad k\\,T(\\mathbf{{u}}) = {L.mat(k * (a * u))}")


def prop_origin() -> str:
    a = M(61, 3, 3, -3, 3)
    z = Matrix.zeros(3, 1)
    check(a * z == z, "T(0)=0")
    return display(f"T(\\mathbf{{0}}) = \\mathbf{{A}}\\mathbf{{0}} = {L.mat(a * z)} = \\mathbf{{0}}") + "\n\nMọi phép biến đổi tuyến tính đều giữ nguyên gốc tọa độ."


def mistake_order() -> str:
    a, b = M(71, 2, 2, -3, 3), M(72, 2, 2, -3, 3)
    while a * b == b * a:
        b = M(72 + 1000 + int(abs(b[0, 0])), 2, 2, -3, 3)
    return (
        f"Áp dụng $\\mathbf{{A}}$ trước, rồi $\\mathbf{{B}}$.\n\n"
        + display(L.mat(a, B("A")), L.mat(b, B("B")))
        + "\n\n**Sai:** viết ma trận của phép hợp theo thứ tự đọc (trái sang phải), tức $\\mathbf{AB}$:\n\n"
        + display(f"\\mathbf{{A}}\\mathbf{{B}} = {L.mat(a * b)}")
        + "\n\n**Đúng:** phép làm trước đứng bên phải (vì nó tác động lên vector trước), tức $\\mathbf{BA}$:\n\n"
        + display(f"\\mathbf{{B}}\\mathbf{{A}} = {L.mat(b * a)}")
    )


def mistake_affine() -> str:
    a = M(81, 2, 2, -3, 3)
    bb = Matrix([2, -1])
    z = Matrix.zeros(2, 1)
    u, v = V(82, 2, -3, 3), V(83, 2, -3, 3)
    t = lambda x: a * x + bb
    check(t(z) != z and t(u + v) != t(u) + t(v), "affine không tuyến tính")
    return (
        f"Xét $S(\\mathbf{{x}}) = \\mathbf{{A}}\\mathbf{{x}} + \\mathbf{{b}}$ với $\\mathbf{{A}} = {L.mat(a)}$, $\\mathbf{{b}} = {L.mat(bb)}$.\n\n"
        + display(f"S(\\mathbf{{0}}) = {L.mat(t(z))} \\neq \\mathbf{{0}}", f"S(\\mathbf{{u}}+\\mathbf{{v}}) = {L.mat(t(u + v))},\\qquad S(\\mathbf{{u}})+S(\\mathbf{{v}}) = {L.mat(t(u) + t(v))}")
        + "\n\n**Sai:** coi $S$ là ánh xạ tuyến tính. Cộng thêm hằng số $\\mathbf{b}$ làm hỏng tính chất $T(\\mathbf{0}) = \\mathbf{0}$ và tính cộng tính. Đây là phép biến đổi **affine**, không phải tuyến tính; ta xử lý bằng cách coi $\\mathbf{b}$ là một cột thêm vào ma trận (kỹ thuật tọa độ thuần nhất)."
    )


EXAMPLES = {"rotate": rotate, "shear-3d": shear_3d, "area-scale": area_scale, "compose": compose,
            "prop-linearity": prop_linearity, "prop-origin": prop_origin, "mistake-order": mistake_order, "mistake-affine": mistake_affine}


def build():
    return write_examples(LESSON, EXAMPLES)
