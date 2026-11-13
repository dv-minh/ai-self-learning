r"""Ví dụ số bài 1.1 Vector."""
from __future__ import annotations

import numpy as np
from sympy import Rational, sqrt

from ..core import latex as L
from ..core.steps import componentwise
from ..core.verify import check, np_close, to_np
from .common import NV, V, display, write_examples

LESSON = "1.1"
B, N = L.bold, L.num_paren


def add() -> str:
    u, v = V(11, 2), V(12, 2)
    r = u + v
    np_close(r, to_np(u) + to_np(v))
    step = componentwise("\\mathbf{u}+\\mathbf{v}", [f"{N(a)} + {N(b)}" for a, b in zip(u, v)], r)
    return display(L.mat(u, B("u")), L.mat(v, B("v")), step.latex)


def lincomb4() -> str:
    u, v, w = V(21, 4), V(22, 4), V(23, 4)
    a, b, c = Rational(2), Rational(-1), Rational(3)
    r = a * u + b * v + c * w
    np_close(r, 2 * to_np(u) - to_np(v) + 3 * to_np(w))
    parts = [f"{N(a)}\\cdot {N(u[i])} + {N(b)}\\cdot {N(v[i])} + {N(c)}\\cdot {N(w[i])}" for i in range(4)]
    step = componentwise("2\\mathbf{u} - \\mathbf{v} + 3\\mathbf{w}", parts, r)
    return display(L.mat(u, B("u")), L.mat(v, B("v")), L.mat(w, B("w")), step.latex)


def norms() -> str:
    u = NV(31, 3)
    l1 = sum(abs(x) for x in u)
    s = sum(x * x for x in u)
    l2 = sqrt(s)
    check(float(l1) == np.linalg.norm(to_np(u), 1) and float(l2) == np.linalg.norm(to_np(u)), "norms numpy")
    return display(
        L.mat(u, B("u")),
        f"\\norm{{\\mathbf{{u}}}}_1 = " + " + ".join(f"|{L.num(x)}|" for x in u) + f" = {L.num(l1)}",
        f"\\norm{{\\mathbf{{u}}}}_2 = \\sqrt{{" + " + ".join(f"{N(x)}^2" for x in u) + f"}} = \\sqrt{{{s}}} = {l2}",
    )


def unit() -> str:
    u = NV(41, 2)
    l2 = sqrt(sum(x * x for x in u))
    e = u / l2
    check(sum(x * x for x in e) == 1, "unit norm")
    return display(
        L.mat(u, B("u")),
        f"\\norm{{\\mathbf{{u}}}}_2 = {l2}",
        f"\\hat{{\\mathbf{{u}}}} = \\frac{{\\mathbf{{u}}}}{{\\norm{{\\mathbf{{u}}}}_2}} = {L.mat(e)}",
        f"\\norm{{\\hat{{\\mathbf{{u}}}}}}_2 = \\sqrt{{" + " + ".join(f"{N(x)}^2" for x in e) + f"}} = 1",
    )


def mistake_norm() -> str:
    u = NV(51, 3)
    l1 = sum(abs(x) for x in u)
    l2 = sqrt(sum(x * x for x in u))
    return (
        "**Sai:** lấy tổng các giá trị tuyệt đối (đó là chuẩn L1) rồi gọi là chuẩn L2:\n\n"
        + display(L.mat(u, B("u")), f"\\norm{{\\mathbf{{u}}}}_2 \\overset{{?}}{{=}} " + " + ".join(f"|{L.num(x)}|" for x in u) + f" = {L.num(l1)}")
        + "\n\n**Đúng:** bình phương từng thành phần, cộng, rồi lấy căn:\n\n"
        + display(f"\\norm{{\\mathbf{{u}}}}_2 = \\sqrt{{" + " + ".join(f"{N(x)}^2" for x in u) + f"}} = {l2}")
    )


EXAMPLES = {"add": add, "lincomb4": lincomb4, "norms": norms, "unit": unit, "mistake-norm": mistake_norm}


def prop_commute() -> str:
    u, v = V(61, 3), V(62, 3)
    check(u + v == v + u, "commute")
    return display(f"\\mathbf{{u}}+\\mathbf{{v}} = {L.mat(u + v)},\\qquad \\mathbf{{v}}+\\mathbf{{u}} = {L.mat(v + u)}") \
        + f"\n\nvới $\\mathbf{{u}} = {L.vec_inline(u)}$ và $\\mathbf{{v}} = {L.vec_inline(v)}$."


def prop_distribute() -> str:
    u, v = V(63, 3), V(64, 3)
    k = Rational(-2)
    check(k * (u + v) == k * u + k * v, "distribute")
    return f"Với $k = {L.num(k)}$, $\\mathbf{{u}} = {L.vec_inline(u)}$, $\\mathbf{{v}} = {L.vec_inline(v)}$:\n\n" + display(
        f"k(\\mathbf{{u}}+\\mathbf{{v}}) = {L.mat(k * (u + v))},\\qquad k\\mathbf{{u}}+k\\mathbf{{v}} = {L.mat(k * u)}+{L.mat(k * v)} = {L.mat(k * u + k * v)}")


def prop_norm_scale() -> str:
    u = NV(65, 3)
    k = Rational(-3)
    nu = sqrt(sum(x * x for x in u))
    nk = sqrt(sum((k * x) ** 2 for x in u))
    check(nk == abs(k) * nu, "norm scale")
    return f"Với $k = {L.num(k)}$ và $\\mathbf{{u}} = {L.vec_inline(u)}$:\n\n" + display(
        f"\\norm{{k\\mathbf{{u}}}}_2 = {nk},\\qquad |k|\\,\\norm{{\\mathbf{{u}}}}_2 = {abs(k)}\\cdot {nu} = {abs(k) * nu}")


def prop_triangle() -> str:
    u, v = NV(66, 2), NV(67, 2)
    s = sum(x * x for x in (u + v))
    nu, nv = sqrt(sum(x * x for x in u)), sqrt(sum(x * x for x in v))
    lhs = float(sqrt(s))
    check(lhs <= float(nu + nv) + 1e-12, "triangle")
    return f"Với $\\mathbf{{u}} = {L.vec_inline(u)}$ và $\\mathbf{{v}} = {L.vec_inline(v)}$:\n\n" + display(
        f"\\norm{{\\mathbf{{u}}+\\mathbf{{v}}}}_2 = \\sqrt{{{s}}} \\approx {lhs:.3f},\\qquad \\norm{{\\mathbf{{u}}}}_2+\\norm{{\\mathbf{{v}}}}_2 = {nu}+{nv} = {nu + nv}") \
        + f"\n\nĐúng như bất đẳng thức tam giác: ${lhs:.3f} \\le {nu + nv}$."


EXAMPLES.update({"prop-commute": prop_commute, "prop-distribute": prop_distribute,
                 "prop-norm-scale": prop_norm_scale, "prop-triangle": prop_triangle})


def build():
    return write_examples(LESSON, EXAMPLES)
