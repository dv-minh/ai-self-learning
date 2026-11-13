r"""Ví dụ số bài 1.2 Tích vô hướng, góc, cosine similarity."""
from __future__ import annotations

import numpy as np
from sympy import Matrix, Rational, sqrt

from ..core import latex as L
from ..core.verify import check, to_np
from .common import NV, V, display, write_examples

LESSON = "1.2"
B, N = L.bold, L.num_paren


def _dot(u: Matrix, v: Matrix):
    d = (u.T * v)[0, 0]
    check(np.isclose(float(d), float(np.dot(to_np(u).ravel(), to_np(v).ravel()))), "dot numpy")
    return d


def dot() -> str:
    u, v = V(11, 3), V(12, 3)
    d = _dot(u, v)
    return display(L.mat(u, B("u")), L.mat(v, B("v")),
                   f"\\mathbf{{u}}\\T\\mathbf{{v}} = " + " + ".join(f"{N(a)}\\cdot {N(b)}" for a, b in zip(u, v)) + f" = {L.num(d)}")


def cosine() -> str:
    u, v = NV(21, 3), NV(22, 3)
    nu, nv = sqrt(sum(x * x for x in u)), sqrt(sum(x * x for x in v))
    c = _dot(u, v) / (nu * nv)
    check(c.is_Rational, "cosine hữu tỉ")
    return display(L.mat(u, B("u")), L.mat(v, B("v")),
                   f"\\mathbf{{u}}\\T\\mathbf{{v}} = {L.num(_dot(u, v))},\\quad \\norm{{\\mathbf{{u}}}}_2 = {nu},\\quad \\norm{{\\mathbf{{v}}}}_2 = {nv}",
                   f"\\cos\\theta = \\frac{{{L.num(_dot(u, v))}}}{{{nu}\\cdot {nv}}} = {L.num(c)}")


def orthogonal() -> str:
    for seed in range(100):
        u, v0 = V(31 + seed, 3, -4, 4), V(131 + seed, 3, -4, 4)
        p = v0 - (_dot(u, v0) / _dot(u, u)) * u
        if all(x.q == 1 for x in p) and any(x != 0 for x in p) and all(abs(x) <= 6 for x in p):
            v = p
            break
    d = _dot(u, v)
    check(d == 0, "orthogonal")
    return display(L.mat(u, B("u")), L.mat(v, B("v")),
                   f"\\mathbf{{u}}\\T\\mathbf{{v}} = " + " + ".join(f"{N(a)}\\cdot {N(b)}" for a, b in zip(u, v)) + f" = {L.num(d)}") \
        + "\n\nTích vô hướng bằng $0$ nên hai vector trực giao."


def project() -> str:
    u, v = V(41, 2, -5, 5), V(42, 2, -5, 5)
    uv, vv = _dot(u, v), _dot(v, v)
    proj = (uv / vv) * v
    resid = u - proj
    check(_dot(resid, v) == 0, "phần dư vuông góc")
    return display(L.mat(u, B("u")), L.mat(v, B("v")),
                   f"\\frac{{\\mathbf{{u}}\\T\\mathbf{{v}}}}{{\\mathbf{{v}}\\T\\mathbf{{v}}}} = \\frac{{{L.num(uv)}}}{{{L.num(vv)}}} = {L.num(uv / vv)}",
                   f"\\mathrm{{proj}}_{{\\mathbf{{v}}}}\\mathbf{{u}} = {L.num(uv / vv)}\\cdot {L.mat(v)} = {L.mat(proj)}",
                   f"\\mathbf{{u}} - \\mathrm{{proj}}_{{\\mathbf{{v}}}}\\mathbf{{u}} = {L.mat(resid)},\\quad "
                   f"\\left(\\mathbf{{u}} - \\mathrm{{proj}}_{{\\mathbf{{v}}}}\\mathbf{{u}}\\right)\\T\\mathbf{{v}} = {L.num(_dot(resid, v))}") \
        + "\n\nPhần dư vuông góc với $\\mathbf{v}$ (tích vô hướng bằng $0$), đúng như kỳ vọng."


def mistake_cosine() -> str:
    u, v = NV(51, 3), NV(52, 3)
    nu, nv = sqrt(sum(x * x for x in u)), sqrt(sum(x * x for x in v))
    d = _dot(u, v)
    wrong, right = d / nu, d / (nu * nv)
    return (
        "**Sai:** chỉ chia cho một chuẩn:\n\n"
        + display(L.mat(u, B("u")), L.mat(v, B("v")), f"\\frac{{\\mathbf{{u}}\\T\\mathbf{{v}}}}{{\\norm{{\\mathbf{{u}}}}_2}} = {L.num(wrong)}")
        + ("\n\nKết quả vượt ra ngoài khoảng $[-1, 1]$ nên chắc chắn sai: cosine luôn nằm trong khoảng này.\n\n" if abs(wrong) > 1 else "\n\n")
        + "**Đúng:** chia cho tích hai chuẩn:\n\n"
        + display(f"\\cos\\theta = \\frac{{{L.num(d)}}}{{{nu}\\cdot {nv}}} = {L.num(right)}")
    )


EXAMPLES = {"dot": dot, "cosine": cosine, "orthogonal": orthogonal, "project": project, "mistake-cosine": mistake_cosine}


def prop_symmetric() -> str:
    u, v = V(61, 3), V(62, 3)
    check(_dot(u, v) == _dot(v, u), "dot symmetric")
    return f"Với $\\mathbf{{u}} = {L.vec_inline(u)}$ và $\\mathbf{{v}} = {L.vec_inline(v)}$:\n\n" + display(
        f"\\mathbf{{u}}\\T\\mathbf{{v}} = {L.num(_dot(u, v))},\\qquad \\mathbf{{v}}\\T\\mathbf{{u}} = {L.num(_dot(v, u))}")


def prop_linear() -> str:
    u, v = V(63, 3), V(64, 3)
    k = Rational(3)
    check(_dot(k * u, v) == k * _dot(u, v), "dot linear")
    return f"Với $k = {L.num(k)}$, $\\mathbf{{u}} = {L.vec_inline(u)}$, $\\mathbf{{v}} = {L.vec_inline(v)}$:\n\n" + display(
        f"(k\\mathbf{{u}})\\T\\mathbf{{v}} = {L.num(_dot(k * u, v))},\\qquad k\\,(\\mathbf{{u}}\\T\\mathbf{{v}}) = {L.num(k)}\\cdot {N(_dot(u, v))} = {L.num(k * _dot(u, v))}")


def prop_self() -> str:
    u = NV(65, 3)
    nu = sqrt(sum(x * x for x in u))
    check(_dot(u, u) == nu ** 2, "u.u = norm^2")
    return f"Với $\\mathbf{{u}} = {L.vec_inline(u)}$:\n\n" + display(
        f"\\mathbf{{u}}\\T\\mathbf{{u}} = {L.num(_dot(u, u))},\\qquad \\norm{{\\mathbf{{u}}}}_2^2 = {nu}^2 = {L.num(nu ** 2)}")


def prop_cauchy() -> str:
    u, v = NV(66, 3), NV(67, 3)
    nu, nv = sqrt(sum(x * x for x in u)), sqrt(sum(x * x for x in v))
    d = _dot(u, v)
    check(abs(d) <= nu * nv, "Cauchy–Schwarz")
    return f"Với $\\mathbf{{u}} = {L.vec_inline(u)}$ và $\\mathbf{{v}} = {L.vec_inline(v)}$:\n\n" + display(
        f"|\\mathbf{{u}}\\T\\mathbf{{v}}| = {L.num(abs(d))},\\qquad \\norm{{\\mathbf{{u}}}}_2\\norm{{\\mathbf{{v}}}}_2 = {nu}\\cdot {nv} = {nu * nv}") \
        + f"\n\nĐúng như bất đẳng thức Cauchy–Schwarz: ${L.num(abs(d))} \\le {nu * nv}$. Nhờ bất đẳng thức này cosine luôn nằm trong $[-1, 1]$."


EXAMPLES.update({"prop-symmetric": prop_symmetric, "prop-linear": prop_linear, "prop-self": prop_self, "prop-cauchy": prop_cauchy})


def build():
    return write_examples(LESSON, EXAMPLES)
