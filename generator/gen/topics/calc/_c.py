"""Tiện ích cho Phần 3 (Giải tích): biểu thức SymPy <-> cú pháp nhập của người học, kiểm chứng bằng sai phân hữu hạn."""
from __future__ import annotations

import random

import numpy as np
from sympy import Basic, Matrix, Rational, Symbol, diff, exp, latex, lambdify, log, sin, cos, sqrt, symbols, sympify

from ...core import latex as L
from ...core.verify import check
from ...schema import Answer, Check, Prompt, Step
from ..linalg._base import TopicSpec, build

PART = "calculus"
x, y, z = symbols("x y z")


def cb(**kw):
    return build(part=PART, **kw)


def tex(e: Basic) -> str:
    """LaTeX của biểu thức, dùng \\ln cho logarit tự nhiên."""
    return latex(e, ln_notation=True)


def to_input(e: Basic) -> str:
    """Biểu thức SymPy -> chuỗi theo cú pháp nhập (dấu mũ ^, ln). Kiểm tra vòng khứ hồi."""
    s = str(e).replace("**", "^").replace("log(", "ln(")
    back = sympify(s.replace("^", "**").replace("ln(", "log("), locals={"x": x, "y": y, "z": z})
    check((back - e).simplify() == 0, f"cú pháp nhập không khứ hồi: {s}")
    return s


def expr_answer(e: Basic, variables: list[str]) -> Answer:
    return Answer(kind="expression", value=to_input(e), variables=variables)


def fd_check(f: Basic, g: Basic, vars_: list[Symbol], points: int = 4) -> None:
    """Kiểm chứng độc lập: đạo hàm riêng g theo vars_[0] khớp sai phân hữu hạn của f (float, tại điểm dương)."""
    ff = lambdify(vars_, f, "numpy")
    gg = lambdify(vars_, g, "numpy")
    rng = np.random.RandomState(7)
    done = 0
    for _ in range(40):
        pt = list(rng.uniform(0.6, 2.4, size=len(vars_)))
        try:
            h = 1e-6
            hi, lo = list(pt), list(pt)
            hi[0] += h
            lo[0] -= h
            num = (float(ff(*hi)) - float(ff(*lo))) / (2 * h)
            ana = float(gg(*pt))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            continue
        if not (np.isfinite(num) and np.isfinite(ana)):
            continue
        check(abs(num - ana) <= 1e-5 * max(1.0, abs(ana)), f"đạo hàm sai: số {num} khác giải tích {ana}")
        done += 1
        if done >= points:
            return
    raise AssertionError("không đủ điểm để kiểm chứng sai phân")


def num(v) -> str:
    return L.num(Rational(v))


def display_f(name: str, args: str, e: Basic) -> str:
    return f"{name}({args}) = {tex(e)}"


__all__ = ["cb", "tex", "to_input", "expr_answer", "fd_check", "num", "display_f", "x", "y", "z", "TopicSpec", "Answer", "Check", "Prompt", "Step", "check",
           "Rational", "Matrix", "diff", "exp", "log", "sin", "cos", "sqrt", "random", "sympify", "L", "symbols"]
