r"""SymPy -> LaTeX theo quy chuẩn ký hiệu (docs/spec.md mục 7).

Chỉ dùng macro đã khai báo: \R, \T, \norm, \rank, \tr, \diag.
"""
from __future__ import annotations

from sympy import Integer, Matrix, Rational


def rat_str(x) -> str:
    """Số hữu tỉ -> chuỗi "p/q" đã rút gọn (số nguyên thì chỉ "p")."""
    r = Rational(x)
    return str(r.p) if r.q == 1 else f"{r.p}/{r.q}"


def num(x) -> str:
    """Số hữu tỉ -> LaTeX."""
    r = Rational(x)
    if r.q == 1:
        return str(r.p)
    sign = "-" if r.p < 0 else ""
    return f"{sign}\\frac{{{abs(r.p)}}}{{{r.q}}}"


def num_paren(x) -> str:
    """Như num nhưng bọc ngoặc khi âm (dùng trong tích)."""
    r = Rational(x)
    return f"({num(r)})" if r < 0 else num(r)


def mat(m: Matrix, name: str | None = None) -> str:
    rows = " \\\\ ".join(" & ".join(num(m[i, j]) for j in range(m.cols)) for i in range(m.rows))
    body = f"\\begin{{bmatrix}} {rows} \\end{{bmatrix}}"
    return f"{name} = {body}" if name else body


def aug(m: Matrix, left: int) -> str:
    """Ma trận mở rộng: `left` cột đầu là hệ số, phần còn lại là vế phải, ngăn bằng vạch đứng."""
    spec = "c" * left + "|" + "c" * (m.cols - left)
    rows = " \\\\ ".join(" & ".join(num(m[i, j]) for j in range(m.cols)) for i in range(m.rows))
    return f"\\left[\\begin{{array}}{{{spec}}} {rows} \\end{{array}}\\right]"


def det_bars(m: Matrix) -> str:
    rows = " \\\\ ".join(" & ".join(num(m[i, j]) for j in range(m.cols)) for i in range(m.rows))
    return f"\\begin{{vmatrix}} {rows} \\end{{vmatrix}}"


def vec_inline(v: Matrix) -> str:
    """Vector cột viết ngang (a, b, c)^T ở dạng \\begin{bmatrix}...\\end{bmatrix} thu gọn."""
    return "(" + ", ".join(num(x) for x in v) + ")\\T"


def vec(v: Matrix, name: str | None = None) -> str:
    return mat(v, name)


def entries(m: Matrix) -> list[list[str]]:
    return [[rat_str(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def dot_expr(row: list, col: list) -> str:
    """a1*b1 + a2*b2 + ... với ngoặc cho số âm."""
    return " + ".join(f"{num_paren(a)} \\cdot {num_paren(b)}" for a, b in zip(row, col))


def bold(name: str) -> str:
    return f"\\mathbf{{{name}}}"


def ident(n: int) -> Matrix:
    return Matrix.eye(n)


__all__ = ["rat_str", "num", "num_paren", "mat", "vec", "entries", "dot_expr", "bold", "Integer"]


def linear_form(coeffs: list, names: list[str]) -> str:
    """a1 x1 + a2 x2 - ... (bỏ hệ số 1, bỏ số hạng 0, dấu đúng)."""
    out = ""
    for c, n in zip(coeffs, names):
        c = Rational(c)
        if c == 0:
            continue
        mag = abs(c)
        body = n if mag == 1 else f"{num(mag)}{n}"
        if not out:
            out = ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return out or "0"


def system(a: Matrix, b: Matrix, var: str = "x") -> str:
    """Hệ Ax = b dạng các phương trình căn lề theo dấu =."""
    names = [f"{var}_{{{j + 1}}}" for j in range(a.cols)]
    lines = [f"{linear_form(list(a[i, :]), names)} &= {num(b[i])}" for i in range(a.rows)]
    return "\\left\\{\\begin{aligned} " + " \\\\ ".join(lines) + " \\end{aligned}\\right."


def expr_mat(rows: list[list[str]]) -> str:
    """Ma trận bmatrix từ các biểu thức LaTeX (dùng trong lời giải từng bước)."""
    body = " \\\\ ".join(" & ".join(r) for r in rows)
    return f"\\begin{{bmatrix}} {body} \\end{{bmatrix}}"
