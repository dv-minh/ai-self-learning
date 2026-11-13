"""Ghi lời giải từng bước."""
from __future__ import annotations

from sympy import Matrix

from ..schema import Step
from . import latex as L


def componentwise(label: str, parts: list[str], result: Matrix, text: str = "Thực hiện trên từng thành phần:") -> Step:
    """Một bước dạng  label = [p1; p2; ...] = result  (vector cột)."""
    col = " \\\\ ".join(parts)
    return Step(text=text, latex=f"{label} = \\begin{{bmatrix}} {col} \\end{{bmatrix}} = {L.mat(result)}")


def matmul_entry_step(a: Matrix, b: Matrix, i: int, j: int, an: str = "A", bn: str = "B") -> Step:
    """Một phần tử c_ij = hàng i của A nhân cột j của B."""
    row = [a[i, k] for k in range(a.cols)]
    col = [b[k, j] for k in range(b.rows)]
    total = sum(x * y for x, y in zip(row, col))
    sub = f"c_{{{i + 1}{j + 1}}}"
    return Step(
        text=f"Phần tử {sub}: lấy hàng {i + 1} của ${an}$ nhân cột {j + 1} của ${bn}$.",
        latex=f"{sub} = {L.dot_expr(row, col)} = {L.num(total)}",
    )


def matmul_steps(a: Matrix, b: Matrix, an: str = "A", bn: str = "B") -> list[Step]:
    """Ghi đủ từng phần tử khi ≤ 6 phần tử; nếu nhiều hơn thì ghi 2 phần tử đầu rồi đưa kết quả."""
    c = a * b
    steps = [
        Step(
            text=f"Kiểm tra kích thước: ${an}$ là ${a.rows}\\times {a.cols}$, ${bn}$ là ${b.rows}\\times {b.cols}$. "
            f"Số cột của ${an}$ bằng số hàng của ${bn}$ nên phép nhân xác định, kết quả là ${c.rows}\\times {c.cols}$."
        )
    ]
    cells = [(i, j) for i in range(c.rows) for j in range(c.cols)]
    shown = cells if len(cells) <= 6 else cells[:2]
    steps += [matmul_entry_step(a, b, i, j, an, bn) for i, j in shown]
    if len(shown) < len(cells):
        steps.append(Step(text="Các phần tử còn lại tính tương tự (hàng nhân cột)."))
    steps.append(Step(text="Kết quả:", latex=L.mat(c, f"{an}{bn}")))
    return steps
