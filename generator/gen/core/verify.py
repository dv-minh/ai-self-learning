"""Kiểm chứng độc lập. NumPy chỉ được dùng ở đây."""
from __future__ import annotations

import numpy as np
from sympy import Matrix


class VerificationError(AssertionError):
    pass


def to_np(m: Matrix) -> np.ndarray:
    return np.array(m.tolist(), dtype=float)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise VerificationError(msg)


def verify_matmul(a: Matrix, b: Matrix, c: Matrix) -> list[str]:
    """c = a*b: so với NumPy (sai số 1e-9) và kiểm kích thước. Trả về nhãn đã kiểm."""
    check(a.cols == b.rows, "kích thước không khớp")
    check((c.rows, c.cols) == (a.rows, b.cols), "kích thước kết quả sai")
    check(np.allclose(to_np(a) @ to_np(b), to_np(c), rtol=1e-9, atol=1e-9), "NumPy không khớp SymPy")
    return ["sympy", "numpy", "shape"]


def verify_hadamard(a: Matrix, b: Matrix, h: Matrix) -> list[str]:
    check(a.shape == b.shape == h.shape, "kích thước Hadamard sai")
    check(np.allclose(to_np(a) * to_np(b), to_np(h), rtol=1e-9, atol=1e-9), "NumPy Hadamard không khớp")
    return ["sympy", "numpy", "shape"]


def np_close(m: Matrix, arr, msg: str = "NumPy không khớp SymPy") -> None:
    arr = np.asarray(arr, dtype=float).reshape(m.shape)
    check(np.allclose(to_np(m), arr, rtol=1e-9, atol=1e-9), msg)


def laplace_det(m: Matrix):
    """Khai triển Laplace theo hàng 1 (đệ quy, chính xác): phương pháp độc lập với det của SymPy."""
    n = m.rows
    if n == 1:
        return m[0, 0]
    return sum(((-1) ** j) * m[0, j] * laplace_det(m.minor_submatrix(0, j)) for j in range(n))


def verify_det(a: Matrix, d) -> list[str]:
    """Laplace so với khử Gauss (qua ref) so với numpy.linalg.det."""
    from .gauss import eliminate

    check(laplace_det(a) == d, "det: Laplace khác kết quả")
    el = eliminate(a, mode="ref")
    prod = 1
    for i in range(a.rows):
        prod *= el.matrix[i, i]
    check((-1) ** el.swaps * prod == d, "det: khử Gauss khác kết quả")
    check(np.isclose(np.linalg.det(to_np(a)), float(d), rtol=1e-8, atol=1e-8), "det: NumPy khác kết quả")
    return ["sympy", "laplace", "gauss", "numpy"]


def verify_inverse(a: Matrix, inv: Matrix) -> list[str]:
    n = a.rows
    check(a * inv == Matrix.eye(n) and inv * a == Matrix.eye(n), "A·A⁻¹ ≠ I")
    np_close(inv, np.linalg.inv(to_np(a)))
    return ["sympy", "identity", "numpy"]


def verify_rank(a: Matrix, r: int) -> None:
    check(a.rank() == r, "rank SymPy khác dự kiến")
    check(int(np.linalg.matrix_rank(to_np(a))) == r, "rank NumPy (SVD) khác dự kiến")


def verify_rref(a: Matrix, r: Matrix, pivots: list[int]) -> list[str]:
    ref, piv = a.rref()
    check(ref == r and list(piv) == list(pivots), "RREF khác sympy.Matrix.rref")
    verify_rank(a, len(pivots))
    return ["sympy", "rref", "numpy-rank"]


def verify_solution(a: Matrix, b: Matrix, x: Matrix) -> None:
    check(a * x == b, "nghiệm không thỏa Ax = b")
    np_close(b, to_np(a) @ to_np(x))


def verify_solution_set(a: Matrix, b: Matrix, x0: Matrix, basis: list[Matrix]) -> list[str]:
    """x0 thỏa Ax = b; mỗi vector cơ sở thỏa Av = 0; số vector bằng số chiều không gian null."""
    verify_solution(a, b, x0)
    for v in basis:
        check(a * v == Matrix.zeros(a.rows, 1), "vector cơ sở không thuộc không gian null")
    check(len(basis) == a.cols - a.rank(), "số vector cơ sở sai")
    if basis:
        check(Matrix.hstack(*basis).rank() == len(basis), "cơ sở phụ thuộc tuyến tính")
    check(a.rank() == Matrix.hstack(a, b).rank(), "hệ không nhất quán")
    return ["sympy", "substitution", "rank"]


def matmul_shape(a_shape: tuple[int, int], b_shape: tuple[int, int]) -> tuple[int, int] | None:
    """Kích thước tích, hoặc None nếu không xác định. Kiểm chứng bằng NumPy trên ma trận 0."""
    expected = (a_shape[0], b_shape[1]) if a_shape[1] == b_shape[0] else None
    try:
        got = (np.zeros(a_shape) @ np.zeros(b_shape)).shape
    except ValueError:
        got = None
    check(got == expected, f"NumPy shape {got} khác {expected}")
    return expected
