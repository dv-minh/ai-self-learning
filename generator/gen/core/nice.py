"""Dựng ma trận "đẹp" để tính tay được (docs/spec.md mục 5)."""
from __future__ import annotations

import random

from sympy import Matrix, Rational

SIMPLE_FRACTIONS = [Rational(1, 2), Rational(-1, 2), Rational(1, 3), Rational(-2, 3), Rational(3, 2), Rational(1, 4)]


def level_range(level: int) -> tuple[int, int]:
    """Khoảng giá trị phần tử theo mức (mục 4)."""
    return (-9, 9) if level in (1, 3, 4, 5) else (-5, 5)


def rand_matrix(
    rng: random.Random, rows: int, cols: int, lo: int, hi: int, frac_prob: float = 0.0
) -> Matrix:
    """Ma trận ngẫu nhiên; frac_prob > 0 cho phép xen vài phân số đơn giản."""
    def one():
        if frac_prob and rng.random() < frac_prob:
            return rng.choice(SIMPLE_FRACTIONS)
        return Rational(rng.randint(lo, hi))

    return Matrix(rows, cols, lambda i, j: one())


def is_trivial(m: Matrix) -> bool:
    """Loại ma trận toàn 0, hàng/cột 0 ngoài ý muốn, hoặc ma trận đơn vị."""
    if all(x == 0 for x in m):
        return True
    for i in range(m.rows):
        if all(m[i, j] == 0 for j in range(m.cols)):
            return True
    for j in range(m.cols):
        if all(m[i, j] == 0 for i in range(m.rows)):
            return True
    if m.rows == m.cols and m == Matrix.eye(m.rows):
        return True
    return False


def nontrivial_matrix(
    rng: random.Random, rows: int, cols: int, lo: int, hi: int, frac_prob: float = 0.0
) -> Matrix:
    for _ in range(1000):
        m = rand_matrix(rng, rows, cols, lo, hi, frac_prob)
        if not is_trivial(m):
            return m
    raise RuntimeError("không dựng được ma trận không tầm thường")


# ---------- Vector đẹp ----------
from functools import lru_cache
from math import isqrt


@lru_cache(maxsize=None)
def _int_norm_tuples(n: int, hi: int) -> tuple[tuple[int, ...], ...]:
    """Các bộ (a1 <= ... <= an) không âm, ít nhất 2 phần tử khác 0, có chuẩn L2 nguyên."""
    from itertools import combinations_with_replacement

    out = []
    for t in combinations_with_replacement(range(0, hi + 1), n):
        s = sum(x * x for x in t)
        if s and isqrt(s) ** 2 == s and sum(1 for x in t if x) >= 2:
            out.append(t)
    return tuple(out)


def int_norm_vector(rng: random.Random, n: int, hi: int = 12, scale_max: int = 1) -> Matrix:
    """Vector nguyên (cột) có chuẩn L2 là số nguyên: ngẫu nhiên hoán vị, đổi dấu, nhân hệ số."""
    t = list(rng.choice(_int_norm_tuples(n, hi)))
    rng.shuffle(t)
    k = rng.randint(1, scale_max)
    return Matrix(n, 1, [k * x * rng.choice([-1, 1]) if x else 0 for x in t])


def rand_vector(rng: random.Random, n: int, lo: int, hi: int, frac_prob: float = 0.0) -> Matrix:
    for _ in range(1000):
        v = rand_matrix(rng, n, 1, lo, hi, frac_prob)
        if any(x != 0 for x in v):
            return v
    raise RuntimeError("không dựng được vector")
