r"""Ma trận trực giao hữu tỉ bằng biến đổi Cayley (docs/spec.md mục 5).

S phản đối xứng nguyên nhỏ  =>  Q = (I - S)(I + S)^{-1} thỏa Q^T Q = I, các phần tử hữu tỉ.
Lọc những Q có mẫu số lớn để tính tay được (mẫu số <= denom_max).
"""
from __future__ import annotations

import random

from sympy import Matrix, Rational, eye

from ...core.verify import check


def skew(rng: random.Random, n: int, mx: int = 3) -> Matrix:
    s = Matrix.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.randint(-mx, mx)
            s[i, j], s[j, i] = v, -v
    return s


def cayley(rng: random.Random, n: int, denom_max: int = 25, nontrivial: bool = True) -> Matrix:
    for _ in range(1000):
        s = skew(rng, n)
        if nontrivial and all(x == 0 for x in s):
            continue
        q = (eye(n) - s) * (eye(n) + s).inv()
        if max(Rational(x).q for x in q) <= denom_max:
            check(q.T * q == eye(n), "Cayley: Q^T Q = I")
            return q
    raise RuntimeError("không dựng được Q hữu tỉ")


def signed_perm(rng: random.Random, n: int) -> Matrix:
    perm = list(range(n))
    rng.shuffle(perm)
    return Matrix(n, n, lambda i, j: rng.choice([-1, 1]) if perm[i] == j else 0)


def rand_orth(rng: random.Random, n: int, rational_prob: float = 0.6) -> Matrix:
    """Ma trận trực giao hữu tỉ: hoán vị có dấu, hoặc (xác suất rational_prob) quay 3-4-5 / Cayley mẫu số nhỏ."""
    q = signed_perm(rng, n)
    if rng.random() < rational_prob:
        if n == 2:
            a, b = rng.choice([(3, 4), (4, 3)])
            r = Matrix([[a, -b], [b, a]]) / 5
        else:
            r = cayley(rng, n, 15)
        q = r * q if rng.random() < 0.5 else q * r
    check(q.T * q == eye(n), "rand_orth")
    return q
