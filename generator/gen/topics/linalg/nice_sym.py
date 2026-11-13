r"""Kho ma trận đối xứng nguyên có trị riêng nguyên (dùng cho bài 1.13).

Quét mọi ma trận đối xứng với phần tử trong [-r, r] bằng NumPy (chỉ để TÌM ứng viên),
rồi mỗi bài khi dùng đều được kiểm lại chính xác bằng SymPy (đa thức đặc trưng, trị riêng).
"""
from __future__ import annotations

import random
from functools import lru_cache
from itertools import product

import numpy as np
from sympy import Matrix


def _unpack(entries: tuple[int, ...], n: int) -> Matrix:
    m = Matrix.zeros(n, n)
    k = 0
    for i in range(n):
        for j in range(i, n):
            m[i, j] = m[j, i] = entries[k]
            k += 1
    return m


@lru_cache(maxsize=None)
def sym_pool(n: int, r: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Danh sách (entries, trị riêng nguyên tăng dần) của ma trận đối xứng n×n, không đường chéo, không hàng 0."""
    k = n * (n + 1) // 2
    combos = np.array(list(product(range(-r, r + 1), repeat=k)), dtype=np.int64)
    mats = np.zeros((len(combos), n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            mats[:, i, j] = combos[:, idx]
            mats[:, j, i] = combos[:, idx]
            idx += 1
    eig = np.linalg.eigvalsh(mats)
    near = np.all(np.abs(eig - np.round(eig)) < 1e-9, axis=1)
    out = []
    for row, e in zip(combos[near], eig[near]):
        a = _unpack(tuple(int(x) for x in row), n)
        offdiag = any(a[i, j] != 0 for i in range(n) for j in range(i + 1, n))
        zero_row = any(all(a[i, j] == 0 for j in range(n)) for i in range(n))
        if offdiag and not zero_row:
            out.append((tuple(int(x) for x in row), tuple(int(round(x)) for x in e)))
    return tuple(out)


def pick_sym(rng: random.Random, n: int, r: int, want=None) -> tuple[Matrix, list[int]]:
    """Chọn ngẫu nhiên một (A, trị riêng) thỏa điều kiện want(trị riêng) -> bool."""
    pool = sym_pool(n, r)
    cands = [p for p in pool if want is None or want(p[1])]
    if not cands:
        raise RuntimeError("kho ma trận đối xứng rỗng với điều kiện này")
    entries, eig = rng.choice(cands)
    return _unpack(entries, n), list(eig)
