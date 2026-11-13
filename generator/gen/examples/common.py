r"""Hạ tầng chung cho ví dụ số trong bài học.

Mỗi bài học có một module trong gen/examples/ khai báo EXAMPLES = {tên: hàm() -> chuỗi MDX}.
Kết quả ghi vào site/src/generated/lesson-<bài>/<tên>.mdx để bài học import. Không có số nào gõ tay trong MDX.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from sympy import Matrix

from ..core.nice import int_norm_vector, nontrivial_matrix, rand_vector
from ..schema import Step

ROOT = Path(__file__).resolve().parents[3]
GENERATED = ROOT / "site" / "src" / "generated"


def rng(seed: int) -> random.Random:
    return random.Random(seed)


def M(seed: int, r: int, c: int, lo: int = -4, hi: int = 4) -> Matrix:
    return nontrivial_matrix(rng(seed), r, c, lo, hi)


def V(seed: int, n: int, lo: int = -5, hi: int = 5) -> Matrix:
    return rand_vector(rng(seed), n, lo, hi)


def NV(seed: int, n: int, hi: int = 12) -> Matrix:
    """Vector có chuẩn L2 nguyên."""
    return int_norm_vector(rng(seed), n, hi)


def display(*lines: str) -> str:
    return "\n\n".join(f"$$\n{ln}\n$$" for ln in lines)


def steps_md(steps: list[Step]) -> str:
    out = []
    for i, s in enumerate(steps, 1):
        out.append(f"{i}. {s.text}")
        if s.latex:
            out.append(f"\n   $$\n   {s.latex}\n   $$\n")
    return "\n".join(out)


def write_examples(lesson: str, examples: dict[str, Callable[[], str]]) -> list[Path]:
    out_dir = GENERATED / f"lesson-{lesson.replace('.', '-')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in examples.items():
        p = out_dir / f"{name}.mdx"
        p.write_text(fn().strip() + "\n", encoding="utf-8")
        paths.append(p)
    return paths
