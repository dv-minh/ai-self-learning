"""Tiện ích cho Phần 7 (Deep Learning) và Phần 8 (Computer Vision): phân số chính xác, kiểm chứng độc lập bằng NumPy hoặc SymPy."""
from __future__ import annotations

from fractions import Fraction as F
from math import isqrt

from ...core import latex as L
from ...core.verify import check
from ...schema import Answer, Check, Prompt, Step
from ..linalg._base import TopicSpec, build, shuffled_choices
from ..opt._o import NUM_HINT, VEC_HINT, choice_exercise, fp, fs, ft, lf, mat_tex, nice, num_ans, pick, vec_ans, vt


def dl(**kw):
    return build(part="deep-learning", **kw)


def cv(**kw):
    return build(part="computer-vision", **kw)


def mat_ans(rows) -> Answer:
    return Answer(kind="matrix", rows=len(rows), cols=len(rows[0]), entries=[[fs(v) for v in r] for r in rows])


def sqrt_q(v: F):
    """Căn bậc hai hữu tỉ chính xác hoặc None."""
    v = F(v)
    if v < 0:
        return None
    a, b = isqrt(v.numerator), isqrt(v.denominator)
    return F(a, b) if a * a == v.numerator and b * b == v.denominator else None


def mm(a, b):
    return [[sum(F(a[i][k]) * F(b[k][j]) for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def mv(a, v):
    return [sum(F(a[i][k]) * F(v[k]) for k in range(len(v))) for i in range(len(a))]


def relu(x):
    return x if x > 0 else F(0)


def rand_mat(rng, r, c, lo=-2, hi=2):
    return [[F(rng.randint(lo, hi)) for _ in range(c)] for _ in range(r)]


def rand_vec(rng, n, lo=-3, hi=3):
    return [F(rng.randint(lo, hi)) for _ in range(n)]


__all__ = ["dl", "cv", "mat_ans", "sqrt_q", "mm", "mv", "relu", "rand_mat", "rand_vec", "F", "L", "check", "Answer", "Check", "Prompt", "Step", "TopicSpec",
           "shuffled_choices", "NUM_HINT", "VEC_HINT", "choice_exercise", "fp", "fs", "ft", "lf", "mat_tex", "nice", "num_ans", "pick", "vec_ans", "vt"]
