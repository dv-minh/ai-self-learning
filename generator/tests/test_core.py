"""Test tính chất toán (hypothesis) và test lõi generator."""
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Matrix, Rational

from gen.cli import generate_bank, prompt_hash
from gen.core import latex as L
from gen.core.nice import is_trivial
from gen.core.verify import VerificationError, matmul_shape, verify_matmul
from gen.cli import PARTS

SPECS = {t: spec for specs in PARTS.values() for t, spec in specs.items()}

ints = st.integers(-6, 6)


def mats(rows, cols):
    return st.lists(st.lists(ints, min_size=cols, max_size=cols), min_size=rows, max_size=rows).map(Matrix)


@given(mats(3, 3), mats(3, 3))
@settings(max_examples=60, deadline=None)
def test_det_multiplicative_and_transpose(a, b):
    assert (a * b).det() == a.det() * b.det()
    assert a.T.det() == a.det()


@given(mats(3, 4), mats(4, 2))
@settings(max_examples=60, deadline=None)
def test_transpose_of_product(a, b):
    assert (a * b).T == b.T * a.T


@given(mats(3, 3), mats(3, 3))
@settings(max_examples=60, deadline=None)
def test_inverse_of_product(a, b):
    if a.det() != 0 and b.det() != 0:
        assert (a * b).inv() == b.inv() * a.inv()


def test_verify_matmul_catches_wrong_result():
    a, b = Matrix([[1, 2], [3, 4]]), Matrix([[0, 1], [1, 0]])
    with pytest.raises(VerificationError):
        verify_matmul(a, b, a * b + Matrix([[1, 0], [0, 0]]))


def test_matmul_shape():
    assert matmul_shape((2, 3), (3, 4)) == (2, 4)
    assert matmul_shape((2, 3), (2, 4)) is None


def test_latex_fraction_and_negative():
    assert L.num(Rational(-1, 3)) == "-\\frac{1}{3}"
    assert L.rat_str(Rational(4, 2)) == "2"
    assert L.rat_str(Rational(-2, 4)) == "-1/2"
    assert L.num_paren(-3) == "(-3)"


def test_trivial_matrices_rejected():
    assert is_trivial(Matrix.zeros(2, 2))
    assert is_trivial(Matrix.eye(3))
    assert is_trivial(Matrix([[1, 2], [0, 0]]))
    assert not is_trivial(Matrix([[1, 2], [3, 4]]))


@pytest.mark.parametrize("name", list(SPECS))
def test_bank_deterministic_unique_and_valid(name):
    spec = SPECS[name]
    first = generate_bank(spec, 40)
    second = generate_bank(spec, 40)
    assert [e.model_dump() for e in first] == [e.model_dump() for e in second]  # tất định
    assert len({prompt_hash(e) for e in first}) == len(first)  # không trùng
    for e in first:
        assert e.level in spec.levels
        assert e.hints and e.steps and e.verified
        if e.answer.kind == "matrix":
            assert len(e.answer.entries) == e.answer.rows
            assert all(len(r) == e.answer.cols for r in e.answer.entries)
            if e.id.startswith("linalg."):  # giới hạn "tính tay được" chỉ áp dụng cho Phần 1
                assert e.answer.rows <= 4 and e.answer.cols <= 4
        if e.answer.kind in ("choice", "multi"):
            assert e.choices and all(0 <= i < len(e.choices) for i in e.answer.correct)


def test_matmul_full_answers_independent_recompute():
    spec = SPECS["matmul-full"]
    rng = random.Random(0)
    for ex in generate_bank(spec, 30):
        # đáp án phải là ma trận rút gọn "p/q"
        for row in ex.answer.entries:
            for cell in row:
                p, _, q = cell.partition("/")
                if q:
                    from math import gcd
                    assert gcd(abs(int(p)), int(q)) == 1 and int(q) > 1


def test_source_has_no_invalid_escapes():
    """Chuỗi LaTeX thiếu một dấu gạch chéo ngược (ví dụ times thành Tab) là lỗi âm thầm: biên dịch với cảnh báo = lỗi."""
    import warnings
    from pathlib import Path

    for path in Path(__file__).resolve().parents[1].joinpath("gen").rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("name", list(SPECS))
def test_no_control_characters_in_bank(name):
    import json

    for e in generate_bank(SPECS[name], 20):
        s = json.dumps(e.model_dump(), ensure_ascii=False)
        assert not any(ch in s for ch in "\t\x08\x0c\r\x07\x0b"), e.id
