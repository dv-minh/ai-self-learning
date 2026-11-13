"""Tiện ích chung cho các module dạng bài tập."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from ...schema import Answer, Check, Exercise, Prompt, Step


def rng_for(type_: str, level: int, seed: int) -> random.Random:
    return random.Random(f"{type_}:{level}:{seed}")


@dataclass(frozen=True)
class TopicSpec:
    type: str
    lesson: str
    levels: tuple[int, ...]
    generate: Callable[[random.Random, int, int], Exercise]


def build(
    *,
    type_: str,
    lesson: str,
    level: int,
    seed: int,
    prompt: Prompt,
    answer: Answer,
    hints: list[str],
    steps: list[Step],
    verified: list[str],
    tags: list[str] | None = None,
    choices: list[str] | None = None,
    check: Check | None = None,
    part: str = "linalg",
) -> Exercise:
    return Exercise(
        id=f"{part}.{type_}.L{level}.{seed:06d}",
        seed=seed,
        lesson=lesson,
        type=type_,
        level=level,
        tags=tags or [],
        prompt=prompt,
        answer=answer,
        check=check or Check(mode="exact"),
        choices=choices,
        hints=hints,
        steps=steps,
        verified=verified,
    )


def shape_answer(shape: tuple[int, int] | None) -> Answer:
    if shape is None:
        return Answer(kind="shape", undefined=True)
    return Answer(kind="shape", rows=shape[0], cols=shape[1])


def shape_tex(shape: tuple[int, int]) -> str:
    return f"{shape[0]} \\times {shape[1]}"


# ---------- Tiện ích dựng đáp án ----------
from sympy import Matrix  # noqa: E402

from ...core import latex as _L  # noqa: E402


def vec_ans(v: Matrix) -> Answer:
    return Answer(kind="vector", rows=v.rows, cols=1, entries=_L.entries(v))


def mat_ans(m: Matrix) -> Answer:
    return Answer(kind="matrix", rows=m.rows, cols=m.cols, entries=_L.entries(m))


def num_ans(x) -> Answer:
    return Answer(kind="number", value=_L.rat_str(x))


def bool_ans(flag: bool) -> Answer:
    return Answer(kind="boolean", value="true" if flag else "false")


def size_tag(m: Matrix) -> str:
    return f"{m.rows}x{m.cols}"


def shuffled_choices(rng: random.Random, correct: list[str], wrong: list[str]) -> tuple[list[str], list[int]]:
    """Trộn lựa chọn, trả về (choices, chỉ số đúng). Loại trùng lặp nội dung."""
    items = [(c, True) for c in correct] + [(w, False) for w in wrong if w not in correct]
    seen, uniq = set(), []
    for c, ok in items:
        if c not in seen:
            seen.add(c)
            uniq.append((c, ok))
    rng.shuffle(uniq)
    return [c for c, _ in uniq], [i for i, (_, ok) in enumerate(uniq) if ok]
