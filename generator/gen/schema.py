"""Nguồn chân lý của định dạng bài tập (docs/spec.md mục 5).

Kiểu TypeScript của site được sinh tự động từ JSON Schema của file này.
Mọi số hữu tỉ lưu dạng chuỗi "p/q" đã rút gọn.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerKind = Literal[
    "number", "vector", "matrix", "shape", "choice", "multi", "boolean", "expression", "solution-set"
]
CheckMode = Literal["exact", "eigvec", "span", "solution-set", "numeric"]


class _Base(BaseModel):
    """Mọi trường (kể cả có giá trị mặc định) luôn xuất hiện trong JSON, nên là bắt buộc ở phía TypeScript."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class Prompt(_Base):
    text: str = Field(description="Đề bài, có thể chứa $...$ (KaTeX).")
    display: list[str] = Field(default_factory=list, description="Công thức display, mỗi phần tử một khối.")
    code: str | None = Field(default=None, description="Đoạn code NumPy (Phần 2), chỉ để đọc.")


class Answer(_Base):
    kind: AnswerKind
    # number: value = "p/q"; boolean: value = "true"/"false"
    value: str | None = None
    # vector / matrix / shape
    rows: int | None = None
    cols: int | None = None
    entries: list[list[str]] | None = None
    # shape: true nếu phép tính không xác định (không có kích thước)
    undefined: bool = False
    # choice / multi: chỉ số các lựa chọn đúng
    correct: list[int] | None = None
    # solution-set: nghiệm riêng và cơ sở không gian null (mỗi vector là danh sách "p/q")
    particular: list[str] | None = None
    basis: list[list[str]] | None = None
    # expression: value = biểu thức chuẩn (cú pháp nhập: + - * / ^, ln, exp, sin, cos, sqrt); variables = tên biến.
    # Người học nhập biểu thức; bộ chấm so giá trị số tại các điểm mẫu cố định (docs/spec.md mục 5).
    variables: list[str] | None = None


class Check(_Base):
    mode: CheckMode = "exact"
    tolerance: float | None = Field(default=None, description="Sai số tương đối cho mode numeric.")
    # solution-set: hệ Ax = b để thế nghiệm của người học vào
    a: list[list[str]] | None = None
    b: list[str] | None = None
    # eigvec: ma trận A (trường a) và trị riêng lam; chấp nhận mọi v ≠ 0 thỏa Av = lam·v
    lam: str | None = None


class Step(_Base):
    text: str
    latex: str | None = None


class Exercise(_Base):
    id: str
    seed: int
    lesson: str
    type: str
    level: int = Field(ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    prompt: Prompt
    answer: Answer
    check: Check = Field(default_factory=Check)
    choices: list[str] | None = None
    hints: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)


class ExerciseBank(_Base):
    """Một file kho: tất cả bài của một dạng."""

    version: int = 1
    lesson: str
    type: str
    exercises: list[Exercise]


def json_schema() -> dict:
    # mode="serialization": trường có giá trị mặc định luôn xuất hiện trong JSON nên là bắt buộc ở phía TypeScript
    return ExerciseBank.model_json_schema(mode="serialization")
