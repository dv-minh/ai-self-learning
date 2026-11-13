"""Registry các dạng bài tập Phần 3 (Giải tích). Đạo hàm do SymPy tính và được kiểm bằng sai phân hữu hạn."""
from importlib import import_module

from ..linalg._base import TopicSpec

TITLES = {
    "3.1": "Đạo hàm", "3.2": "Quy tắc chuỗi (chain rule)", "3.3": "Đạo hàm riêng và gradient", "3.4": "Jacobian",
    "3.5": "Hessian", "3.6": "Đạo hàm theo vector và ma trận", "3.7": "Đồ thị tính toán và vi phân tự động",
}

_MODULES = ["c3_1_3_derivatives", "c3_3_7_multivariable"]

SPECS: dict[str, TopicSpec] = {}
for _name in _MODULES:
    try:
        _mod = import_module(f".{_name}", __package__)
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(_name):
            continue
        raise
    for _spec in _mod.SPECS:
        if _spec.type in SPECS:
            raise ValueError(f"trùng mã dạng: {_spec.type}")
        SPECS[_spec.type] = _spec
