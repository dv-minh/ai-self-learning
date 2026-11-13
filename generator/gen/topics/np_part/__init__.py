"""Registry các dạng bài tập Phần 2 (NumPy cho ML). Đáp án do NumPy thật tính lúc build."""
from importlib import import_module

from ..linalg._base import TopicSpec

TITLES = {
    "2.1": "ndarray, shape, dtype", "2.2": "Chỉ số và cắt lát", "2.3": "@, * và .T", "2.4": "Rút gọn theo trục",
    "2.5": "Broadcasting", "2.6": "reshape, stack, concatenate", "2.7": "np.linalg", "2.8": "Số thực dấu phẩy động",
    "2.9": "Vector hóa công thức ML",
}

_MODULES = ["p2_1_2_arrays", "p2_3_4_ops", "p2_5_6_shapes", "p2_7_8_9_linalg_float_ml"]

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
