"""Registry các dạng bài tập Phần 5 (Tối ưu). Mọi phép tính bằng Fraction; kiểm chứng độc lập bằng SymPy, NumPy hoặc liệt kê."""
from importlib import import_module

from ..linalg._base import TopicSpec

TITLES = {
    "5.1": "Hàm lồi", "5.2": "Gradient descent", "5.3": "SGD và mini-batch", "5.4": "Momentum", "5.5": "Adam, RMSProp, AdaGrad",
    "5.6": "Lịch learning rate", "5.7": "Regularization L1, L2 và weight decay",
}

_MODULES = ["o5_1_convex", "o5_2_gd", "o5_3_sgd", "o5_4_5_adaptive", "o5_6_7_lr_reg"]

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
