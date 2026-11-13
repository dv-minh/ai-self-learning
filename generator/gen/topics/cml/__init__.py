"""Registry các dạng bài tập Phần 6 (ML cổ điển). Mọi phép tính bằng Fraction; kiểm chứng độc lập bằng NumPy, SymPy hoặc liệt kê."""
from importlib import import_module

from ..linalg._base import TopicSpec

TITLES = {
    "6.1": "Hồi quy tuyến tính", "6.2": "Hồi quy logistic và softmax", "6.3": "Overfitting, bias–variance, cross-validation",
    "6.4": "Độ đo phân loại", "6.5": "K láng giềng gần nhất (KNN)", "6.6": "K-means", "6.7": "Cây quyết định và random forest",
    "6.8": "Gradient boosting", "6.9": "SVM",
}

_MODULES = ["m6_1_2_regression", "m6_3_4_eval", "m6_5_6_7_instance", "m6_8_9_boost_svm"]

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
