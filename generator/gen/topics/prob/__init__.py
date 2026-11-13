"""Registry các dạng bài tập Phần 4 (Xác suất – thống kê). Mọi đáp án là phân số chính xác, kiểm chứng bằng liệt kê/đếm."""
from importlib import import_module

from ..linalg._base import TopicSpec

TITLES = {
    "4.1": "Xác suất cơ bản và xác suất có điều kiện", "4.2": "Công thức Bayes và Naive Bayes", "4.3": "Kỳ vọng, phương sai, hiệp phương sai",
    "4.4": "Phân phối: Bernoulli, nhị thức, chuẩn", "4.5": "MLE và MAP", "4.6": "Entropy, cross-entropy, KL divergence",
}

_MODULES = ["p4_1_2_basic_bayes", "p4_3_4_rv_dist", "p4_5_6_mle_info"]

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
