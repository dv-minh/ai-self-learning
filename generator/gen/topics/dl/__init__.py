"""Registry các dạng bài tập Phần 7 (Deep Learning). Mọi phép tính bằng Fraction; kiểm chứng độc lập bằng NumPy hoặc SymPy."""
from importlib import import_module

from ..linalg._base import TopicSpec

TITLES = {
    "7.1": "Perceptron và MLP", "7.2": "Hàm kích hoạt, softmax, cross-entropy", "7.3": "Lan truyền ngược (backpropagation)",
    "7.4": "Khởi tạo trọng số, BatchNorm, LayerNorm", "7.5": "Dropout, residual connection, embedding", "7.6": "Attention và Transformer",
    "7.7": "Fine-tuning, LoRA và đọc code PyTorch",
}

_MODULES = ["d7_1_3_nets", "d7_4_5_norm", "d7_6_7_transformer"]

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
