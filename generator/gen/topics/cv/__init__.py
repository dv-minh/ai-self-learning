"""Registry các dạng bài tập Phần 8 (Computer Vision). Mọi phép tính bằng Fraction; kiểm chứng độc lập bằng NumPy hoặc liệt kê."""
from importlib import import_module

from ..linalg._base import TopicSpec

TITLES = {
    "8.1": "Ảnh là tensor, tích chập, padding, stride", "8.2": "Pooling và trường tiếp nhận", "8.3": "CNN và ResNet",
    "8.4": "Vision Transformer và patch embedding", "8.5": "Augmentation và transfer learning", "8.6": "Phát hiện vật thể (detection)",
    "8.7": "Phân đoạn (segmentation)", "8.8": "Self-supervised, contrastive và CLIP", "8.9": "Mô hình khuếch tán (diffusion)",
}

_MODULES = ["c8_1_3_cnn", "c8_4_9_modern"]

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
