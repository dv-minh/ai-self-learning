"""Registry các dạng bài tập Phần 1 (Đại số tuyến tính)."""
from importlib import import_module

from ._base import TopicSpec

# Module cũ (bài 1.5): mỗi module có một SPEC
_SINGLE = [
    "matmul_shape", "matmul_entry", "matmul_full", "matvec",
    "matmul_chain_shape", "matmul_noncommute", "hadamard_vs_matmul",
]
# Module mới: mỗi module có danh sách SPECS
_MULTI = [
    "l1_1_vectors", "l1_2_dot", "l1_3_matrix", "l1_4_matops", "l1_6_special",
    "l1_7_gauss", "l1_8_det", "l1_9_inverse", "l1_10_space", "l1_11_maps", "l1_12_eigen",
    "l1_13_symmetric", "l1_14_svd", "l1_15_ml",
]

SPECS: dict[str, TopicSpec] = {}


def _register(spec: TopicSpec) -> None:
    if spec.type in SPECS:
        raise ValueError(f"trùng mã dạng: {spec.type}")
    SPECS[spec.type] = spec


for _name in _SINGLE:
    _register(import_module(f".{_name}", __package__).SPEC)
for _name in _MULTI:
    try:
        _mod = import_module(f".{_name}", __package__)
    except ModuleNotFoundError as e:  # module chưa viết xong (đang trong GĐ2)
        if e.name and e.name.endswith(_name):
            continue
        raise
    for _spec in _mod.SPECS:
        _register(_spec)
