"""Ví dụ số cho bài học. `build_all()` sinh mọi partial MDX vào site/src/generated/."""
from importlib import import_module

LESSON_MODULES = ["lesson_1_1", "lesson_1_2", "lesson_1_3", "lesson_1_4", "lesson_1_5",
                  "lesson_1_6", "lesson_1_7", "lesson_1_8", "lesson_1_9", "lesson_1_10", "lesson_1_11",
                  "lesson_1_12", "lesson_1_13", "lesson_1_14", "lesson_1_15"]


# Phần 2–4: mỗi module build() cho cả phần; chưa viết thì bỏ qua
PART_MODULES = ["part2", "part3", "part4", "part5", "part6", "part7", "part8"]


def build_all() -> list:
    paths = []
    for name in LESSON_MODULES:
        paths += import_module(f".{name}", __package__).build()
    for name in PART_MODULES:
        try:
            paths += import_module(f".{name}", __package__).build()
        except ModuleNotFoundError as e:
            if e.name and e.name.endswith(name):
                continue
            raise
    return paths
