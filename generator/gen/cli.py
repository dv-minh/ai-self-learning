"""python -m gen build|schema (docs/spec.md mục 5)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from importlib import import_module

from .schema import Exercise, ExerciseBank, json_schema

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "public" / "exercises"
# mã phần -> (module chứa SPECS, tiêu đề). Module nào chưa viết thì bỏ qua.
PART_MODULES = {
    "linalg": ("gen.topics.linalg", "Phần 1. Đại số tuyến tính"),
    "numpy": ("gen.topics.np_part", "Phần 2. NumPy cho ML"),
    "calculus": ("gen.topics.calc", "Phần 3. Giải tích"),
    "probability": ("gen.topics.prob", "Phần 4. Xác suất – thống kê"),
    "optimization": ("gen.topics.opt", "Phần 5. Tối ưu"),
    "classical-ml": ("gen.topics.cml", "Phần 6. ML cổ điển"),
    "deep-learning": ("gen.topics.dl", "Phần 7. Deep Learning"),
    "computer-vision": ("gen.topics.cv", "Phần 8. Computer Vision"),
}


def load_parts() -> dict:
    parts = {}
    for key, (mod, _title) in PART_MODULES.items():
        try:
            parts[key] = import_module(mod).SPECS
        except ModuleNotFoundError as e:
            if e.name and mod.startswith(e.name) or e.name == mod:
                continue
            raise
    return parts


PARTS = load_parts()
BASE_SEED = 1


def prompt_hash(ex: Exercise) -> str:
    # thứ tự lựa chọn được trộn ngẫu nhiên nên không tính vào "đề khác nhau": so theo tập lựa chọn đã sắp xếp
    # Phần 5–8: thứ tự lựa chọn là ngẫu nhiên nên không tính là "đề khác nhau" (so theo tập lựa chọn sắp xếp).
    # Phần 1–4 giữ cách cũ để kho bài đã có không đổi (np-dtype vốn ít biểu thức, xem PROCESS.md).
    newer = ex.id.split(".")[0] in ("optimization", "classical-ml", "deep-learning", "computer-vision")
    choices = sorted(ex.choices) if (newer and ex.choices) else ex.choices
    key = json.dumps([ex.type, ex.prompt.model_dump(), choices], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode()).hexdigest()


def generate_bank(spec, count: int, level: int | None = None, base_seed: int = BASE_SEED) -> list[Exercise]:
    """Sinh `count` bài của một dạng, chia đều theo các mức; khử trùng lặp theo hash của đề."""
    levels = [level] if level else list(spec.levels)
    from .topics.linalg._base import rng_for

    out: list[Exercise] = []
    seen: set[str] = set()
    for idx, lv in enumerate(levels):
        want = count // len(levels) + (1 if idx < count % len(levels) else 0)
        got, seed, attempts = 0, base_seed, 0
        while got < want:
            attempts += 1
            if attempts > want * 200:
                raise RuntimeError(f"{spec.type} L{lv}: không đủ {want} bài khác nhau (chỉ được {got})")
            ex = spec.generate(rng_for(spec.type, lv, seed), lv, seed)
            seed += 1
            h = prompt_hash(ex)
            if h in seen:
                continue
            seen.add(h)
            out.append(ex)
            got += 1
    return out


def write_bank(part: str, spec, exercises: list[Exercise]) -> Path:
    path = OUT / part / spec.lesson / f"{spec.type}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    bank = ExerciseBank(lesson=spec.lesson, type=spec.type, exercises=exercises)
    path.write_text(bank.model_dump_json(indent=None) + "\n", encoding="utf-8")
    return path


LESSON_TITLES = {
    "1.1": "Vector", "1.2": "Tích vô hướng, góc, cosine similarity", "1.3": "Ma trận và ký hiệu",
    "1.4": "Cộng ma trận, nhân vô hướng", "1.5": "Nhân ma trận", "1.6": "Ma trận đặc biệt",
    "1.7": "Hệ phương trình và khử Gauss", "1.8": "Định thức", "1.9": "Ma trận nghịch đảo",
    "1.10": "Không gian vector và hạng", "1.11": "Ánh xạ tuyến tính", "1.12": "Trị riêng, vector riêng",
    "1.13": "Ma trận đối xứng và xác định dương", "1.14": "SVD, xấp xỉ hạng thấp, PCA", "1.15": "Ma trận trong ML",
}
PART_TITLES = {k: v[1] for k, v in PART_MODULES.items()}


def lesson_titles(part: str) -> dict:
    if part == "linalg":
        return LESSON_TITLES
    try:
        return import_module(PART_MODULES[part][0]).TITLES
    except (ModuleNotFoundError, AttributeError):
        return {}


def write_manifest() -> Path:
    """Danh mục kho bài cho trang /luyen-tap: phần → bài → dạng → mức, kích thước, số bài."""
    import re

    parts: dict = {}
    order = {k: i for i, k in enumerate(PART_MODULES)}
    for part_dir in sorted((p for p in OUT.iterdir() if p.is_dir()), key=lambda p: order.get(p.name, 99)):
        lessons: dict = {}
        for f in sorted(part_dir.glob("*/*.json")):
            bank = json.loads(f.read_text(encoding="utf-8"))
            exs = bank["exercises"]
            levels = sorted({e["level"] for e in exs})
            sizes = sorted({t for e in exs for t in e.get("tags", []) if re.fullmatch(r"\d+x\d+", t)})
            lessons.setdefault(bank["lesson"], {"title": lesson_titles(part_dir.name).get(bank["lesson"], bank["lesson"]), "types": {}})
            lessons[bank["lesson"]]["types"][bank["type"]] = {
                "levels": levels, "sizes": sizes, "count": len(exs),
                "perLevel": {str(lv): sum(1 for e in exs if e["level"] == lv) for lv in levels},
            }
        parts[part_dir.name] = {"title": PART_TITLES.get(part_dir.name, part_dir.name), "lessons": dict(sorted(lessons.items(), key=lambda kv: [int(x) for x in kv[0].split(".")]))}
    path = OUT / "manifest.json"
    path.write_text(json.dumps({"version": 1, "parts": parts}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def cmd_build(args) -> int:
    parts = PARTS if args.all else {args.part: PARTS[args.part]} if args.part else {"linalg": PARTS["linalg"]}
    total = 0
    for part, specs in parts.items():
        for t, spec in specs.items():
            if args.type and t != args.type:
                continue
            exs = generate_bank(spec, args.count, args.level)
            total += len(exs)
            if args.print:
                for ex in exs:
                    print(ex.model_dump_json(indent=2, exclude_none=True))
            else:
                p = write_bank(part, spec, exs)
                print(f"{t}: {len(exs)} bài, 100% qua kiểm chứng -> {p.relative_to(ROOT)}")
    print(f"Tổng: {total} bài.")
    if not args.print:
        print(f"manifest -> {write_manifest().relative_to(ROOT)}")
    if not args.type and not args.print:
        from .examples import build_all

        paths = build_all()
        print(f"ví dụ bài học: {len(paths)} partial MDX -> site/src/generated/")
    return 0


def cmd_schema(args) -> int:
    path = ROOT / "site" / "src" / "types" / "exercise.schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Đã ghi {path.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định không phải UTF-8
    ap =argparse.ArgumentParser(prog="gen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--part")
    b.add_argument("--all", action="store_true")
    b.add_argument("--type")
    b.add_argument("--level", type=int)
    b.add_argument("--count", type=int, default=200)
    b.add_argument("--print", action="store_true")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("schema")
    s.set_defaults(fn=cmd_schema)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
