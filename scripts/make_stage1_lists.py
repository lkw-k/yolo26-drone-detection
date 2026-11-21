"""Build stage 1 image lists, dropping images listed in leakage exclude.txt."""

from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/stage1"))
    ap.add_argument("--exclude", type=Path, default=Path("data/stage1/leakage/exclude.txt"))
    ap.add_argument("--allow-missing-exclude", action="store_true", help="only for throughput trials")
    args = ap.parse_args()

    if args.exclude.exists():
        excl = {Path(p).resolve() for p in args.exclude.read_text(encoding="utf-8").split("\n") if p.strip()}
    elif args.allow_missing_exclude:
        excl = set()
        print("WARNING: no leakage exclude list; lists are for trials only")
    else:
        raise SystemExit(f"missing {args.exclude}; run scripts/leakage_check.py first")

    lists = args.root / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "seraphim_train.txt"), ("test", "seraphim_val.txt")):
        imgs = sorted(p.resolve() for p in (args.root / "seraphim" / split / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
        keep = [p for p in imgs if p not in excl]
        (lists / name).write_text("\n".join(p.as_posix() for p in keep) + "\n", encoding="utf-8")
        print(f"{name}: {len(keep)} kept, {len(imgs) - len(keep)} excluded")


if __name__ == "__main__":
    main()
