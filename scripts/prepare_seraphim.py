"""Extract seraphim zips into a standard YOLO layout and validate labels.

Output layout:
    data/stage1/seraphim/{train,test}/{images,labels}/

Checks: every image has a label file, every label line has 5 fields,
class id == 0 (drone), normalized coords in [0, 1], w/h > 0.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract(raw: Path, out: Path) -> None:
    for split in ("train", "test"):
        for kind in ("images", "labels"):
            dst = out / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            for zp in sorted((raw / split / kind).glob("*.zip")):
                print(f"extract {zp} -> {dst}")
                with zipfile.ZipFile(zp) as z:
                    for m in z.infolist():
                        if m.is_dir():
                            continue
                        # Flatten any inner folders: keep basename only.
                        name = Path(m.filename).name
                        target = dst / name
                        if target.exists():
                            continue
                        with z.open(m) as src, open(target, "wb") as f:
                            f.write(src.read())


def fix_polygons(out: Path) -> list[str]:
    """Convert polygon (segmentation) lines into their bounding boxes."""
    fixed = []
    for split in ("train", "test"):
        for lp in (out / split / "labels").glob("*.txt"):
            lines = lp.read_text().splitlines()
            if all(len(l.split()) in (0, 5) for l in lines):
                continue
            new = []
            for l in lines:
                parts = l.split()
                if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                    xy = [min(max(float(v), 0.0), 1.0) for v in parts[1:]]
                    xs, ys = xy[0::2], xy[1::2]
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    new.append(f"{parts[0]} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}")
                else:
                    new.append(l)
            lp.write_text("\n".join(new) + "\n")
            fixed.append(str(lp))
    return fixed


def validate(out: Path) -> dict:
    report = {}
    for split in ("train", "test"):
        imgs = {p.stem: p for p in (out / split / "images").iterdir() if p.suffix.lower() in IMG_EXTS}
        lbls = {p.stem: p for p in (out / split / "labels").glob("*.txt")}
        classes: Counter = Counter()
        bad_lines, empty, n_boxes = 0, 0, 0
        for stem, lp in lbls.items():
            lines = [l for l in lp.read_text().splitlines() if l.strip()]
            if not lines:
                empty += 1
            for l in lines:
                parts = l.split()
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                c, *xywh = parts
                classes[c] += 1
                n_boxes += 1
                x, y, w, h = map(float, xywh)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad_lines += 1
        report[split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "images_without_label": len(imgs.keys() - lbls.keys()),
            "labels_without_image": len(lbls.keys() - imgs.keys()),
            "empty_label_files": empty,
            "boxes": n_boxes,
            "class_counts": dict(classes),
            "bad_lines": bad_lines,
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/seraphim"))
    ap.add_argument("--out", type=Path, default=Path("data/stage1/seraphim"))
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()
    if not args.skip_extract:
        extract(args.raw, args.out)
    fixed = fix_polygons(args.out)
    print(f"polygon label files converted to boxes: {len(fixed)} {fixed}")
    rep = validate(args.out)
    rep["polygon_files_fixed"] = fixed
    print(json.dumps(rep, indent=2))
    (args.out / "validation_report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
