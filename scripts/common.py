"""Shared paths, class map, label helpers, and manifest I/O for the conversion scripts."""

import os
import random
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw

pl.Config.set_tbl_formatting("ASCII_MARKDOWN")  # cp949 Windows consoles cannot print box-drawing chars
pl.Config.set_tbl_hide_dataframe_shape(True)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
YOLO = DATA / "yolo"
MANIFESTS = DATA / "manifests"
REPORTS = DATA / "reports"

CLASSES = ["drone", "bird", "airplane", "helicopter"]
MAX_SIDE = 1920  # longer side cap for stored images (spec §1)
SEED = 0

MANIFEST_COLUMNS = [
    "image", "source", "group", "split", "viewpoint", "role",
    "n_drone", "n_bird", "n_airplane", "n_helicopter", "w", "h",
]


def yolo_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    """Pixel xyxy -> normalized (cx, cy, bw, bh), clipped to the image. None if <1px after clipping."""
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def clip_yolo_line(cx: float, cy: float, bw: float, bh: float, w: int, h: int):
    """Re-clip an existing normalized box through pixel space."""
    return yolo_box((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, w, h)


def write_label(path: Path, boxes: list[tuple[int, tuple]]) -> None:
    """boxes: [(class_id, (cx, cy, bw, bh)), ...]; an empty list writes an empty file (background image)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for c, b in boxes))


def label_path(image: Path) -> Path:
    """Ultralytics convention: .../images/... -> .../labels/....txt"""
    parts = list(image.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def link_image(src: Path, dst: Path) -> None:
    """Hardlink an unchanged image from raw (same volume, no extra disk)."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def save_resized(src: Path, dst: Path, max_side: int = MAX_SIDE) -> tuple[int, int]:
    """Save src to dst as JPEG q95 with the longer side capped at max_side. Returns stored (w, h)."""
    if dst.exists():
        with Image.open(dst) as im:
            return im.size
    with Image.open(src) as im:
        im = im.convert("RGB")
        scale = max_side / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=95)
        tmp.replace(dst)
        return im.size


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def manifest_row(image: Path, source: str, group: str, split: str, viewpoint: str,
                 boxes: list[tuple[int, tuple]], w: int, h: int) -> dict:
    counts = [sum(1 for c, _ in boxes if c == k) for k in range(len(CLASSES))]
    role = "pos" if counts[0] else ("neg" if boxes else "bg")
    return dict(zip(MANIFEST_COLUMNS, [str(image), source, group, split, viewpoint, role, *counts, w, h]))


def write_manifest(source: str, rows: list[dict]) -> pl.DataFrame:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows, schema=MANIFEST_COLUMNS, orient="row").sort("image")
    df.write_csv(MANIFESTS / f"{source}.csv")
    print_summary(df)
    return df


def print_summary(df: pl.DataFrame) -> None:
    print(
        df.group_by("split", "role").agg(
            pl.len().alias("images"),
            pl.col("group").n_unique().alias("groups"),
            *[pl.col(f"n_{c}").sum() for c in CLASSES],
        ).sort("split", "role")
    )


def split_groups(group_sizes: dict[str, int], fractions: dict[str, float], seed: int = SEED) -> dict[str, str]:
    """Assign whole groups to splits so image counts approach the target fractions.

    Groups are visited largest first (ties shuffled) and each goes to the split with the
    largest remaining deficit, so a few big groups cannot overshoot the small splits.
    """
    rng = random.Random(seed)
    groups = list(group_sizes)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -group_sizes[g])
    total = sum(group_sizes.values())
    target = {s: f * total for s, f in fractions.items()}
    count = dict.fromkeys(fractions, 0)
    out = {}
    for g in groups:
        s = max(fractions, key=lambda k: target[k] - count[k])
        out[g] = s
        count[s] += group_sizes[g]
    return out


def contact_sheet(items: list[tuple[Path, str]], out: Path, cols: int = 5, cell: int = 384) -> None:
    """Tile images with their YOLO boxes drawn (if a label exists) and a caption."""
    colors = ["red", "lime", "cyan", "yellow"]
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "black")
    for i, (img_path, caption) in enumerate(items):
        if img_path is None:  # blank cell
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            lbl = label_path(img_path)
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    c, cx, cy, bw, bh = line.split()
                    cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
                    x1, y1 = (cx - bw / 2) * im.width, (cy - bh / 2) * im.height
                    x2, y2 = (cx + bw / 2) * im.width, (cy + bh / 2) * im.height
                    pad = max(im.size) / cell * 4  # keep tiny boxes visible after downscale
                    draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], outline=colors[int(c)],
                                   width=max(2, round(max(im.size) / cell * 2)))
            im.thumbnail((cell, cell))
            sheet.paste(im, ((i % cols) * cell, (i // cols) * cell))
        ImageDraw.Draw(sheet).text(((i % cols) * cell + 4, (i // cols) * cell + 4), caption, fill="white")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
