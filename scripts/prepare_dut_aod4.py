"""Convert DUT Anti-UAV and AOD-4 to YOLO format for stage 2.

DUT: VOC xml, single class "UAV" -> 0 drone. Official train -> train, val -> val.
     The official test zip has no annotations, so it is not used.
AOD-4: Roboflow YOLOv8 export, classes [airplane, bird, drone, helicopter] -> [2, 1, 0, 3].
     Roboflow's split is by frame, so it is re-split by source video (rule 2): the group
     key is the file name without its trailing frame number (yt-<id>, V_DRONE_xxx,
     Anti-UAV sequence name, ...). Single photos (pexels/unsplash) are their own group.

Files listed in an exclude file (leakage check output) are dropped.

Usage:
    uv run python scripts/prepare_dut_aod4.py --exclude data/stage2/leakage/aod4_dut/exclude.txt
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "stage2"
AOD_MAP = {0: 2, 1: 1, 2: 0, 3: 3}
SEED = 0


def norm(p: str | Path) -> str:
    return Path(p).resolve().as_posix().lower()


def dut(exclude: set[str]) -> None:
    for split in ["train", "val"]:
        src = RAW / "dut" / "extracted" / split
        oi, ol = OUT / "dut" / split / "images", OUT / "dut" / split / "labels"
        oi.mkdir(parents=True, exist_ok=True)
        ol.mkdir(parents=True, exist_ok=True)
        n = boxes = skipped = 0
        for xml in sorted((src / "xml").glob("*.xml")):
            img = src / "img" / (xml.stem + ".jpg")
            if not img.exists() or norm(img) in exclude:
                skipped += 1
                continue
            r = ET.parse(xml).getroot()
            W, H = int(r.find("size/width").text), int(r.find("size/height").text)
            lines = []
            for o in r.findall("object"):
                name = o.find("name").text.strip().lower()
                assert name == "uav", f"unexpected class {name} in {xml}"
                b = o.find("bndbox")
                x0, y0, x1, y1 = (float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"])
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
                if x1 <= x0 or y1 <= y0:
                    continue
                lines.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            stem = f"dut_{split}_{xml.stem}"
            shutil.copy2(img, oi / f"{stem}.jpg")
            (ol / f"{stem}.txt").write_text("".join(l + "\n" for l in lines))
            n += 1
            boxes += len(lines)
        print(f"DUT {split}: {n} images, {boxes} boxes, skipped {skipped}")


def group_key(name: str) -> str:
    base = re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", name, flags=re.I)
    m = re.match(r"^(2019\d{4}_\d{6}_\d+_\d+)_\d+$", base)  # Anti-UAV sequence frames
    if m:
        return m.group(1)
    m = re.match(r"^(yt-[A-Za-z0-9_-]{11})", base)  # YouTube video id
    if m:
        return m.group(1)
    m = re.match(r"^(V_[A-Z]+_\d{3})", base)  # V_DRONE_048158_199: video id is the first 3 digits
    if m:
        return m.group(1)
    return re.sub(r"[-_ ]?\d+$", "", base) or base


def aod4(exclude: set[str], val_frac: float) -> None:
    src = RAW / "aod4" / "extracted"
    items = []
    for split in ["train", "valid", "test"]:
        for img in sorted((src / split / "images").iterdir()):
            items.append((img, src / split / "labels" / (img.stem + ".txt")))
    groups = sorted({group_key(i.name) for i, _ in items})
    rng = random.Random(SEED)
    val_groups = set(rng.sample(groups, round(val_frac * len(groups))))
    cnt, skipped = Counter(), 0
    for s in ["train", "val"]:
        (OUT / "aod4" / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / "aod4" / s / "labels").mkdir(parents=True, exist_ok=True)
    for img, lab in items:
        if norm(img) in exclude:
            skipped += 1
            continue
        s = "val" if group_key(img.name) in val_groups else "train"
        lines = []
        for l in lab.read_text().splitlines() if lab.exists() else []:
            p = l.split()
            if not p:
                continue
            c = AOD_MAP[int(p[0])]
            if len(p) == 5:
                lines.append(" ".join([str(c)] + p[1:]))
            else:  # polygon line -> bounding box
                assert len(p) >= 7 and len(p) % 2 == 1, f"bad line in {lab}: {l}"
                xs, ys = [float(v) for v in p[1::2]], [float(v) for v in p[2::2]]
                x0, x1 = max(min(xs), 0.0), min(max(xs), 1.0)
                y0, y1 = max(min(ys), 0.0), min(max(ys), 1.0)
                if x1 <= x0 or y1 <= y0:
                    continue
                lines.append(f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
                cnt[(s, "polygon")] += 1
            cnt[(s, c)] += 1
        shutil.copy2(img, OUT / "aod4" / s / "images" / img.name)
        (OUT / "aod4" / s / "labels" / (img.stem + ".txt")).write_text("".join(l + "\n" for l in lines))
        cnt[(s, "img")] += 1
    print(f"AOD-4: {len(groups)} groups ({len(val_groups)} val), skipped {skipped}")
    print(sorted(cnt.items(), key=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()
    exclude = {norm(l) for l in a.exclude.read_text().splitlines() if l.strip()}
    dut(exclude)
    aod4(exclude, a.val_frac)


if __name__ == "__main__":
    main()
