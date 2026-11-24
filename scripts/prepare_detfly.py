"""Convert Det-Fly (air-to-air, 3840x2160, VOC xml, class "UAV") for stage 2.

Det-Fly has no sequence ids; file names are running frame numbers per folder (010, 020).
To approximate a sequence-level split (rule 2), frames are cut into blocks of --block
consecutive numbers; --val-frac of the blocks go to val, and --buffer frames on each side
of a val block are dropped from train so neighbouring frames do not leak.

Images are resized so the long side is 1280 and saved as JPEG.

Usage:
    uv run python scripts/prepare_detfly.py --img-root data/raw/detfly/img --ann-root data/raw/detfly/ann
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stage2" / "detfly"
MAX_SIDE = 1280
SEED = 0


def convert(job: tuple[str, str, str]) -> tuple[str, int, str]:
    xml, img, split = job
    r = ET.parse(xml).getroot()
    W, H = int(r.find("size/width").text), int(r.find("size/height").text)
    lines = []
    for o in r.findall("object"):
        name = o.find("name").text.strip().lower()
        if name != "uav":
            return split, 0, f"{xml}: unexpected class {name}"
        b = o.find("bndbox")
        x0, y0, x1, y1 = (float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"])
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 > x0 and y1 > y0:
            lines.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    im = cv2.imread(img, cv2.IMREAD_COLOR)
    if im is None:
        return split, 0, f"{img}: unreadable"
    h, w = im.shape[:2]
    if (w, h) != (W, H):
        return split, 0, f"{img}: size {w}x{h} != xml {W}x{H}"
    s = MAX_SIDE / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    stem = "detfly_" + Path(img).stem
    cv2.imwrite(str(OUT / split / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (OUT / split / "labels" / f"{stem}.txt").write_text("".join(l + "\n" for l in lines))
    return split, len(lines), ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", type=Path, required=True, help="folder containing 010/ and 020/ images")
    ap.add_argument("--ann-root", type=Path, required=True, help="folder containing 010/ and 020/ xml")
    ap.add_argument("--block", type=int, default=300)
    ap.add_argument("--buffer", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rng = random.Random(SEED)
    for s in ["train", "val"]:
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

    jobs, missing, buffered = [], 0, 0
    for folder in ["010", "020"]:
        xmls = sorted((a.ann_root / folder).glob("*.xml"))
        nums = {int(x.stem): x for x in xmls}
        blocks = sorted({n // a.block for n in nums})
        val_blocks = set(rng.sample(blocks, max(1, round(a.val_frac * len(blocks)))))
        for n, x in nums.items():
            img = a.img_root / folder / (x.stem + ".jpg")
            if not img.exists():
                missing += 1
                continue
            b = n // a.block
            if b in val_blocks:
                split = "val"
            else:
                near = ((n - a.buffer) // a.block in val_blocks) or ((n + a.buffer) // a.block in val_blocks)
                if near:
                    buffered += 1
                    continue
                split = "train"
            jobs.append((str(x), str(img), split))
    print(f"jobs {len(jobs)}, xml without image {missing}, dropped as buffer {buffered}")
    cnt, errors = Counter(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for split, nb, err in ex.map(convert, jobs, chunksize=32):
            if err:
                errors.append(err)
            else:
                cnt[split] += 1
                cnt[split + "_boxes"] += nb
    (OUT / "errors.txt").write_text("\n".join(errors) + "\n")
    print(dict(cnt), "errors", len(errors))


if __name__ == "__main__":
    main()
