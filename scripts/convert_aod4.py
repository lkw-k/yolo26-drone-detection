"""AOD-4 (Roboflow YOLO) -> data/yolo/aod4 with class remap and source filtering. Spec §3.

Excluded: YouTube frames (`yt-*`, D4) and Anti-UAV-origin frames (D6).
Groups: `V_<CLASS>_<id>` video for Svanstrom-style frames, otherwise the original file name
before Roboflow's `.rf.<hash>` suffix (keeps augmented copies together). Groups listed in
data/manifests/aod4_group_merge.csv (written by dedupe_leak.py: videos of one session) are merged.
Split train/val = 85/15.
"""

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import MANIFESTS, RAW, YOLO, clip_yolo_line, image_size, label_path, link_image, manifest_row, split_groups, write_label, write_manifest

SRC = RAW / "aod4"
DST = YOLO / "aod4"
RAW_SPLITS = ["train", "valid", "test"]
REMAP = {0: 2, 1: 1, 2: 0, 3: 3}  # AOD-4 airplane,bird,drone,helicopter -> ours
ANTIUAV = re.compile(r"^\d{8}_\d{6}_\d+_\d+_\d+")
VIDEO = re.compile(r"^(V_[A-Z]+_\d+)_")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.jpg$")


def excluded(name: str) -> str | None:
    if name.startswith("yt-"):
        return "youtube"
    if ANTIUAV.match(name):
        return "antiuav"
    return None


def group_of(name: str) -> str:
    m = VIDEO.match(name)
    return "aod4_" + (m.group(1) if m else RF_SUFFIX.sub("", name))


def convert(args: tuple[Path, str]) -> dict:
    src_img, raw_split = args
    dst_img = DST / "images" / raw_split / src_img.name
    link_image(src_img, dst_img)
    w, h = image_size(dst_img)
    boxes = []
    for line in (SRC / raw_split / "labels" / (src_img.stem + ".txt")).read_text().splitlines():
        v = line.split()
        if v:
            box = clip_yolo_line(*map(float, v[1:5]), w, h)
            if box:
                boxes.append((REMAP[int(v[0])], box))
    write_label(label_path(dst_img), boxes)
    return manifest_row(dst_img, "aod4", group_of(src_img.name), "", "mixed", boxes, w, h)


def main() -> None:
    jobs, skipped = [], {"youtube": 0, "antiuav": 0}
    for s in RAW_SPLITS:
        for p in sorted((SRC / s / "images").glob("*.jpg")):
            reason = excluded(p.name)
            if reason:
                skipped[reason] += 1
            else:
                jobs.append((p, s))
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(convert, jobs, chunksize=64))
    merge_file = MANIFESTS / "aod4_group_merge.csv"
    if merge_file.exists():
        with open(merge_file, newline="") as f:
            merged = {r["group"]: r["merged"] for r in csv.DictReader(f)}
        for r in rows:
            r["group"] = merged.get(r["group"], r["group"])
        print(f"aod4: applied {len(merged)} group merges from {merge_file.name}")
    # split each stratum (V_AIRPLANE / V_DRONE / V_HELICOPTER / other) separately so val keeps every class
    strata: dict[str, dict[str, int]] = {}
    for r in rows:
        stratum = r["group"].split("_")[2] if r["group"].startswith("aod4_V_") else "other"
        sizes = strata.setdefault(stratum, {})
        sizes[r["group"]] = sizes.get(r["group"], 0) + 1
    assign = {}
    for sizes in strata.values():
        assign |= split_groups(sizes, {"train": 0.85, "val": 0.15})
    for r in rows:
        r["split"] = assign[r["group"]]
    print(f"aod4: kept {len(rows)}, excluded {skipped}")
    write_manifest("aod4", rows)


if __name__ == "__main__":
    main()
