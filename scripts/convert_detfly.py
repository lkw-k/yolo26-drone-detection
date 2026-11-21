"""Det-Fly (4K, VOC xml) -> data/yolo/detfly, plus scene grouping for the split. Spec §3, D2.

Steps:
  convert  resize to 1920x1080, write labels, cache 64x36 grayscale thumbnails
  group    link frames whose thumbnails are close (RMS diff <= threshold), take connected
           components as continuity segments, merge segments of the same place/session by
           average-linkage clustering (--location-cut), merge locations whose closest frames are
           within --merge-rms, split locations train/val/test = 70/15/15, write manifest
           and a contact sheet for visual review. Run without --threshold first to see the
           nearest-neighbour distance histogram.

Usage:
  uv run scripts/convert_detfly.py convert
  uv run scripts/convert_detfly.py group
  uv run scripts/convert_detfly.py group --threshold 2.5 --location-cut 20 --merge-rms 18
"""

import argparse
import json
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage

from common import RAW, REPORTS, SEED, YOLO, contact_sheet, label_path, manifest_row, save_resized, split_groups, write_label, write_manifest, yolo_box

SRC = RAW / "detfly"
DST = YOLO / "detfly"
THUMBS = DST / "thumbs.npz"
BOXES = DST / "boxes.csv"
THUMB_SIZE = (64, 36)
LOCATION_CUT = 20.0  # RMS (0-255) between segment-mean 16x9 thumbnails; chosen from data/reports/detfly_groups.jpg
MERGE_RMS = 18.0  # merge locations whose closest frames are this close; >=20 makes one location 39%
MAX_GROUP_FRACTION = 0.35  # stop if one group holds >=35% of frames (one real session is ~32%)


def convert_one(xml_path: Path) -> tuple[str, list, int, int, np.ndarray]:
    folder = xml_path.parent.name
    src_img = SRC / "JPEGImages" / folder / (xml_path.stem + ".jpg")
    dst_img = DST / "images" / folder / src_img.name
    root = ET.parse(xml_path).getroot()
    sw, sh = int(root.findtext("size/width")), int(root.findtext("size/height"))
    w, h = save_resized(src_img, dst_img)
    sx, sy = w / sw, h / sh
    boxes = []
    for obj in root.iter("object"):
        if obj.findtext("name") != "UAV":
            raise ValueError(f"{xml_path}: unexpected class {obj.findtext('name')!r}")
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        box = yolo_box(x1 * sx, y1 * sy, x2 * sx, y2 * sy, w, h)
        if box:
            boxes.append((0, box))
    write_label(label_path(dst_img), boxes)
    with Image.open(dst_img) as im:
        thumb = np.asarray(im.convert("L").resize(THUMB_SIZE, Image.BILINEAR), dtype=np.uint8)
    return str(dst_img), boxes, w, h, thumb


def convert() -> None:
    xmls = sorted((SRC / "Annotations").glob("*/*.xml"))
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(convert_one, xmls, chunksize=16))
    images = [r[0] for r in results]
    np.savez_compressed(THUMBS, images=np.array(images), thumbs=np.stack([r[4] for r in results]))
    pl.DataFrame(
        {"image": images, "boxes": [json.dumps(r[1]) for r in results], "w": [r[2] for r in results], "h": [r[3] for r in results]}
    ).write_csv(BOXES)
    print(f"detfly: converted {len(results)} images -> {DST}")


def pairwise_rms(x: np.ndarray, chunk: int = 2048):
    """Yield (row_start, RMS distance block) for all rows of x against x."""
    sq = (x**2).sum(1)
    for i in range(0, len(x), chunk):
        d2 = sq[i : i + chunk, None] + sq[None, :] - 2 * x[i : i + chunk] @ x.T
        yield i, np.sqrt(np.maximum(d2, 0) / x.shape[1])


def find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def group(threshold: float | None, location_cut: float, merge_rms: float = MERGE_RMS) -> None:
    z = np.load(THUMBS)
    images, x = z["images"], z["thumbs"].reshape(len(z["thumbs"]), -1).astype(np.float32)
    n = len(x)
    nn = np.empty(n, np.float32)
    pairs = []
    for i, d in pairwise_rms(x):
        rows = np.arange(i, i + len(d))
        d[rows - i, rows] = np.inf  # ignore self
        nn[rows] = d.min(1)
        if threshold is not None:
            a, b = np.nonzero(d <= threshold)
            pairs.append(np.stack([a + i, b], 1))
    print("nearest-neighbour RMS distance percentiles (0-255 scale):")
    for p in [10, 25, 50, 75, 90, 95, 99, 99.9]:
        print(f"  p{p}: {np.percentile(nn, p):.2f}")
    hist, edges = np.histogram(nn, bins=40, range=(0, float(np.percentile(nn, 99.5))))
    for c, e in zip(hist, edges):
        print(f"  {e:6.2f} {'#' * int(60 * c / hist.max())}")
    if threshold is None:
        print("pick a threshold from the gap above and rerun with --threshold")
        return

    parent = np.arange(n)
    for a, b in np.concatenate(pairs):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(parent, i) for i in range(n)])
    _, seg = np.unique(roots, return_inverse=True)  # continuity segments
    print(f"threshold {threshold}: {seg.max() + 1} continuity segments")

    # Level 2: the same place splits into many segments, so merge segments whose mean
    # 16x9 thumbnails are close (average linkage) into location/session clusters.
    coarse = np.stack([np.asarray(Image.fromarray(t).resize((16, 9), Image.BOX), np.float32) for t in z["thumbs"]]).reshape(n, -1)
    means = np.stack([coarse[seg == s].mean(0) for s in range(seg.max() + 1)])
    loc_of_seg = fcluster(linkage(means, "average"), location_cut * np.sqrt(means.shape[1]), "distance")
    _, gid = np.unique(loc_of_seg[seg], return_inverse=True)

    # Level 3: average linkage still splits one session when lighting drifts (e.g. a sky-only
    # session landed in both train and test). Merge locations whose closest frame pair is within
    # merge_rms (single linkage at frame level). See data/reports/detfly_location_pairs.jpg.
    n_loc = gid.max() + 1
    loc_parent = np.arange(n_loc)
    for i, d in pairwise_rms(x):
        a, b = np.nonzero((d <= merge_rms) & (gid[i : i + len(d), None] != gid[None, :]))
        for la, lb in set(zip(gid[a + i].tolist(), gid[b].tolist())):
            ra, rb = find(loc_parent, la), find(loc_parent, lb)
            if ra != rb:
                loc_parent[ra] = rb
    _, gid = np.unique([find(loc_parent, g) for g in gid], return_inverse=True)
    print(f"merge rms {merge_rms}: {n_loc} -> {gid.max() + 1} locations")
    sizes = np.bincount(gid)
    order = np.argsort(-sizes)
    print(f"location cut {location_cut}: {len(sizes)} locations; largest {sizes[order[:10]].tolist()}")
    if sizes.max() >= MAX_GROUP_FRACTION * n:
        raise SystemExit(f"largest location holds {sizes.max() / n:.0%} of frames (>= {MAX_GROUP_FRACTION:.0%}); lower --location-cut")

    names = {g: f"detfly_L{rank:03d}" for rank, g in enumerate(order)}
    assign = split_groups({names[g]: int(sizes[g]) for g in range(len(sizes))}, {"train": 0.70, "val": 0.15, "test": 0.15})
    meta = {r["image"]: r for r in pl.read_csv(BOXES).to_dicts()}
    rows = []
    for img, g in zip(images, gid):
        m = meta[str(img)]
        rows.append(manifest_row(Path(img), "detfly", names[g], assign[names[g]], "a2a", json.loads(m["boxes"]), m["w"], m["h"]))
    write_manifest("detfly", rows)

    # contact sheet: one row of 6 random frames per location, largest first
    rng = random.Random(SEED)
    items = []
    for g in order:
        members = [Path(img) for img, k in zip(images, gid) if k == g]
        picks = rng.sample(members, min(6, len(members)))
        items += [(p, f"{names[g]} n={sizes[g]} {assign[names[g]]}" if j == 0 else "") for j, p in enumerate(picks)]
        items += [(None, "")] * (6 - len(picks))
    out = REPORTS / "detfly_groups.jpg"
    contact_sheet(items, out, cols=6, cell=240)
    print(f"contact sheet -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["convert", "group"])
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--location-cut", type=float, default=LOCATION_CUT)
    ap.add_argument("--merge-rms", type=float, default=MERGE_RMS)
    args = ap.parse_args()
    convert() if args.step == "convert" else group(args.threshold, args.location_cut, args.merge_rms)
