"""Select, download and convert AOT frames for stage 2.

AOT has very few drone frames (7 flights), so it is used mainly for false-alarm
negatives (bird / airplane / helicopter) plus the drone flights not held out.

Held-out test flights (3 drone flights + 10% of the other flights) are written to
data/test_only/aot_holdout/ and are never downloaded into the training data.
Frames that contain an "Airborne" (unknown) or "Flock" object are skipped because
their class cannot be mapped.

Split train/val by flight. Images are resized so the long side is 1280 and saved as JPEG.

Usage:
    uv run python scripts/prepare_aot.py --select        # write selection csvs only
    uv run python scripts/prepare_aot.py --download      # download + convert selected frames
"""

from __future__ import annotations

import argparse
import io
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "aot"
OUT = ROOT / "data" / "stage2" / "aot"
HOLDOUT = ROOT / "data" / "test_only" / "aot_holdout"
BUCKET = "https://airborne-obj-detection-challenge-training.s3.amazonaws.com"
CLS = {"Drone": 0, "Bird": 1, "Airplane": 2, "Helicopter": 3}
TEST_DRONE_FLIGHTS = [  # one ~16px flight and two ~6px flights
    "4d196dd04a23411193d525b2356a1187",
    "7af99a22970c4a2180a247e1117a2848",
    "c59cde6e570e450eb50342213a503dbc",
]
MAX_SIDE = 1280
SEED = 0


def load() -> pd.DataFrame:
    parts = []
    for p in ["part1", "part2", "part3"]:
        d = pd.read_csv(RAW / f"{p}_groundtruth.csv",
                        usecols=["flight_id", "img_name", "frame", "id", "size_width", "size_height",
                                 "gt_left", "gt_top", "gt_right", "gt_bottom"])
        d["part"] = p
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["cls"] = d["id"].fillna("").str.rstrip("0123456789")
    d["w"] = d.gt_right - d.gt_left
    return d


def select(neg_per_class: int, n_bg: int, per_flight: int, val_frac: float) -> None:
    rng = random.Random(SEED)
    d = load()
    flights = sorted(d.flight_id.unique())
    drone_flights = set(d.loc[d.cls == "Drone", "flight_id"])
    others = [f for f in flights if f not in drone_flights]
    test_flights = set(TEST_DRONE_FLIGHTS) | set(rng.sample(others, round(0.1 * len(others))))

    HOLDOUT.mkdir(parents=True, exist_ok=True)
    (HOLDOUT / "flights.txt").write_text("\n".join(sorted(test_flights)) + "\n")
    d[d.flight_id.isin(test_flights)].to_csv(HOLDOUT / "groundtruth.csv", index=False)

    tr = d[~d.flight_id.isin(test_flights)]
    unknown = set(tr.loc[tr.cls.isin(["Airborne", "Flock"]), "img_name"])
    tr = tr[~tr.img_name.isin(unknown)]
    obj = tr[tr.cls.isin(CLS)]

    # Drone frames from the non-held-out drone flights, every 2nd frame (10Hz -> 5Hz).
    dr = obj[obj.cls == "Drone"].drop_duplicates("img_name").sort_values(["flight_id", "frame"])
    drone_imgs = set(dr.groupby("flight_id").nth(slice(None, None, 2)).img_name)

    # Negatives: frames whose objects are only bird/airplane/helicopter, >= 8px, spread over flights.
    neg_imgs: set[str] = set()
    for c in ["Bird", "Airplane", "Helicopter"]:
        cand = obj[(obj.cls == c) & (obj.w >= 8) & ~obj.img_name.isin(drone_imgs)].drop_duplicates("img_name")
        picked = []
        for _, g in cand.groupby("flight_id"):
            g = g.sort_values("frame")
            step = max(len(g) // per_flight, 1)
            picked += list(g.img_name.iloc[::step][:per_flight])
        rng.shuffle(picked)
        neg_imgs |= set(picked[:neg_per_class])

    # Background frames (no object at all), spread over flights.
    has_obj = set(tr.loc[tr.cls != "", "img_name"])
    bg = tr[~tr.img_name.isin(has_obj)].drop_duplicates("img_name")
    bg_flights = sorted(bg.flight_id.unique())
    bg_imgs = set()
    for f in rng.sample(bg_flights, min(n_bg, len(bg_flights))):
        bg_imgs.add(rng.choice(list(bg.loc[bg.flight_id == f, "img_name"])))

    chosen = drone_imgs | neg_imgs | bg_imgs
    rows = tr[tr.img_name.isin(chosen)]
    train_flights = sorted(rows.flight_id.unique())
    val_flights = set(rng.sample(train_flights, round(val_frac * len(train_flights))))
    # Keep at least one drone flight in val.
    tr_drone = sorted(set(dr.flight_id) & set(train_flights))
    if not val_flights & set(tr_drone):
        val_flights.add(rng.choice(tr_drone))
    rows = rows.assign(split=np.where(rows.flight_id.isin(val_flights), "val", "train"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "selection.csv", index=False)
    fr = rows.drop_duplicates("img_name")
    print(f"test flights {len(test_flights)} | selected frames {len(fr)} "
          f"(drone {len(drone_imgs)}, neg {len(neg_imgs)}, bg {len(bg_imgs)})")
    print(fr.split.value_counts().to_string())
    print(rows[rows.cls.isin(CLS)].groupby(["split", "cls"]).size().to_string())


def fetch(part: str, flight: str, img: str, split: str, labels: list[str], w0: int, h0: int) -> str | None:
    dst = OUT / split / "images" / (Path(img).stem + ".jpg")
    if dst.exists():
        return None
    url = f"{BUCKET}/{part}/Images/{flight}/{img}"
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return f"{img}: {e}"
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return f"{img}: decode failed"
    h, w = im.shape[:2]
    if (w, h) != (w0, h0):
        return f"{img}: size {w}x{h} != csv {w0}x{h0}"
    s = MAX_SIDE / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    lab = OUT / split / "labels" / (Path(img).stem + ".txt")
    lab.write_text("".join(l + "\n" for l in labels))
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None


def download(workers: int) -> None:
    rows = pd.read_csv(OUT / "selection.csv")
    for s in ["train", "val"]:
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)
    jobs = []
    for img, g in rows.groupby("img_name"):
        r0 = g.iloc[0]
        W, H = int(r0.size_width), int(r0.size_height)
        labels = []
        for r in g[g.cls.isin(CLS)].itertuples():
            x0, y0 = max(r.gt_left, 0), max(r.gt_top, 0)
            x1, y1 = min(r.gt_right, W), min(r.gt_bottom, H)
            if x1 <= x0 or y1 <= y0:
                continue
            labels.append(f"{CLS[r.cls]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                          f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        jobs.append((r0.part, r0.flight_id, img, r0.split, labels, W, H))
    errors = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            if (e := f.result()):
                errors.append(e)
            if i % 500 == 0:
                print(f"{i}/{len(jobs)} done, errors {len(errors)}", flush=True)
    (OUT / "download_errors.txt").write_text("\n".join(errors) + "\n")
    print(f"finished {len(jobs)} frames, errors {len(errors)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--neg-per-class", type=int, default=2000)
    ap.add_argument("--n-bg", type=int, default=500)
    ap.add_argument("--per-flight", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.select:
        select(a.neg_per_class, a.n_bg, a.per_flight, a.val_frac)
    if a.download:
        download(a.workers)


if __name__ == "__main__":
    main()
