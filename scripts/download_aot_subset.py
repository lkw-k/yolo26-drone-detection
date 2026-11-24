"""Select and download a ~15k-frame subset of AOT (Airborne Object Tracking).

Subset composition:
  - every frame that contains a Drone box
  - Bird / Airplane / Helicopter frames, spread round-robin across flights,
    at most one frame per 10-frame (1 s) bucket per flight
  - a small set of empty-sky frames (background images)
Frames containing `Airborne` (unknown type) or `Flock` are skipped.
Images are stored as JPEG (quality 95) to save disk space.

Usage:
  uv run python scripts/download_aot_subset.py select
  uv run python scripts/download_aot_subset.py download
"""

import argparse
import io
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from PIL import Image

AOT_ROOT = Path("data/raw/aot")
MANIFEST = AOT_ROOT / "subset_manifest.csv"
IMAGE_DIR = AOT_ROOT / "images"
BASE_URL = "https://airborne-obj-detection-challenge-training.s3.amazonaws.com"
PARTS = ["part1", "part2", "part3"]

TOTAL = 15_000
N_EMPTY = 1_000
SEED = 0


def load_groundtruth() -> pl.DataFrame:
    frames = []
    for part in PARTS:
        df = pl.read_csv(AOT_ROOT / part / "ImageSets" / "groundtruth.csv", infer_schema_length=100_000)
        frames.append(df.with_columns(pl.lit(part).alias("part")))
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.with_columns(
        pl.col("id").str.replace(r"\d+$", "").str.to_titlecase().alias("cls")  # "BIrd" -> "Bird"
    )


def round_robin(frames: pl.DataFrame, n: int, bucket: int, rng: random.Random) -> list[str]:
    """Pick up to n frames, one per `bucket` frames per flight, cycling over flights."""
    frames = frames.with_columns((pl.col("frame") // bucket).alias("b")).unique(["flight_id", "b"], keep="first")
    per_flight = {}
    for (fid,), g in frames.sort("frame").group_by(["flight_id"], maintain_order=True):
        names = g["img_name"].to_list()
        rng.shuffle(names)
        per_flight[fid] = names
    flights = sorted(per_flight)
    rng.shuffle(flights)
    picked = []
    while len(picked) < n and flights:
        nxt = []
        for fid in flights:
            if per_flight[fid] and len(picked) < n:
                picked.append(per_flight[fid].pop())
            if per_flight[fid]:
                nxt.append(fid)
        flights = nxt
    return picked


def select() -> None:
    rng = random.Random(SEED)
    df = load_groundtruth()
    per_frame = df.group_by(["part", "flight_id", "img_name", "frame"]).agg(
        pl.col("cls").drop_nulls().unique().alias("classes")
    )

    has = lambda c: pl.col("classes").list.contains(c)  # noqa: E731
    per_frame = per_frame.filter(~has("Airborne") & ~has("Flock"))

    drone = per_frame.filter(has("Drone"))
    rest = per_frame.filter(~has("Drone"))
    n_neg = TOTAL - drone.height - N_EMPTY
    groups = {
        "bird": rest.filter(has("Bird")),
        "airplane": rest.filter(~has("Bird") & has("Airplane")),
        "helicopter": rest.filter(~has("Bird") & ~has("Airplane") & has("Helicopter")),
    }
    per_group = n_neg // len(groups)

    chosen = {name: "drone" for name in drone["img_name"].to_list()}
    for cat, g in groups.items():
        for name in round_robin(g, per_group, bucket=10, rng=rng):
            chosen[name] = cat
    empty = per_frame.filter(pl.col("classes").list.len() == 0)
    for name in round_robin(empty, N_EMPTY, bucket=50, rng=rng):
        chosen[name] = "empty"

    manifest = (
        per_frame.filter(pl.col("img_name").is_in(list(chosen)))
        .with_columns(pl.col("img_name").replace_strict(chosen).alias("category"))
        .select("part", "flight_id", "img_name", "frame", "category")
        .sort("part", "flight_id", "frame")
    )
    manifest.write_csv(MANIFEST)
    print(manifest.group_by("category").agg(pl.len(), pl.col("flight_id").n_unique().alias("flights")))
    print(f"total {manifest.height} -> {MANIFEST}")


def fetch(row: dict) -> str:
    out = IMAGE_DIR / row["part"] / row["flight_id"] / (Path(row["img_name"]).stem + ".jpg")
    if out.exists():
        return "skip"
    url = f"{BASE_URL}/{row['part']}/Images/{row['flight_id']}/{row['img_name']}"
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            img = Image.open(io.BytesIO(data))
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            img.save(tmp, "JPEG", quality=95)
            tmp.replace(out)
            return "ok"
        except Exception as e:  # network hiccups: retry with backoff
            if attempt == 3:
                return f"fail {url}: {e}"
            time.sleep(2**attempt)
    return "fail"


def download(workers: int) -> None:
    rows = pl.read_csv(MANIFEST).to_dicts()
    done = fails = 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed(ex.submit(fetch, r) for r in rows):
            res = fut.result()
            done += 1
            if res.startswith("fail"):
                fails += 1
                print(res, flush=True)
            if done % 500 == 0:
                print(f"{done}/{len(rows)} done, {fails} failed", flush=True)
    print(f"finished: {done}/{len(rows)}, {fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "download"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    select() if args.step == "select" else download(args.workers)
