"""md5 + 64-bit pHash for every manifest image, every Anti-UAV300 RGB frame, and pathikg. Spec §4.

Each part is written to data/manifests/hashes/<part>.parquet and skipped on rerun, then all
parts are concatenated into data/manifests/hashes.parquet with columns:
  key     image path (manifest), "antiuav/<split>/<seq>/<frame>", or "pathikg/<file>/<row>"
  source  manifest source, "antiuav_raw", or "pathikg"
  split   manifest split, Anti-UAV raw split, or "pathikg"
  md5     md5 of the file bytes (null for decoded video frames)
  phash   imagehash.phash as uint64

seraphim images are 640x640 letterboxed, so their uniform border is cropped before pHash.

Usage:
  uv run scripts/hash_images.py
"""

import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import polars as pl
from PIL import Image

from common import MANIFESTS, RAW

PARTS = MANIFESTS / "hashes"
OUT = MANIFESTS / "hashes.parquet"
SOURCES = ["seraphim", "dut", "detfly", "aot", "aod4", "antiuav"]
BORDER_TOL = 8  # max per-pixel deviation (0-255) for a row/column to count as letterbox padding


def phash_int(im: Image.Image) -> int:
    return int(str(imagehash.phash(im)), 16)


def crop_border(im: Image.Image) -> Image.Image:
    """Drop uniform rows/columns (letterbox padding) matching the top-left corner colour."""
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    pad = np.abs(a - a[0, 0]).max(axis=2) <= BORDER_TOL
    rows, cols = np.nonzero(~pad.all(axis=1))[0], np.nonzero(~pad.all(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return im
    return im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def hash_file(args: tuple[str, bool]) -> tuple[str, int]:
    path, crop = args
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im = crop_border(im) if crop else im
        return hashlib.md5(data).hexdigest(), phash_int(im)


def hash_video(seq_dir: Path) -> list[tuple[str, int]]:
    """pHash of every frame of visible.mp4 (frames that the json covers)."""
    n = len(json.loads((seq_dir / "visible.json").read_text())["exist"])
    cap = cv2.VideoCapture(str(seq_dir / "visible.mp4"))
    out, idx = [], 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break
        out.append((f"antiuav/{seq_dir.parent.name}/{seq_dir.name}/{idx:06d}", phash_int(Image.fromarray(frame[:, :, ::-1]))))
        idx += 1
    cap.release()
    return out


def hash_parquet(path: Path) -> list[tuple[str, str, int]]:
    """pathikg parquet shard: the image column holds {'bytes', 'path'} structs."""
    df = pl.read_parquet(path, columns=["image"])
    out = []
    for i, img in enumerate(df["image"].to_list()):
        data = img["bytes"]
        with Image.open(io.BytesIO(data)) as im:
            out.append((f"pathikg/{path.stem}/{i}", hashlib.md5(data).hexdigest(), phash_int(im)))
    return out


def frame(keys, source, split, md5, phash) -> pl.DataFrame:
    return pl.DataFrame(
        {"key": keys, "source": source, "split": split, "md5": md5, "phash": phash},
        schema={"key": pl.String, "source": pl.String, "split": pl.String, "md5": pl.String, "phash": pl.UInt64},
    )


def run_part(name: str, build) -> None:
    out = PARTS / f"{name}.parquet"
    if out.exists():
        print(f"{name}: cached")
        return
    df = build()
    df.write_parquet(out)
    print(f"{name}: {len(df)} hashes")


def manifest_part(source: str, ex: ProcessPoolExecutor):
    def build():
        m = pl.read_csv(MANIFESTS / f"{source}.csv", columns=["image", "split"])
        res = list(ex.map(hash_file, [(p, source == "seraphim") for p in m["image"]], chunksize=64))
        return frame(m["image"], source, m["split"], [r[0] for r in res], [r[1] for r in res])
    return build


def antiuav_part(split: str, ex: ProcessPoolExecutor):
    def build():
        seqs = sorted(p for p in (RAW / "antiuav300" / split).iterdir() if (p / "visible.mp4").exists())
        res = [r for rs in ex.map(hash_video, seqs) for r in rs]
        return frame([k for k, _ in res], "antiuav_raw", split, [None] * len(res), [h for _, h in res])
    return build


def pathikg_part(ex: ProcessPoolExecutor):
    def build():
        shards = sorted((RAW / "pathikg" / "data").glob("*.parquet"))
        if not shards:
            raise SystemExit("pathikg not found: download it to data/raw/pathikg first (spec §4)")
        res = [r for rs in ex.map(hash_parquet, shards) for r in rs]
        return frame([r[0] for r in res], "pathikg", "pathikg", [r[1] for r in res], [r[2] for r in res])
    return build


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        for s in SOURCES:
            run_part(s, manifest_part(s, ex))
        for split in ["train", "val", "test"]:
            run_part(f"antiuav_raw_{split}", antiuav_part(split, ex))
        if (PARTS / "pathikg.parquet").exists() or (RAW / "pathikg").exists():
            run_part("pathikg", pathikg_part(ex))
    df = pl.concat([pl.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))])
    df.write_parquet(OUT)
    print(df.group_by("source", "split").len().sort("source", "split"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
