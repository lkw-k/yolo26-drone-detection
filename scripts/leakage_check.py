"""Find training images that match test-set frames (exact + perceptual hash).

Test frames are used ONLY to compute hashes for leakage removal. They are
never written into any training/validation list.

seraphim images are resized+padded to 640x640, so before hashing we crop
uniform padding borders from both sides. Test frames are also hashed with
a horizontal flip because source datasets may contain flipped copies.

Matching uses multi-index hashing: a 64-bit pHash is split into 8 bytes;
two hashes within Hamming distance <= 7 must share at least one byte.

Outputs (in --out):
    train_hashes.npz, test_hashes.npz   cached hashes
    matches.csv                         train_path,test_ref,phash_dist,dhash_dist,md5_equal
    exclude.txt                         train image paths to drop
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def crop_padding(img: np.ndarray, tol: int = 4) -> np.ndarray:
    """Remove near-uniform border rows/cols (letterbox padding)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    corner = int(np.median([g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]]))
    row_pad = (np.abs(g.astype(np.int16) - corner).max(axis=1) <= tol)
    col_pad = (np.abs(g.astype(np.int16) - corner).max(axis=0) <= tol)
    rows = np.where(~row_pad)[0]
    cols = np.where(~col_pad)[0]
    if len(rows) < h * 0.2 or len(cols) < w * 0.2:
        return img  # mostly uniform image, don't crop
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def hash_array(img: np.ndarray) -> tuple[int, int]:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ph = imagehash.phash(pil)
    dh = imagehash.dhash(pil)
    return int(str(ph), 16), int(str(dh), 16)


def hash_train_file(path: str) -> tuple[str, int, int, str] | None:
    data = Path(path).read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    ph, dh = hash_array(crop_padding(img))
    return path, ph, dh, hashlib.md5(data).hexdigest()


def hash_test_source(args: tuple[str, int]) -> list[tuple[str, int, int, str]]:
    """Hash frames of one test sequence (video file or image folder)."""
    src, stride = args
    p = Path(src)
    out = []

    def add(ref: str, img: np.ndarray, md5: str = "") -> None:
        ph, dh = hash_array(img)
        out.append((ref, ph, dh, md5))
        ph, dh = hash_array(cv2.flip(img, 1))
        out.append((ref + "#flip", ph, dh, ""))

    if p.is_file() and p.suffix.lower() in VID_EXTS:
        cap = cv2.VideoCapture(str(p))
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    add(f"{p}@{i}", frame)
            i += 1
        cap.release()
    elif p.is_dir():
        frames = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        for f in frames[::stride]:
            data = f.read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                add(str(f), img, hashlib.md5(data).hexdigest())
    return out


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def save(path: Path, rows: list[tuple[str, int, int, str]]) -> None:
    np.savez(
        path,
        ref=np.array([r[0] for r in rows]),
        ph=np.array([r[1] for r in rows], dtype=np.uint64),
        dh=np.array([r[2] for r in rows], dtype=np.uint64),
        md5=np.array([r[3] for r in rows]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dirs", nargs="+", type=Path, required=True, help="image folders to check")
    ap.add_argument("--test-sources", nargs="+", type=Path, required=True, help="test sequence videos or frame folders")
    ap.add_argument("--stride", type=int, default=3, help="frame sampling stride for test sequences")
    ap.add_argument("--phash-thr", type=int, default=6)
    ap.add_argument("--dhash-thr", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("data/stage1/leakage"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tr_cache, te_cache = args.out / "train_hashes.npz", args.out / "test_hashes.npz"
    if not tr_cache.exists():
        files = [str(f) for d in args.train_dirs for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]
        print(f"hashing {len(files)} train images")
        with ProcessPoolExecutor(args.workers) as ex:
            rows = [r for r in ex.map(hash_train_file, files, chunksize=256) if r]
        save(tr_cache, rows)
    if not te_cache.exists():
        print(f"hashing {len(args.test_sources)} test sequences (stride {args.stride})")
        with ProcessPoolExecutor(args.workers) as ex:
            rows = [r for seq in ex.map(hash_test_source, [(str(s), args.stride) for s in args.test_sources]) for r in seq]
        save(te_cache, rows)

    tr, te = np.load(tr_cache), np.load(te_cache)
    print(f"train hashes {len(tr['ph'])}, test hashes {len(te['ph'])}")

    # Exact md5 matches.
    te_md5 = {m for m in te["md5"] if m}
    md5_hits = {i for i, m in enumerate(tr["md5"]) if m in te_md5}

    # Multi-index candidate search on pHash bytes.
    te_bytes = te["ph"].view(np.uint8).reshape(-1, 8)
    tr_bytes = tr["ph"].view(np.uint8).reshape(-1, 8)
    index = [defaultdict(list) for _ in range(8)]
    for j, row in enumerate(te_bytes):
        for k in range(8):
            index[k][row[k]].append(j)

    matches = []
    for i, row in enumerate(tr_bytes):
        cand = set()
        for k in range(8):
            cand.update(index[k].get(row[k], ()))
        if not cand and i not in md5_hits:
            continue
        cand = np.fromiter(cand, dtype=np.int64) if cand else np.zeros(0, np.int64)
        pd = popcount64(np.bitwise_xor(te["ph"][cand], tr["ph"][i])) if len(cand) else np.zeros(0)
        dd = popcount64(np.bitwise_xor(te["dh"][cand], tr["dh"][i])) if len(cand) else np.zeros(0)
        ok = (pd <= args.phash_thr) & (dd <= args.dhash_thr)
        if ok.any() or i in md5_hits:
            if ok.any():
                best = int(np.argmin(pd + dd * (~ok) * 100))
                j, p_, d_ = int(cand[best]), int(pd[best]), int(dd[best])
            else:
                j, p_, d_ = -1, -1, -1
            matches.append((str(tr["ref"][i]), str(te["ref"][j]) if j >= 0 else "md5", p_, d_, i in md5_hits))

    with open(args.out / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_path", "test_ref", "phash_dist", "dhash_dist", "md5_equal"])
        w.writerows(matches)
    (args.out / "exclude.txt").write_text("\n".join(sorted({m[0] for m in matches})) + "\n", encoding="utf-8")
    print(f"matches: {len(matches)} (md5 exact: {len(md5_hits)}) -> {args.out / 'exclude.txt'}")


if __name__ == "__main__":
    main()
