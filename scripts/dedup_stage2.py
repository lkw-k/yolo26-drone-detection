"""Deduplicate the stage 2 merge (rule 3): exact md5 + perceptual hash.

A pair is a duplicate when md5 is equal, or pHash <= --phash-thr and dHash <= --dhash-thr.
Only pairs that matter for a merge are removed:
  - pairs from different sources (cross-dataset duplicates), and
  - pairs across train/val (would leak into validation).
Near-identical consecutive frames inside one source and one split are expected for video
data and are kept. For a cross-split pair the val copy is dropped: most such pairs come from
fixed cameras (same background, tiny drone at a different position), so dropping val keeps the
training data and leaves a val set with no near-duplicate in train. For a cross-source pair
inside one split, the copy from the later source in SOURCE_ORDER is dropped.

Writes data/stage2/dedup/{hashes.npz, pairs.csv, exclude.txt} (exclude paths are repo-relative).

Usage:
    uv run python scripts/dedup_stage2.py --workers 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from leakage_check import hash_train_file, popcount64  # noqa: E402

S2 = ROOT / "data" / "stage2"
SOURCE_ORDER = ["dut", "detfly", "aot", "aod4", "seraphim_replay"]  # earlier = kept
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phash-thr", type=int, default=4)
    ap.add_argument("--dhash-thr", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = S2 / "dedup"
    out.mkdir(parents=True, exist_ok=True)

    files = [f for s in SOURCE_ORDER for sp in ["train", "val"]
             for f in sorted((S2 / s / sp / "images").iterdir()) if f.suffix.lower() in IMG_EXTS]
    cache = out / "hashes.npz"
    if cache.exists() and len(np.load(cache)["ref"]) == len(files):
        h = np.load(cache)
        ref, ph, dh, md5 = list(h["ref"]), h["ph"], h["dh"], list(h["md5"])
    else:
        print(f"hashing {len(files)} images")
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(hash_train_file, [str(f) for f in files], chunksize=256))
        bad = [str(f) for f, r in zip(files, rows) if r is None]
        assert not bad, f"unreadable images: {bad[:5]}"
        ref = [Path(r[0]).relative_to(ROOT).as_posix() for r in rows]
        ph = np.array([r[1] for r in rows], np.uint64)
        dh = np.array([r[2] for r in rows], np.uint64)
        md5 = [r[3] for r in rows]
        np.savez(cache, ref=np.array(ref), ph=ph, dh=dh, md5=np.array(md5))

    src = np.array([r.split("/")[2] for r in ref])  # data/stage2/<source>/<split>/images/...
    split = np.array([r.split("/")[3] for r in ref])
    rank = {s: i for i, s in enumerate(SOURCE_ORDER)}

    def loser(i: int, j: int) -> int:
        if split[i] != split[j]:
            return i if split[i] == "val" else j
        return i if rank[src[i]] > rank[src[j]] else j

    pairs, drop = [], set()
    by_md5 = defaultdict(list)
    for i, m in enumerate(md5):
        by_md5[m].append(i)
    for idx in by_md5.values():
        for k in idx[1:]:
            pairs.append((ref[idx[0]], ref[k], 0, 0, True))
            drop.add(loser(idx[0], k) if (src[idx[0]] != src[k] or split[idx[0]] != split[k]) else k)

    b = ph.view(np.uint8).reshape(-1, 8)
    index = [defaultdict(list) for _ in range(8)]
    for i, row in enumerate(b):
        for k in range(8):
            index[k][row[k]].append(i)
    index = [{key: np.array(v) for key, v in t.items()} for t in index]
    for i in range(len(ref)):
        cand = np.unique(np.concatenate([index[k][b[i, k]] for k in range(8)]))
        cand = cand[cand > i]
        if not len(cand):
            continue
        cand = cand[(src[cand] != src[i]) | (split[cand] != split[i])]
        if not len(cand):
            continue
        pd_ = popcount64(np.bitwise_xor(ph[cand], ph[i]))
        dd_ = popcount64(np.bitwise_xor(dh[cand], dh[i]))
        for j, p_, d_ in zip(cand[(pd_ <= a.phash_thr) & (dd_ <= a.dhash_thr)],
                             pd_[(pd_ <= a.phash_thr) & (dd_ <= a.dhash_thr)],
                             dd_[(pd_ <= a.phash_thr) & (dd_ <= a.dhash_thr)]):
            pairs.append((ref[i], ref[j], int(p_), int(d_), False))
            drop.add(loser(i, int(j)))

    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "phash_dist", "dhash_dist", "md5_equal"])
        w.writerows(pairs)
    (out / "exclude.txt").write_text("\n".join(sorted(ref[i] for i in drop)) + "\n", encoding="utf-8")
    by = defaultdict(int)
    for i in drop:
        by[(src[i], split[i])] += 1
    print(f"images {len(ref)}, duplicate pairs {len(pairs)}, dropped {len(drop)}: {dict(by)}")


if __name__ == "__main__":
    main()
