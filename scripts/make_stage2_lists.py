"""Build stage 2 seraphim replay subset and train/val lists.

--replay   copy a fixed random subset of seraphim (train -> train, test -> val) into
           data/stage2/seraphim_replay/{train,val} so it can be uploaded with the rest.
--lists    write data/stage2/lists/stage2_{train,val}.txt with absolute paths for this
           machine, from data/stage2/<source>/<split>/images, minus data/stage2/dedup/exclude.txt.
           Run it on the machine that trains (paths are absolute).

Usage:
    uv run python scripts/make_stage2_lists.py --replay
    uv run python scripts/make_stage2_lists.py --lists
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "data" / "stage2"
SOURCES = ["dut", "aod4", "aot", "detfly", "seraphim_replay"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 0


def replay(n_train: int, n_val: int) -> None:
    rng = random.Random(SEED)
    src = ROOT / "data" / "stage1" / "lists"
    for split, lst, n in [("train", "seraphim_train.txt", n_train), ("val", "seraphim_val.txt", n_val)]:
        files = sorted(l.strip() for l in (src / lst).read_text(encoding="utf-8").splitlines() if l.strip())
        pick = rng.sample(files, n)
        oi, ol = S2 / "seraphim_replay" / split / "images", S2 / "seraphim_replay" / split / "labels"
        oi.mkdir(parents=True, exist_ok=True)
        ol.mkdir(parents=True, exist_ok=True)
        for f in pick:
            p = Path(f)
            lab = p.parent.parent / "labels" / (p.stem + ".txt")
            shutil.copy2(p, oi / p.name)
            shutil.copy2(lab, ol / lab.name)
        print(f"seraphim replay {split}: {len(pick)} of {len(files)}")


def lists() -> None:
    ex_file = S2 / "dedup" / "exclude.txt"
    assert ex_file.exists(), "run dedup_stage2.py first"
    exclude = {l.strip() for l in ex_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    out = S2 / "lists"
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        files, cnt = [], Counter()
        for s in SOURCES:
            d = S2 / s / split / "images"
            assert d.is_dir(), f"missing {d}"
            for f in sorted(d.iterdir()):
                rel = f.relative_to(ROOT).as_posix()
                if f.suffix.lower() in IMG_EXTS and rel not in exclude:
                    files.append(f.resolve().as_posix())
                    cnt[s] += 1
        (out / f"stage2_{split}.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
        print(split, len(files), dict(cnt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--lists", action="store_true")
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-val", type=int, default=1500)
    a = ap.parse_args()
    if a.replay:
        replay(a.n_train, a.n_val)
    if a.lists:
        lists()


if __name__ == "__main__":
    main()
