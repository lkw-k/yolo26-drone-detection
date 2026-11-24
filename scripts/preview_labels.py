"""Draw labels on random images of a converted source for visual review -> data/reports/preview_<source>.jpg

Usage: uv run scripts/preview_labels.py dut [--n 20]
"""

import argparse
import random
from pathlib import Path

import polars as pl

from common import CLASSES, MANIFESTS, REPORTS, SEED, contact_sheet

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()

df = pl.read_csv(MANIFESTS / f"{args.source}.csv")
rows = df.to_dicts()
random.Random(SEED).shuffle(rows)
items = []
for r in rows[: args.n]:
    counts = " ".join(f"{c[:2]}{r[f'n_{c}']}" for c in CLASSES if r[f"n_{c}"])
    items.append((Path(r["image"]), f"{r['split']} {r['role']} {counts}"))
out = REPORTS / f"preview_{args.source}.jpg"
contact_sheet(items, out)
print(out)
