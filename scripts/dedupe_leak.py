"""Remove test leakage, pathikg images, and cross-dataset duplicates. Spec §4.

Steps (in order, each image removed at most once, reasons logged to data/reports/removed.csv):
  1. leak     train/val image matches a test-only image (md5 equal or pHash Hamming <= LEAK_HAMMING).
              Test-only pool: every split starting with "test" in the manifests except DUT test (D3: DUT's
              official splits share videos, so it is only reported) + all Anti-UAV300 RGB frames.
              No NCC check here: a false positive only costs one training image.
  2. pathikg  seraphim image matches a pathikg image (md5 equal, or Hamming <= DUP_HAMMING and NCC >= VERIFY_NCC) (D4)
  3. dup      remaining train/val duplicates: md5 equal, or Hamming <= DUP_HAMMING and NCC >= VERIFY_NCC, between
              different sources, or between seraphim train and seraphim val. Near-identical frames inside one
              source split are temporal neighbours, not copies, so they are kept.
              Each cluster keeps one image: val before train, then source priority.
     aod4     near-duplicates across AOD-4 groups of the same stratum are different videos of one session
              (same static camera), so their groups are merged instead: the script writes
              data/manifests/aod4_group_merge.csv and stops; rerun convert_aod4.py, then this script.
pHash alone confuses plain-sky images (a helicopter and a drone on grey sky at distance 4-6), so non-exact
matches in steps 2-3 are verified with the normalized cross-correlation of 32x32 grayscale thumbnails
(true copies ~1.0, sky look-alikes 0.3-0.7).
Outputs: data/manifests/final.csv, data/reports/removed.csv, data/reports/dedupe_leak.md,
         data/reports/boundary_{leak,dup}.jpg (pairs just inside/outside the thresholds).

Usage:
  uv run scripts/dedupe_leak.py
"""

import argparse
import io
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from PIL import Image

from common import MANIFESTS, RAW, REPORTS, SEED, contact_sheet
from hash_images import crop_border

SOURCES = ["seraphim", "dut", "detfly", "aot", "aod4", "antiuav"]
PRIORITY = ["detfly", "dut", "aot", "aod4", "seraphim"]  # closest to the original first
LEAK_HAMMING = 6
DUP_HAMMING = 4
VERIFY_NCC = 0.85
DARK = 8  # pixels this dark in either thumbnail (rotation padding) are left out of the NCC
AOD4_MERGE = MANIFESTS / "aod4_group_merge.csv"
CHUNK = 128


def popdist(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.bitwise_count(np.bitwise_xor(q[:, None], ref[None, :]))


def nearest(q: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min Hamming distance and argmin of each q against ref."""
    def block(i):
        d = popdist(q[i : i + CHUNK], ref)
        return i, d.min(1), d.argmin(1)
    dist, arg = np.empty(len(q), np.uint8), np.empty(len(q), np.int64)
    with ThreadPoolExecutor() as ex:
        for i, m, a in ex.map(block, range(0, len(q), CHUNK)):
            dist[i : i + len(m)], arg[i : i + len(m)] = m, a
    return dist, arg


def close_pairs(h: np.ndarray, src: np.ndarray, grp: np.ndarray, spl: np.ndarray, max_d: int) -> np.ndarray:
    """All (i, j), i < j, with Hamming <= max_d across sources, across seraphim splits, or across aod4 groups."""
    def block(i):
        d = popdist(h[i : i + CHUNK], h)
        a, b = np.nonzero(d <= max_d)
        a = a + i
        same = src[a] == src[b]
        keep = (a < b) & (~same | ((src[a] == "seraphim") & (spl[a] != spl[b])) | ((src[a] == "aod4") & (grp[a] != grp[b])))
        return np.stack([a[keep], b[keep]], 1)
    with ThreadPoolExecutor() as ex:
        parts = list(ex.map(block, range(0, len(h), CHUNK)))
    return np.concatenate(parts) if parts else np.empty((0, 2), np.int64)


def find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def load_key(key: str) -> Image.Image:
    """Open any hashes.parquet key as an RGB image (for contact sheets)."""
    if key.startswith("antiuav/"):
        _, split, seq, idx = key.split("/")
        cap = cv2.VideoCapture(str(RAW / "antiuav300" / split / seq / "visible.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        cap.release()
        return Image.fromarray(frame[:, :, ::-1])
    if key.startswith("pathikg/"):
        _, shard, row = key.split("/")
        img = pl.read_parquet(RAW / "pathikg" / "data" / f"{shard}.parquet", columns=["image"])["image"][int(row)]
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return Image.open(key).convert("RGB")


_THUMBS: dict[str, np.ndarray] = {}


def to_thumb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()


def thumb(key: str) -> np.ndarray:
    """32x32 grayscale thumbnail (seraphim letterbox cropped, as in hash_images)."""
    if key not in _THUMBS:
        im = load_key(key)
        _THUMBS[key] = to_thumb(crop_border(im) if "seraphim" in key else im)
    return _THUMBS[key]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation over pixels that are not rotation padding in either image."""
    m = (a > DARK) & (b > DARK)
    if m.sum() < 256:  # mostly dark (night frames): padding mask would leave too little, use every pixel
        m = np.ones_like(m)
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-6))


def verified(pairs: list[tuple[str, str]]) -> np.ndarray:
    """NCC >= VERIFY_NCC for each (key, key) pair; thumbnails are built in parallel first."""
    keys = sorted({k for p in pairs for k in p} - set(_THUMBS))
    # pathikg images live in ~500 MB parquet shards: read each shard once
    by_shard: dict[str, list[str]] = {}
    for k in keys:
        if k.startswith("pathikg/"):
            by_shard.setdefault(k.split("/")[1], []).append(k)
    for shard, ks in by_shard.items():
        col = pl.read_parquet(RAW / "pathikg" / "data" / f"{shard}.parquet", columns=["image"])["image"]
        for k in ks:
            _THUMBS[k] = to_thumb(Image.open(io.BytesIO(col[int(k.split("/")[2])]["bytes"])))
    keys = [k for k in keys if k not in _THUMBS]
    with ThreadPoolExecutor(16) as ex:
        for k, t in zip(keys, ex.map(thumb, keys)):
            _THUMBS[k] = t
    return np.array([ncc(thumb(a), thumb(b)) >= VERIFY_NCC for a, b in pairs], bool)


def boundary_sheet(pairs: list[tuple[str, str, int]], out: Path, tmp: Path) -> None:
    """Contact sheet of (candidate, match, distance) pairs; non-file keys are cached as jpgs in tmp."""
    tmp.mkdir(parents=True, exist_ok=True)
    items = []
    for n, (a, b, d) in enumerate(pairs):
        for k, key in enumerate((a, b)):
            p = Path(key)
            if not p.exists():
                p = tmp / f"{n}_{k}.jpg"
                load_key(key).save(p)
            items.append((p, f"d={d} {key.split(chr(92))[-1][-40:]}" if k == 0 else key.replace(chr(92), "/").split("/data/yolo/")[-1][-44:]))
    if items:
        contact_sheet(items, out, cols=4, cell=320)


def stratum(group: str) -> str:
    return group.split("_")[2] if group.startswith("aod4_V_") else "other"


def merge_groups(pairs: np.ndarray, grp: np.ndarray) -> None:
    """Union AOD-4 groups linked by verified near-duplicates (same stratum). Stops if anything new merged."""
    links = {(grp[a], grp[b]) for a, b in pairs if stratum(grp[a]) == stratum(grp[b])}
    if not links:
        print("aod4: no cross-group near-duplicates left")
        return
    old = dict(pl.read_csv(AOD4_MERGE).iter_rows()) if AOD4_MERGE.exists() else {}
    names = sorted({g for link in links for g in link} | set(old.values()))
    idx = {g: k for k, g in enumerate(names)}
    parent = np.arange(len(names))
    for a, b in links:
        ra, rb = find(parent, idx[a]), find(parent, idx[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # root = alphabetically first name
    root = lambda g: names[find(parent, idx[g])] if g in idx else g
    new = {orig: root(cur) for orig, cur in old.items()}
    new |= {g: root(g) for g in names if root(g) != g}
    pl.DataFrame({"group": list(new), "merged": list(new.values())}).sort("group").write_csv(AOD4_MERGE)
    raise SystemExit(
        f"aod4: {len(links)} cross-group links -> {len(new)} groups merged into {len(set(new.values()))}; wrote {AOD4_MERGE}\n"
        "rerun: uv run scripts/convert_aod4.py && uv run scripts/dedupe_leak.py"
    )


def main(sheets: bool) -> None:
    m = pl.concat([pl.read_csv(MANIFESTS / f"{s}.csv", schema_overrides={"group": pl.String}) for s in SOURCES])
    hashes = pl.read_parquet(MANIFESTS / "hashes.parquet")
    m = m.join(hashes.select(pl.col("key").alias("image"), "md5", "phash"), on="image", how="left")
    if m["phash"].null_count():
        raise SystemExit(f"{m['phash'].null_count()} manifest images have no hash; rerun hash_images.py")

    is_test = m["split"].str.starts_with("test")
    cand = m.filter(~is_test).with_row_index("i")
    pool = pl.concat([
        m.filter(is_test & (pl.col("source") != "dut")).select(pl.col("image").alias("key"), "source", "md5", "phash"),
        hashes.filter(pl.col("source") == "antiuav_raw").select("key", "source", "md5", "phash"),
    ])
    removed: dict[int, tuple[str, str, int]] = {}  # cand index -> (reason, matched key, distance)
    ch = cand["phash"].to_numpy()

    # 1. test leakage
    pool_md5 = dict(zip(pool["md5"].drop_nulls(), pool.filter(pl.col("md5").is_not_null())["key"]))
    for i, md5 in zip(cand["i"], cand["md5"]):
        if md5 in pool_md5:
            removed[i] = ("leak_md5", pool_md5[md5], 0)
    pd_, pa = nearest(ch, pool["phash"].to_numpy())
    pool_keys, pool_src = pool["key"].to_numpy(), pool["source"].to_numpy()
    for i in np.nonzero(pd_ <= LEAK_HAMMING)[0]:
        removed.setdefault(int(i), (f"leak_{pool_src[pa[i]]}", pool_keys[pa[i]], int(pd_[i])))

    # 2. pathikg (seraphim only)
    pk = hashes.filter(pl.col("source") == "pathikg")
    if len(pk) == 0:
        raise SystemExit("pathikg hashes missing; download pathikg and rerun hash_images.py (spec §4)")
    pk_md5 = dict(zip(pk["md5"], pk["key"]))
    ser = np.nonzero(cand["source"].to_numpy() == "seraphim")[0]
    kd, ka = nearest(ch[ser], pk["phash"].to_numpy())
    pk_keys = pk["key"].to_numpy()
    cand_keys = cand["image"].to_numpy()
    near = [j for j in range(len(ser)) if 0 < kd[j] <= DUP_HAMMING]
    ok = dict(zip(near, verified([(cand_keys[ser[j]], pk_keys[ka[j]]) for j in near])))
    for j, i in enumerate(ser):
        md5 = cand["md5"][int(i)]
        if md5 in pk_md5:
            removed.setdefault(int(i), ("pathikg_md5", pk_md5[md5], 0))
        elif kd[j] == 0 or ok.get(j, False):
            removed.setdefault(int(i), ("pathikg", pk_keys[ka[j]], int(kd[j])))
    print(f"pathikg: {len(near)} non-exact pHash matches, {sum(ok.values())} pass NCC >= {VERIFY_NCC}")

    # 3. duplicates among the remaining train/val images
    alive = np.array([i for i in range(len(cand)) if i not in removed])
    src, grp, spl = (cand[c].to_numpy()[alive] for c in ("source", "group", "split"))
    parent = np.arange(len(alive))
    def union(a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb
    md5s = cand["md5"].to_numpy()[alive]
    first = {}
    for k, md5 in enumerate(md5s):
        union(k, first.setdefault(md5, k))
    pairs = close_pairs(ch[alive], src, grp, spl, DUP_HAMMING + 1)  # +1 only for the boundary sheet
    pair_d = np.bitwise_count(ch[alive][pairs[:, 0]] ^ ch[alive][pairs[:, 1]]) if len(pairs) else np.empty(0, np.uint8)
    akeys = cand["image"].to_numpy()[alive]
    within = pairs[pair_d <= DUP_HAMMING]
    within_d = pair_d[pair_d <= DUP_HAMMING]
    need = np.nonzero(within_d > 0)[0]
    ok = np.ones(len(within), bool)
    ok[need] = verified([(akeys[a], akeys[b]) for a, b in within[need]])
    print(f"dup: {len(within)} pHash pairs, {len(need)} non-exact, {int(ok[need].sum())} pass NCC >= {VERIFY_NCC}")
    aod4_pair = (src[within[:, 0]] == "aod4") & (src[within[:, 1]] == "aod4")
    merge_groups(within[ok & aod4_pair], grp)
    for a, b in within[ok & ~aod4_pair]:
        union(int(a), int(b))
    roots = np.array([find(parent, k) for k in range(len(alive))])
    rank = np.array([(0 if s == "val" else 1, PRIORITY.index(o)) for s, o in zip(spl, src)])
    keys = cand["image"].to_numpy()
    for r in np.unique(roots[np.bincount(roots, minlength=len(alive))[roots] > 1]):
        members = np.nonzero(roots == r)[0]
        keep = members[np.lexsort((rank[members, 1], rank[members, 0]))[0]]
        for k in members:
            if k != keep:
                d = int(np.bitwise_count(ch[alive][k] ^ ch[alive][keep]))
                removed[int(alive[k])] = ("dup_md5" if md5s[k] == md5s[keep] else "dup", keys[alive[keep]], d)

    # outputs
    rm = pl.DataFrame(
        [(int(i), *v) for i, v in sorted(removed.items())],
        schema={"i": pl.UInt32, "reason": pl.String, "matched": pl.String, "distance": pl.Int64}, orient="row",
    )
    rm_full = cand.join(rm, on="i").select("image", "source", "group", "split", "role", "reason", "matched", "distance")
    rm_full.write_csv(REPORTS / "removed.csv")
    final = m.join(rm_full.select("image"), on="image", how="anti").drop("md5", "phash")
    final.write_csv(MANIFESTS / "final.csv")

    by_reason = rm_full.group_by("reason", "source").len().sort("reason", "source")
    before = m.group_by("source", "split").len().rename({"len": "before"})
    after = final.group_by("source", "split").len().rename({"len": "after"})
    table = before.join(after, on=["source", "split"], how="left").fill_null(0).sort("source", "split")
    dut_near = int(((popdist(cand.filter(pl.col("source") == "dut")["phash"].to_numpy(),
                             m.filter((pl.col("source") == "dut") & is_test)["phash"].to_numpy())) <= DUP_HAMMING).any(1).sum())
    lines = [
        "# dedupe_leak report", "",
        f"Thresholds: leak Hamming <= {LEAK_HAMMING}, duplicate/pathikg Hamming <= {DUP_HAMMING}. md5 matches always count.", "",
        "## Removed by reason and source", "", _md(by_reason), "",
        "## Images per source and split", "", _md(table), "",
        f"DUT train/val images within Hamming {DUP_HAMMING} of a DUT test image: {dut_near} (D3: kept; DUT's official splits "
        "share videos, so test_dut is a contaminated reference, not a held-out test)", "",
        f"Non-exact pHash matches in pathikg/dup steps must also have NCC >= {VERIFY_NCC} (32x32 grayscale, rotation padding masked).", "",
        f"AOD-4 groups merged as one session: {pl.read_csv(AOD4_MERGE).height if AOD4_MERGE.exists() else 0} (see {AOD4_MERGE.name}).", "",
        f"Total removed: {len(rm_full)} of {len(cand)} train/val images. final.csv: {len(final)} rows.",
    ]
    (REPORTS / "dedupe_leak.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if sheets:
        rng = random.Random(SEED)
        near = [int(i) for i in np.nonzero((pd_ >= LEAK_HAMMING - 1) & (pd_ <= LEAK_HAMMING + 1))[0]]
        leak_pairs = [(keys[i], pool_keys[pa[i]], int(pd_[i])) for i in sorted(rng.sample(near, min(10, len(near))), key=lambda i: pd_[i])]
        boundary_sheet(leak_pairs, REPORTS / "boundary_leak.jpg", REPORTS / "boundary_tmp")
        sel = np.nonzero((pair_d >= DUP_HAMMING - 1) & (pair_d <= DUP_HAMMING + 1))[0] if len(pairs) else []
        sel = sorted(rng.sample(list(sel), min(10, len(sel))), key=lambda s: pair_d[s])
        dup_pairs = [(keys[alive[pairs[s, 0]]], keys[alive[pairs[s, 1]]], int(pair_d[s])) for s in sel]
        boundary_sheet(dup_pairs, REPORTS / "boundary_dup.jpg", REPORTS / "boundary_tmp")
        print("boundary sheets -> data/reports/boundary_{leak,dup}.jpg")


def _md(df: pl.DataFrame) -> str:
    cols = df.columns
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sheets", action="store_true")
    main(not ap.parse_args().no_sheets)
