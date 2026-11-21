"""Evaluate drone detectors on the held-out test lists. Spec §7.

Step 1 (per model): predict (conf 0.001) on stage2_val and every test list, cache the drone boxes to
  data/runs/eval/<name>/preds_<list>.parquet, and record the model's own stage2-val F1-best conf.
Step 2 (always, from the caches): metrics for every evaluated model with ONE shared conf threshold =
  mean of the stage2-yolo26n and stage2-yolo26s val F1-best confs (whatever exists, noted in the report).
  The threshold is never chosen on test data.

Usage:
  uv run scripts/eval.py --weights data/runs/stage2-yolo26n/weights/best.pt
  uv run scripts/eval.py --baseline           # data/weights/baseline_yolov11s.pt, imgsz 640
  uv run scripts/eval.py --summary            # recompute metrics from the caches only
  add --limit 50 to predict on only 50 images per list (pipeline check)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from common import DATA, MANIFESTS, REPORTS, label_path

EVAL = DATA / "runs" / "eval"
SPLITS = DATA / "splits"
BASELINE = DATA / "weights" / "baseline_yolov11s.pt"
BASELINE_URL = "https://huggingface.co/sapoepsilon/yolov11s-drone-detector/resolve/main/best.pt"
TEST_LISTS = ["test_antiuav", "test_detfly", "test_aot_drone", "test_dut", "test_fa"]
IOUS = np.linspace(0.5, 0.95, 10)
SIZE_BUCKETS = {"<8": (0, 8), "8-16": (8, 16), "16-32": (16, 32), ">=32": (32, 1e9)}
SHARED_FROM = ["stage2-yolo26n", "stage2-yolo26s"]


# ---------- prediction ----------

def drone_class(names: dict) -> int:
    hits = [k for k, v in names.items() if str(v).lower() in {"drone", "uav", "drones"}]
    if len(hits) != 1:
        raise SystemExit(f"cannot find a single drone class in {names}")
    return hits[0]


def predict_list(model, images: list[str], imgsz: int, cls: int) -> pl.DataFrame:
    rows = []
    for r in model.predict(images, imgsz=imgsz, conf=0.001, stream=True, verbose=False, batch=8):
        b = r.boxes
        keep = (b.cls.cpu().numpy().astype(int) == cls)
        for (x1, y1, x2, y2), c in zip(b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep]):
            rows.append((r.path, float(c), float(x1), float(y1), float(x2), float(y2)))
    return pl.DataFrame(rows, schema=["image", "conf", "x1", "y1", "x2", "y2"], orient="row")


def cpu_latency(weights: Path, imgsz: int, images: list[str]) -> float:
    from ultralytics import YOLO
    m = YOLO(str(weights))
    for p in images[:10]:
        m.predict(p, imgsz=imgsz, device="cpu", verbose=False)
    t = time.perf_counter()
    for p in images[10:110]:
        m.predict(p, imgsz=imgsz, device="cpu", verbose=False)
    return (time.perf_counter() - t) / len(images[10:110]) * 1000


def run_predictions(weights: Path, name: str, imgsz: int, limit: int | None, with_val: bool) -> None:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    cls = drone_class(model.names)
    out = EVAL / name
    out.mkdir(parents=True, exist_ok=True)
    lists = (["stage2_val"] if with_val else []) + TEST_LISTS
    for lst in lists:
        images = (SPLITS / f"{lst}.txt").read_text().splitlines()[:limit]
        print(f"{name}: predicting {lst} ({len(images)} images, imgsz {imgsz})")
        predict_list(model, images, imgsz, cls).write_parquet(out / f"preds_{lst}.parquet")
        (out / f"images_{lst}.txt").write_text("\n".join(images) + "\n")
    lat_imgs = (SPLITS / "test_antiuav.txt").read_text().splitlines()[:110]
    meta = {
        "weights": str(weights), "imgsz": imgsz, "limit": limit,
        "params": sum(p.numel() for p in model.model.parameters()),
        "weights_mb": weights.stat().st_size / 1e6,
        "cpu_ms": cpu_latency(weights, imgsz, lat_imgs),
    }
    if with_val:
        meta["val_f1_conf"] = f1_best_conf(load_eval(out, "stage2_val"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: {meta}")


# ---------- metrics ----------

def load_gt(images: list[str], manifest: pl.DataFrame) -> dict[str, np.ndarray]:
    """Drone GT boxes (xyxy pixels) per image from the YOLO label files."""
    wh = dict(zip(manifest["image"], zip(manifest["w"], manifest["h"])))
    gt = {}
    for p in images:
        w, h = wh[p]
        boxes = []
        for line in label_path(Path(p)).read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            if c == "0":
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        gt[p] = np.array(boxes, dtype=np.float64).reshape(-1, 4)
    return gt


_MANIFEST = None


def manifest() -> pl.DataFrame:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = pl.read_csv(MANIFESTS / "final.csv", schema_overrides={"group": pl.String})
    return _MANIFEST


def load_eval(out: Path, lst: str) -> tuple[dict, dict]:
    images = (out / f"images_{lst}.txt").read_text().splitlines()
    preds = pl.read_parquet(out / f"preds_{lst}.parquet")
    by_img = {p: np.empty((0, 5)) for p in images}
    for (img,), g in preds.group_by("image"):
        by_img[img] = g.select("x1", "y1", "x2", "y2", "conf").to_numpy()
    return by_img, load_gt(images, manifest())


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def size_of(boxes: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))


def match(preds: dict, gts: dict, bucket: tuple[float, float] | None = None):
    """Greedy matching by descending conf. Returns (conf[N], tp[N, len(IOUS)], n_gt).

    With a size bucket (COCO area-range rule): GT outside the bucket are 'ignore'; a detection matched
    to an ignored GT is dropped; an unmatched detection outside the bucket is dropped."""
    confs, tps, n_gt = [], [], 0
    for img, p in preds.items():
        g = gts[img]
        ign = np.zeros(len(g), bool) if bucket is None else ~((size_of(g) >= bucket[0]) & (size_of(g) < bucket[1]))
        n_gt += int((~ign).sum())
        if len(p) == 0:
            continue
        p = p[np.argsort(-p[:, 4])]
        ious = iou(p, g) if len(g) else np.zeros((len(p), 0))
        tp = np.zeros((len(p), len(IOUS)), bool)
        drop = np.zeros((len(p), len(IOUS)), bool)
        for t, thr in enumerate(IOUS):
            used = np.zeros(len(g), bool)
            for i in range(len(p)):
                if len(g) == 0:
                    break
                cand = np.where(~used & (ious[i] >= thr))[0]
                if len(cand) == 0:
                    continue
                # prefer non-ignored GT, then highest IoU
                j = cand[np.lexsort((-ious[i, cand], ign[cand]))[0]]
                used[j] = True
                if ign[j]:
                    drop[i, t] = True
                else:
                    tp[i, t] = True
        if bucket is not None:
            ps = size_of(p[:, :4])
            outside = ~((ps >= bucket[0]) & (ps < bucket[1]))
            drop |= (~tp & ~drop) & outside[:, None]
        # a detection dropped at IoU 0.5 is excluded everywhere (keeps arrays aligned; tiny effect at higher IoU)
        keep = ~drop[:, 0]
        confs.append(p[keep, 4])
        tps.append(tp[keep])
    if not confs:
        return np.empty(0), np.empty((0, len(IOUS)), bool), n_gt
    return np.concatenate(confs), np.concatenate(tps), n_gt


def ap_curve(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> np.ndarray:
    """COCO 101-point interpolated AP for each IoU threshold."""
    if n_gt == 0:
        return np.full(len(IOUS), np.nan)
    o = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[o], 0)
    fpc = np.cumsum(~tp[o], 0)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    out = []
    for t in range(len(IOUS)):
        p = np.concatenate([[0], prec[:, t], [0]])
        r = np.concatenate([[0], rec[:, t], [1]])
        p = np.flip(np.maximum.accumulate(np.flip(p)))
        x = np.linspace(0, 1, 101)
        idx = np.searchsorted(r, x, side="left")
        out.append(np.mean(np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0)))
    return np.array(out)


def f1_best_conf(ev: tuple[dict, dict]) -> float:
    conf, tp, n_gt = match(*ev)
    if len(conf) == 0 or n_gt == 0:
        return 0.25
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o, 0])
    prec, rec = tpc / np.arange(1, len(o) + 1), tpc / n_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(conf[o][np.argmax(f1)])


def at_conf(conf: np.ndarray, tp: np.ndarray, n_gt: int, thr: float) -> dict:
    k = conf >= thr
    tpn, fpn = int(tp[k, 0].sum()), int((~tp[k, 0]).sum())
    r = tpn / n_gt if n_gt else float("nan")
    p = tpn / (tpn + fpn) if tpn + fpn else float("nan")
    return {"recall": r, "precision": p, "tp": tpn, "fp": fpn, "n_gt": n_gt}


def fa_rates(preds: dict, thr: float) -> dict:
    m = manifest().filter(pl.col("image").is_in(list(preds)))
    kinds = {
        "bird_only": (pl.col("n_bird") > 0) & (pl.col("n_airplane") == 0) & (pl.col("n_helicopter") == 0),
        "airplane_only": (pl.col("n_airplane") > 0) & (pl.col("n_bird") == 0) & (pl.col("n_helicopter") == 0),
        "helicopter_only": (pl.col("n_helicopter") > 0) & (pl.col("n_bird") == 0) & (pl.col("n_airplane") == 0),
        "empty_sky": pl.col("role") == "bg",
    }
    out = {}
    for k, expr in kinds.items():
        imgs = m.filter(expr & (pl.col("n_drone") == 0))["image"].to_list()
        n = sum(int((preds[i][:, 4] >= thr).sum()) for i in imgs)
        out[k] = {"frames": len(imgs), "drone_preds_per_frame": n / len(imgs) if imgs else float("nan")}
    return out


def evaluate(name: str, thr: float) -> dict:
    out = EVAL / name
    res = {"name": name, "conf_threshold": thr, **json.loads((out / "meta.json").read_text()), "lists": {}}
    for lst in TEST_LISTS:
        preds, gts = load_eval(out, lst)
        if lst == "test_fa":
            res["lists"][lst] = {"false_alarms": fa_rates(preds, thr)}
            continue
        conf, tp, n_gt = match(preds, gts)
        ap = ap_curve(conf, tp, n_gt)
        entry = {"AP50": float(ap[0]), "mAP50-95": float(np.nanmean(ap)), **at_conf(conf, tp, n_gt, thr), "size": {}}
        for b, rng in SIZE_BUCKETS.items():
            c, t, n = match(preds, gts, rng)
            entry["size"][b] = {"AP50": float(ap_curve(c, t, n)[0]), **at_conf(c, t, n, thr)}
        res["lists"][lst] = entry
    (EVAL / f"{name}.json").write_text(json.dumps(res, indent=2))
    return res


def summary() -> None:
    names = sorted(p.name for p in EVAL.iterdir() if (p / "meta.json").exists()) if EVAL.exists() else []
    if not names:
        raise SystemExit("no evaluated models yet")
    metas = {n: json.loads((EVAL / n / "meta.json").read_text()) for n in names}
    src = [n for n in SHARED_FROM if "val_f1_conf" in metas.get(n, {})]
    note = ""
    if not src:  # e.g. smoke runs: fall back to every model that has a val conf
        src = [n for n in names if "val_f1_conf" in metas[n]]
        note = f" (fallback: {', '.join(SHARED_FROM)} not evaluated yet)"
    if not src:
        raise SystemExit("no model has a stage2-val conf yet; evaluate a trained model first")
    thr = float(np.mean([metas[n]["val_f1_conf"] for n in src]))
    results = [evaluate(n, thr) for n in names]

    f = lambda v: "n/a" if v != v else f"{v:.3f}"
    lines = [
        "# Evaluation summary", "",
        f"Shared conf threshold {thr:.3f} = mean stage2-val F1-best conf of {', '.join(src)}{note}. Never tuned on test data.", "",
        "## Drone AP50 / recall per test list", "",
        "| model | " + " | ".join(f"{l} AP50 | {l} R" for l in TEST_LISTS if l != "test_fa") + " |",
        "|---|" + "---|---|" * (len(TEST_LISTS) - 1),
    ]
    for r in results:
        lines.append(f"| {r['name']} | " + " | ".join(f"{f(r['lists'][l]['AP50'])} | {f(r['lists'][l]['recall'])}" for l in TEST_LISTS if l != "test_fa") + " |")
    lines += ["", "## AP50 by GT size (sqrt(w*h) px)", ""]
    for l in [x for x in TEST_LISTS if x != "test_fa"]:
        lines += [f"**{l}**", "", "| model | " + " | ".join(SIZE_BUCKETS) + " |", "|---|" + "---|" * len(SIZE_BUCKETS)]
        for r in results:
            s = r["lists"][l]["size"]
            lines.append(f"| {r['name']} | " + " | ".join(f"{f(s[b]['AP50'])} (n={s[b]['n_gt']})" for b in SIZE_BUCKETS) + " |")
        lines.append("")
    lines += ["## False alarms on test_fa (drone predictions per frame)", "", "| model | bird_only | airplane_only | helicopter_only | empty_sky |", "|---|---|---|---|---|"]
    for r in results:
        fa = r["lists"]["test_fa"]["false_alarms"]
        lines.append(f"| {r['name']} | " + " | ".join(f"{f(fa[k]['drone_preds_per_frame'])} (n={fa[k]['frames']})" for k in fa) + " |")
    lines += ["", "## Cost", "", "| model | imgsz | params (M) | weights (MB) | CPU ms/img |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['imgsz']} | {r['params'] / 1e6:.2f} | {r['weights_mb']:.1f} | {r['cpu_ms']:.0f} |")
    if any(r.get("limit") for r in results):
        lines += ["", "WARNING: some models were evaluated with --limit (partial lists); not comparable."]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", type=Path)
    g.add_argument("--baseline", action="store_true")
    g.add_argument("--summary", action="store_true")
    ap.add_argument("--name", help="defaults to the run folder name (e.g. stage2-yolo26n)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.weights:
        name = args.name or args.weights.parent.parent.name
        run_predictions(args.weights, name, args.imgsz, args.limit, with_val=True)
    elif args.baseline:
        if not BASELINE.exists():
            raise SystemExit(f"baseline weights missing. Download them yourself:\n  curl -L -o {BASELINE} {BASELINE_URL}")
        run_predictions(BASELINE, "baseline-yolov11s", 640, args.limit, with_val=False)
    summary()


if __name__ == "__main__":
    main()
