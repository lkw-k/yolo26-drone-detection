"""Evaluate a trained detector on the held-out test sets (drone class only).

Test sets (never used for training, validation or threshold selection):
  - anti_uav300 : Anti-UAV300 RGB test, 91 sequences (visible.mp4 + visible.json), ground-to-air
  - aot         : AOT held-out flights (data/test_only/aot_holdout), air-to-air
                  drone frames from the 3 held-out drone flights + negative frames
                  (bird-only / airplane-only / helicopter-only / empty) from held-out flights

The confidence threshold is picked on the stage 2 VALIDATION set (max drone F1), never on test.

Steps:
  uv run python scripts/eval_test.py pick-conf --weights W            # val -> conf.json
  uv run python scripts/eval_test.py aot-download                      # fetch AOT held-out frames
  uv run python scripts/eval_test.py predict --weights W --set anti_uav300 --shard 0/3 --device 0
  uv run python scripts/eval_test.py metrics --weights W                # -> eval/<run>/report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test_only"
AUAV = TEST / "anti_uav300" / "test"
AOT = TEST / "aot_holdout"
AOT_IMG = AOT / "images"
BUCKET = "https://airborne-obj-detection-challenge-training.s3.amazonaws.com"
MAX_SIDE = 1280
SIZE_BUCKETS = {"<8": (0, 8), "8-16": (8, 16), "16-32": (16, 32), ">=32": (32, 1e9)}
SEED = 0


def out_dir(weights: Path) -> Path:
    # runs/stage2/<run>/weights/best.pt -> eval/<run>
    d = ROOT / "eval" / weights.resolve().parents[1].name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- conf threshold (validation only)
def pick_conf(a) -> None:
    from ultralytics import YOLO

    data = ROOT / "data" / "stage2" / "stage2_local.yaml"
    assert data.exists(), "run on the training machine (stage2_local.yaml missing)"
    m = YOLO(str(a.weights))
    r = m.val(data=str(data), imgsz=a.imgsz, batch=16, device=a.device, plots=False,
              project=str(out_dir(a.weights)), name="val", exist_ok=True)
    f1 = r.box.f1_curve[0]  # class 0 = drone, over r.box.px confidence grid
    i = int(np.argmax(f1))
    res = {"conf": float(r.box.px[i]), "val_drone_f1": float(f1[i]),
           "val_drone_p": float(r.box.p_curve[0][i]), "val_drone_r": float(r.box.r_curve[0][i]),
           "source": "stage2 val list (max drone F1)"}
    (out_dir(a.weights) / "conf.json").write_text(json.dumps(res, indent=2))
    print(res)


# ---------------------------------------------------------------- AOT held-out frame selection
def aot_select(n_drone_stride: int, n_neg: int) -> list[dict]:
    import pandas as pd

    d = pd.read_csv(AOT / "groundtruth.csv")
    d["cls"] = d["cls"].fillna("")
    rng = random.Random(SEED)
    per_img = d.groupby("img_name").agg(cls=("cls", lambda s: frozenset(s)), flight=("flight_id", "first"),
                                        frame=("frame", "first"), part=("part", "first"),
                                        W=("size_width", "first"), H=("size_height", "first"))
    unknown = per_img.cls.apply(lambda s: bool(s & {"Airborne", "Flock"}))
    per_img = per_img[~unknown]
    picks = []
    dr = per_img[per_img.cls.apply(lambda s: "Drone" in s)].sort_values(["flight", "frame"])
    for _, g in dr.groupby("flight"):
        picks += [(i, "drone") for i in g.index[::n_drone_stride]]
    kinds = {"bird_only": {"Bird"}, "airplane_only": {"Airplane"}, "helicopter_only": {"Helicopter"}, "empty": {""}}
    for kind, s in kinds.items():
        cand = per_img[per_img.cls.apply(lambda c, s=s: set(c) == s)]
        # spread over flights: round-robin one frame per flight
        by_f = {f: list(g.sort_values("frame").index) for f, g in cand.groupby("flight")}
        for v in by_f.values():
            rng.shuffle(v)
        chosen, flights = [], sorted(by_f)
        while len(chosen) < n_neg and any(by_f.values()):
            for f in flights:
                if by_f[f] and len(chosen) < n_neg:
                    chosen.append(by_f[f].pop())
        picks += [(i, kind) for i in chosen]
    drone_gt = {k: g[["gt_left", "gt_top", "gt_right", "gt_bottom"]].to_numpy().tolist()
                for k, g in d[d.cls == "Drone"].groupby("img_name")}
    rows = []
    for img, kind in picks:
        r = per_img.loc[img]
        rows.append({"img": img, "kind": kind, "flight": r.flight, "part": r.part, "W": int(r.W), "H": int(r.H),
                     "gt": drone_gt.get(img, [])})
    return rows


def aot_fetch(r: dict) -> str | None:
    dst = AOT_IMG / (Path(r["img"]).stem + ".jpg")
    if dst.exists():
        return None
    url = f"{BUCKET}/{r['part']}/Images/{r['flight']}/{r['img']}"
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return f"{r['img']}: {e}"
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None or im.shape[1] != r["W"] or im.shape[0] != r["H"]:
        return f"{r['img']}: bad image"
    s = MAX_SIDE / max(im.shape[:2])
    im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None


def aot_download(a) -> None:
    rows = aot_select(a.drone_stride, a.n_neg)
    AOT_IMG.mkdir(parents=True, exist_ok=True)
    (AOT / "eval_frames.json").write_text(json.dumps(rows))
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["kind"]] += 1
    print("selected", dict(cnt))
    with ThreadPoolExecutor(a.workers) as ex:
        errs = [e for e in ex.map(aot_fetch, rows) if e]
    (AOT / "download_errors.txt").write_text("\n".join(errs) + "\n")
    print("errors", len(errs))


# ---------------------------------------------------------------- frame sources
def frames_anti_uav(shard: int, n_shard: int):
    seqs = sorted(p for p in AUAV.iterdir() if p.is_dir())[shard::n_shard]
    for s in seqs:
        j = json.loads((s / "visible.json").read_text())
        cap = cv2.VideoCapture(str(s / "visible.mp4"))
        i = 0
        while i < len(j["exist"]):
            ok, im = cap.read()
            if not ok:
                break
            e, rect = j["exist"][i], j["gt_rect"][i]
            if e and rect and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                x, y, w, h = rect
                yield im, f"{s.name}@{i}", "drone", [[x, y, x + w, y + h]], 1.0
            elif not e:
                yield im, f"{s.name}@{i}", "no_drone", [], 1.0
            # exist=1 without a valid box: unlabeled, skipped
            i += 1
        cap.release()


def frames_aot(shard: int, n_shard: int):
    rows = json.loads((AOT / "eval_frames.json").read_text())[shard::n_shard]
    for r in rows:
        f = AOT_IMG / (Path(r["img"]).stem + ".jpg")
        im = cv2.imread(str(f))
        if im is None:
            continue
        s = im.shape[1] / r["W"]  # image was resized; boxes stay in original pixels
        yield im, r["img"], r["kind"], r["gt"], s


def predict(a) -> None:
    from ultralytics import YOLO

    m = YOLO(str(a.weights))
    shard, n = map(int, a.shard.split("/"))
    src = frames_anti_uav if a.set == "anti_uav300" else frames_aot
    out, batch, t0 = [], [], time.time()

    def flush():
        res = m.predict([b[0] for b in batch], imgsz=a.imgsz, conf=0.001, max_det=100, device=a.device,
                        half=True, verbose=False)
        for (_, ref, kind, gt, s), r in zip(batch, res):
            b = r.boxes
            keep = (b.cls == 0).cpu().numpy()
            xyxy = b.xyxy.cpu().numpy()[keep] / s  # back to original pixels
            out.append({"ref": ref, "kind": kind, "gt": np.array(gt, np.float32).reshape(-1, 4),
                        "pred": xyxy.astype(np.float32), "score": b.conf.cpu().numpy()[keep].astype(np.float32)})
        batch.clear()

    for item in src(shard, n):
        batch.append(item)
        if len(batch) == 16:
            flush()
            if len(out) % 2000 == 0:
                print(f"{a.set} shard {shard}: {len(out)} frames, {time.time() - t0:.0f}s", flush=True)
    if batch:
        flush()
    p = out_dir(a.weights) / f"pred_{a.set}_{shard}of{n}.pkl"
    p.write_bytes(pickle.dumps(out))
    print(f"done {len(out)} frames -> {p}")


# ---------------------------------------------------------------- metrics
def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def size(boxes: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))


def match(frames, thr: float, lo: float = 0, hi: float = 1e9):
    """COCO-style greedy matching with GTs outside [lo, hi) ignored.
    Returns scores, tp flags (ignored dets removed), n_gt, and per-GT best matched score."""
    scores, tps, gt_hit_scores, n_gt = [], [], [], 0
    for f in frames:
        g, p, s = f["gt"], f["pred"], f["score"]
        gs = size(g)
        g_ign = (gs < lo) | (gs >= hi)
        n_gt += int((~g_ign).sum())
        order = np.argsort(-s)
        p, s = p[order], s[order]
        used = np.zeros(len(g), bool)
        hit = np.full(len(g), -1.0)
        ious = iou(p, g) if len(g) and len(p) else np.zeros((len(p), len(g)))
        ps = size(p)
        for k in range(len(p)):
            bj = -1
            for want_ign in (False, True):  # match non-ignored GTs first, then ignored ones
                cand = np.where(~used & (g_ign == want_ign) & (ious[k] >= thr))[0] if len(g) else []
                if len(cand):
                    bj = int(cand[np.argmax(ious[k, cand])])
                    break
            if bj >= 0:
                used[bj] = True
                if g_ign[bj]:
                    continue  # matched an ignored GT: drop det
                hit[bj] = s[k]
                scores.append(s[k]); tps.append(1)
            else:
                if ps[k] < lo or ps[k] >= hi:
                    continue  # unmatched det outside the size range: ignored
                scores.append(s[k]); tps.append(0)
        gt_hit_scores += list(hit[~g_ign])
    return np.array(scores), np.array(tps), n_gt, np.array(gt_hit_scores)


def ap_from(scores, tps, n_gt) -> float:
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-scores)
    tp = np.cumsum(tps[o]); fp = np.cumsum(1 - tps[o])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    if not len(rec):
        return 0.0
    env = np.flip(np.maximum.accumulate(np.flip(prec)))  # precision envelope (max precision at recall >= r)
    x = np.linspace(0, 1, 101)  # COCO 101-point interpolation
    idx = np.searchsorted(rec, x, side="left")
    q = np.where(idx < len(env), env[np.minimum(idx, len(env) - 1)], 0.0)  # 0 beyond max recall
    return float(q.mean())


def metrics_for(frames, conf: float, min_score: float = 0.01) -> dict:
    # Drop near-zero scores before matching (speed); effect on AP is negligible.
    frames = [{**f, "pred": f["pred"][f["score"] >= min_score], "score": f["score"][f["score"] >= min_score]}
              for f in frames]
    pos = [f for f in frames if len(f["gt"])]
    res = {"frames": len(frames), "drone_frames": len(pos), "drone_boxes": int(sum(len(f["gt"]) for f in pos))}
    if pos:
        s, t, n, hits = match(frames, 0.5)
        res["AP50"] = ap_from(s, t, n)
        res["mAP50-95"] = float(np.mean([ap_from(*match(frames, th)[:3]) for th in np.arange(0.5, 0.96, 0.05)]))
        tp = int(((s >= conf) & (t == 1)).sum()); fp = int(((s >= conf) & (t == 0)).sum())
        P, R = tp / max(tp + fp, 1), tp / max(n, 1)
        res.update(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9))
        res["by_size"] = {}
        for name, (lo, hi) in SIZE_BUCKETS.items():
            s2, t2, n2, h2 = match(frames, 0.5, lo, hi)
            res["by_size"][name] = {"gt": n2, "AP50": ap_from(s2, t2, n2),
                                    "recall": float((h2 >= conf).mean()) if n2 else float("nan")}
    neg = defaultdict(list)
    for f in frames:
        if not len(f["gt"]):
            neg[f["kind"]].append(int((f["score"] >= conf).sum()))
    res["false_alarms"] = {k: {"frames": len(v), "drone_preds_per_frame": float(np.mean(v)),
                               "frames_with_alarm": float(np.mean(np.array(v) > 0))} for k, v in neg.items()}
    return res


def cost(weights: Path, imgsz: int) -> dict:
    import torch
    from ultralytics import YOLO

    m = YOLO(str(weights))
    n_par = sum(p.numel() for p in m.model.parameters())
    im = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
    torch.set_num_threads(4)
    for _ in range(3):
        m.predict(im, imgsz=imgsz, device="cpu", verbose=False)
    ts = []
    for _ in range(20):
        t = time.perf_counter(); m.predict(im, imgsz=imgsz, device="cpu", verbose=False); ts.append(time.perf_counter() - t)
    return {"params": n_par, "weight_mb": weights.stat().st_size / 1e6, "cpu_ms_median": float(np.median(ts) * 1e3),
            "cpu_threads": 4, "imgsz": imgsz, "note": "end-to-end predict() on a 1920x1080 frame, PyTorch CPU"}


def metrics(a) -> None:
    d = out_dir(a.weights)
    conf = json.loads((d / "conf.json").read_text())
    rep = {"weights": str(a.weights), "imgsz": a.imgsz, "conf": conf}
    for st in ["anti_uav300", "aot"]:
        frames = [f for p in sorted(d.glob(f"pred_{st}_*.pkl")) for f in pickle.loads(p.read_bytes())]
        if frames:
            rep[st] = metrics_for(frames, conf["conf"])
    if not a.skip_cost:
        rep["cost"] = cost(a.weights, a.imgsz)
    (d / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pick-conf", "aot-download", "predict", "metrics"])
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/stage2/yolo26n_1280/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--set", choices=["anti_uav300", "aot"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--drone-stride", type=int, default=2)
    ap.add_argument("--n-neg", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-cost", action="store_true")
    a = ap.parse_args()
    {"pick-conf": pick_conf, "aot-download": aot_download, "predict": predict, "metrics": metrics}[a.step](a)


if __name__ == "__main__":
    main()
