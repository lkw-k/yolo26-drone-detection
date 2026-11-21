"""One-command YOLO26 training with early stopping, an overall progress bar, and crash-safe auto-resume. Spec §6.

Usage:
  uv run scripts/train.py --model n --stage 1
  uv run scripts/train.py --model s --stage 2
  uv run scripts/train.py --model n --stage 1 --smoke   # 200 images per split, 3 epochs, imgsz 640
  uv run scripts/train.py --model n --stage 1 --fresh   # keep the old run as <name>-old-<timestamp>, start over

Rerunning the same command after a crash or power loss resumes from weights/last.pt, falling back to
last_safe.pt and then the newest epoch{N}.pt if a file is unreadable. A finished run (done.json) is not rerun.
"""

import argparse
import ctypes
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, ROOT, SEED  # noqa: E402

RUNS = DATA / "runs"
WEIGHTS = DATA / "weights"
SPLITS = DATA / "splits"
SMOKE = dict(n_images=200, epochs=3, imgsz=640, save_period=1)  # 3 epochs so kill/resume can be tested


def load_resumable(path: Path) -> dict | None:
    """Checkpoint dict if path is a readable, unfinished training checkpoint, else None."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # truncated/corrupt file
        print(f"  {path.name}: unreadable ({type(e).__name__})")
        return None
    if not isinstance(ckpt, dict) or ckpt.get("epoch", -1) < 0 or ckpt.get("optimizer") is None:
        print(f"  {path.name}: not resumable (finished or stripped)")
        return None
    if ckpt["epoch"] + 1 >= ckpt.get("train_args", {}).get("epochs", 0):
        print(f"  {path.name}: last epoch already trained")
        return None
    return ckpt


def training_finished(run: Path) -> bool:
    """Killed after the last epoch (e.g. during final validation): last.pt is stripped or complete, best.pt exists."""
    w = run / "weights"
    if not (w / "best.pt").exists() or not (w / "last.pt").exists():
        return False
    try:
        ckpt = torch.load(w / "last.pt", map_location="cpu", weights_only=False)
    except Exception:
        return False
    epochs = ckpt.get("train_args", {}).get("epochs", 0)
    return ckpt.get("epoch", -1) < 0 or ckpt["epoch"] + 1 >= epochs


def find_resume(run: Path) -> Path | None:
    w = run / "weights"
    snapshots = sorted(w.glob("epoch*.pt"), key=lambda p: int(p.stem[5:]), reverse=True)
    for p in [w / "last.pt", w / "last_safe.pt", *snapshots]:
        if p.exists() and load_resumable(p) is not None:
            return p
    return None


def make_smoke_data(stage: int) -> Path:
    """Random 200-image subsets of the stage lists plus a data.yaml pointing at them."""
    out = SPLITS / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    src = yaml.safe_load((SPLITS / f"stage{stage}.yaml").read_text())
    for key in ("train", "val"):
        lines = (SPLITS / src[key]).read_text().splitlines()
        (out / f"stage{stage}_{key}.txt").write_text("\n".join(random.Random(SEED).sample(lines, SMOKE["n_images"])) + "\n")
    cfg = {**src, "path": out.as_posix(), "train": f"stage{stage}_train.txt", "val": f"stage{stage}_val.txt"}
    p = out / f"stage{stage}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_run_info(run: Path, data_yaml: Path, resumed_from: Path | None) -> None:
    info_path = run / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    event = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "resumed_from": str(resumed_from) if resumed_from else None,
    }
    if not info:
        d = yaml.safe_load(data_yaml.read_text())
        info["data"] = {k: {"file": d[k], "md5": md5(Path(d["path"]) / d[k])} for k in ("train", "val")}
        info["data"]["yaml"] = str(data_yaml)
    info.setdefault("events", []).append(event)
    run.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2))


def keep_awake(on: bool) -> None:
    """Block system sleep for this process only (Windows SetThreadExecutionState)."""
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def safe_copy(src: Path, dst: Path) -> None:
    """Copy, fsync, then atomically replace dst so dst is always a complete file."""
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)


def drone_ap50(trainer) -> float | None:
    m = getattr(trainer.validator, "metrics", None)
    box = getattr(m, "box", None)
    if box is None or 0 not in list(getattr(box, "ap_class_index", [])):
        return None
    return float(box.ap50[list(box.ap_class_index).index(0)])


class Progress:
    """Overall progress line printed after every saved epoch (Ultralytics keeps its own batch bars)."""

    def __init__(self):
        self.t0, self.e0, self.best = None, 0, (None, -1)

    def on_train_start(self, trainer):
        self.t0, self.e0 = time.time(), trainer.start_epoch

    def on_model_save(self, trainer):
        done, total = trainer.epoch + 1, trainer.epochs
        ap = drone_ap50(trainer)
        if ap is not None and (self.best[0] is None or ap > self.best[0]):
            self.best = (ap, done)
        t = trainer.tloss
        if isinstance(t, dict):  # ultralytics 8.4 keeps named losses
            losses = {k.removesuffix("_loss"): float(v) for k, v in t.items()}
        else:
            losses = {k.removesuffix("_loss"): float(v) for k, v in zip(trainer.loss_names, t.tolist())} if t is not None else {}
        rate = (time.time() - self.t0) / max(1, done - self.e0)
        bar = tqdm.format_meter(done, total, time.time() - self.t0, ncols=40, prefix="epoch", unit="ep", ascii=True)
        bar = bar.split("[")[0].strip()  # keep "epoch:  23%|###   | 23/100"; ETA is computed below for this session
        best_ep = trainer.stopper.best_epoch  # already 1-based (stopper is called with epoch + 1)
        parts = [
            bar,
            f"box {losses.get('box', float('nan')):.2f} cls {losses.get('cls', float('nan')):.2f}",
            f"drone mAP50 {ap:.3f}" if ap is not None else "drone mAP50 n/a",
            f"(best {self.best[0]:.3f} @ep{self.best[1]})" if self.best[0] is not None else "",
            f"fitness best @ep{best_ep}",
            f"patience {done - best_ep}/{trainer.args.patience}",
            f"GPU {torch.cuda.memory_reserved() / 1e9:.1f}G" if torch.cuda.is_available() else "",
            f"ETA {tqdm.format_interval(rate * (total - done))}",
        ]
        tqdm.write("\n>> " + " | ".join(p for p in parts if p) + "\n")
        # crash-safe copy of the checkpoint Ultralytics just wrote
        last = Path(trainer.last)
        if last.exists():
            safe_copy(last, last.with_name("last_safe.pt"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["n", "s"], required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"train_stage{args.stage}.yaml").read_text())
    suffix = "-smoke" if args.smoke else ""
    name = f"stage{args.stage}-yolo26{args.model}{suffix}"
    run = RUNS / name

    if args.fresh and run.exists():
        old = run.with_name(f"{name}-old-{datetime.now():%Y%m%d-%H%M%S}")
        run.rename(old)
        print(f"--fresh: moved previous run to {old}")
    if (run / "done.json").exists():
        print(f"{name} already finished: {json.loads((run / 'done.json').read_text())}")
        print("use --fresh to train again")
        return

    from ultralytics import YOLO

    if run.exists() and training_finished(run):
        done = {"best": str(run / "weights" / "best.pt"), "note": "all epochs were trained; the process stopped before done.json",
                "finished": datetime.now().astimezone().isoformat(timespec="seconds")}
        (run / "done.json").write_text(json.dumps(done, indent=2))
        print(f"{name} had finished training: {done}\nuse --fresh to train again")
        return

    progress = Progress()
    resume = find_resume(run) if run.exists() else None
    if resume:
        print(f"resuming {name} from {resume}")
        model = YOLO(str(resume))
        train_kwargs = dict(resume=True)
        data_yaml = Path(yaml.safe_load((run / "args.yaml").read_text())["data"])
    else:
        if args.stage == 1:
            WEIGHTS.mkdir(parents=True, exist_ok=True)
            start = WEIGHTS / f"yolo26{args.model}.pt"  # auto-downloaded on first use
        else:
            start = RUNS / f"stage1-yolo26{args.model}{suffix}" / "weights" / "best.pt"
            if not start.exists():
                raise SystemExit(f"stage 2 needs {start}; run: uv run scripts/train.py --model {args.model} --stage 1{' --smoke' if args.smoke else ''}")
        data_rel = cfg.pop("data")
        data_yaml = make_smoke_data(args.stage) if args.smoke else ROOT / data_rel
        if args.smoke:
            cfg.update(epochs=SMOKE["epochs"], imgsz=SMOKE["imgsz"], save_period=SMOKE["save_period"])
        print(f"starting {name} from {start}")
        model = YOLO(str(start))
        train_kwargs = dict(data=str(data_yaml), project=str(RUNS), name=name, exist_ok=True, **cfg)

    model.add_callback("on_train_start", progress.on_train_start)
    model.add_callback("on_model_save", progress.on_model_save)
    write_run_info(run, data_yaml, resume)
    keep_awake(True)
    try:
        model.train(**train_kwargs)
    finally:
        keep_awake(False)
    trainer = model.trainer
    done = {
        "best": str(Path(trainer.best)),
        "epochs_run": trainer.epoch + 1,
        "best_epoch": trainer.stopper.best_epoch,
        "best_fitness": float(trainer.best_fitness or 0),
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (run / "done.json").write_text(json.dumps(done, indent=2))
    print(f"done: {done}")


if __name__ == "__main__":
    main()
