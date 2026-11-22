"""Stage 1 pretraining on seraphim. Same config for n and s (only --model differs).

Paths are resolved from the repo root so the same script runs on Windows and Linux.

Usage:
    uv run python scripts/train_stage1.py --model n                                   # local, 1 GPU
    uv run python scripts/train_stage1.py --model n --cfg configs/stage1_train_server.yaml --device 0,1,2
    uv run python scripts/train_stage1.py --model n --resume                          # continue after a crash
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def write_data_yaml() -> Path:
    """Write a data yaml with an absolute path for this machine."""
    base = yaml.safe_load((ROOT / "configs" / "seraphim.yaml").read_text())
    base["path"] = (ROOT / "data" / "stage1").as_posix()
    out = ROOT / "data" / "stage1" / "seraphim_local.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["n", "s"], required=True)
    ap.add_argument("--cfg", default="configs/stage1_train.yaml")
    ap.add_argument("--device", default="0", help="e.g. 0 or 0,1,2 for DDP")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu-aug", action="store_true", help="run training augmentation on the GPU (ydet.gpu_trainer)")
    ap.add_argument("--name", default=None, help="run name (default yolo26{model}_640)")
    args = ap.parse_args()

    trainer = None
    if args.gpu_aug:
        # DDP workers re-import the trainer class, so the repo root must be on PYTHONPATH.
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
        sys.path.insert(0, str(ROOT))
        from ydet.gpu_trainer import GPUAugTrainer

        trainer = GPUAugTrainer

    name = args.name or f"yolo26{args.model}_640"
    project = ROOT / "runs" / "stage1"
    if args.resume:
        YOLO(str(project / name / "weights" / "last.pt")).train(resume=True, trainer=trainer)
        return

    assert (ROOT / "data" / "stage1" / "leakage" / "exclude.txt").exists(), "run leakage_check.py first"
    assert (ROOT / "data" / "stage1" / "lists" / "seraphim_train.txt").exists(), "run make_stage1_lists.py first"

    model = YOLO(str(ROOT / "weights" / f"yolo26{args.model}.pt"))
    model.train(
        cfg=str(ROOT / args.cfg),
        data=str(write_data_yaml()),
        project=str(project),
        name=name,
        device=args.device,
        exist_ok=False,
        trainer=trainer,
    )


if __name__ == "__main__":
    main()
