"""Stage 2 main training, starting from the stage 1 best weights. Same config for n and s.

Usage:
    uv run python scripts/train_stage2.py --model n --init runs/stage1/yolo26n_640_gpuaug_b96/weights/best.pt \
        --device 0,1,2 --gpu-aug
    uv run python scripts/train_stage2.py --model n --resume --gpu-aug      # continue after a crash
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
    base = yaml.safe_load((ROOT / "configs" / "stage2.yaml").read_text())
    base["path"] = (ROOT / "data" / "stage2").as_posix()
    out = ROOT / "data" / "stage2" / "stage2_local.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["n", "s"], required=True)
    ap.add_argument("--init", type=Path, help="stage 1 best.pt (required unless --resume)")
    ap.add_argument("--cfg", default="configs/stage2_train_server.yaml")
    ap.add_argument("--device", default="0,1,2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu-aug", action="store_true")
    ap.add_argument("--name", default=None, help="run name (default yolo26{model}_1280)")
    args = ap.parse_args()

    trainer = None
    if args.gpu_aug:
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
        sys.path.insert(0, str(ROOT))
        from ydet.gpu_trainer import GPUAugTrainer

        trainer = GPUAugTrainer

    name = args.name or f"yolo26{args.model}_1280"
    project = ROOT / "runs" / "stage2"
    if args.resume:
        YOLO(str(project / name / "weights" / "last.pt")).train(resume=True, trainer=trainer)
        return

    assert args.init and args.init.exists(), "--init must point to the stage 1 best.pt"
    assert f"yolo26{args.model}" in args.init.as_posix(), "--init must be the same model size"
    for f in ["lists/stage2_train.txt", "lists/stage2_val.txt", "dedup/exclude.txt"]:
        assert (ROOT / "data" / "stage2" / f).exists(), f"missing data/stage2/{f}"

    YOLO(str(args.init)).train(
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
