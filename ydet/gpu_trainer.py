"""DetectionTrainer that moves training augmentation to the GPU.

Only the training dataloader and preprocess_batch are replaced. Validation, early
stopping, EMA, checkpoints, plots and DDP are the stock Ultralytics code paths.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DistributedSampler
from ultralytics.models.yolo.detect import DetectionTrainer

from ydet.gpu_aug import GPUAugLoader, GPUAugment, RawDetDataset, collate_raw


class GPUAugTrainer(DetectionTrainer):
    def get_dataloader(self, dataset_path, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        dataset = RawDetDataset(dataset_path, imgsz=self.args.imgsz)
        sampler = DistributedSampler(dataset, shuffle=True, seed=self.args.seed) if rank != -1 else None
        n_gpu = max(torch.cuda.device_count(), 1)
        workers = max(min((os.cpu_count() or 1) // n_gpu, self.args.workers), 0)
        gen = torch.Generator()
        gen.manual_seed(self.args.seed + max(rank, 0))
        self.gpu_aug = GPUAugment(self.args, self.args.imgsz)
        return GPUAugLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_raw,
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
            drop_last=False,
            generator=gen,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        if "raw_img" not in batch:
            return super().preprocess_batch(batch)
        raw = batch["raw_img"].to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, enabled=False):
            img, cls, bboxes, bidx = self.gpu_aug(raw, batch["raw_labels"], mosaic=self.train_loader.dataset.mosaic)
        return {"img": img, "cls": cls, "bboxes": bboxes, "batch_idx": bidx, "im_file": batch["im_file"]}
