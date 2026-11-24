"""GPU-side training augmentation for YOLO detection.

CPU workers only decode JPEGs into uint8 tensors. Everything else runs on the GPU:
mosaic (4 images from the same batch), random affine (scale/translate/rotate/shear),
HSV jitter and horizontal flip. Parameters and ordering follow Ultralytics v8_transforms:
    Mosaic -> RandomPerspective -> RandomHSV -> RandomFlip(lr)

Differences vs. Ultralytics (documented on purpose):
- Mosaic partners are drawn from the same mini-batch instead of the whole dataset,
  so each decoded image is reused instead of decoding 4 images per sample.
- Mosaic is applied per batch (hyp.mosaic is 1.0 by default, 0.0 after close_mosaic).
- Images stay float between steps (no intermediate uint8 rounding).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PAD_VALUE = 114.0


def img2label_path(p: str) -> str:
    sa, sb = f"{Path('/') / 'images' / ''}", f"{Path('/') / 'labels' / ''}"
    p = str(Path(p))
    return str(Path(sb.join(p.rsplit(sa, 1))).with_suffix(".txt"))


def read_label(path: str) -> np.ndarray:
    """Return (n, 5) float32 [cls, x, y, w, h] normalized; empty if missing."""
    try:
        rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return np.zeros((0, 5), np.float32)
    rows = [r for r in rows if len(r) == 5]
    return np.array(rows, np.float32).reshape(-1, 5)


class RawDetDataset(Dataset):
    """Returns decoded uint8 BGR images (imgsz x imgsz) and normalized labels."""

    def __init__(self, list_file: str, imgsz: int = 640):
        root = Path(list_file).parent
        files = [l.strip() for l in Path(list_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.im_files = [f if Path(f).is_absolute() else str((root / f).resolve()) for f in files]
        self.imgsz = imgsz
        self.mosaic = True
        arrs = [read_label(img2label_path(f)) for f in self.im_files]
        self._labels = arrs
        # Ultralytics-style label dicts, used by plot_training_labels().
        self.labels = [{"cls": a[:, :1], "bboxes": a[:, 1:]} for a in arrs]

    def close_mosaic(self, hyp=None) -> None:
        self.mosaic = False

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, i: int):
        path = self.im_files[i]
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        lab = self._labels[i].copy()
        s = self.imgsz
        h, w = im.shape[:2]
        if (h, w) != (s, s):  # letterbox to s x s, keep labels normalized to the new canvas
            r = s / max(h, w)
            nh, nw = round(h * r), round(w * r)
            im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            top, left = (s - nh) // 2, (s - nw) // 2
            canvas = np.full((s, s, 3), int(PAD_VALUE), np.uint8)
            canvas[top : top + nh, left : left + nw] = im
            im = canvas
            if len(lab):
                lab[:, 1] = (lab[:, 1] * nw + left) / s
                lab[:, 2] = (lab[:, 2] * nh + top) / s
                lab[:, 3] = lab[:, 3] * nw / s
                lab[:, 4] = lab[:, 4] * nh / s
        return torch.from_numpy(np.ascontiguousarray(im)), torch.from_numpy(lab), path


def collate_raw(batch):
    imgs = torch.stack([b[0] for b in batch])  # (B, H, W, 3) uint8 BGR
    labels = []
    for k, (_, lab, _) in enumerate(batch):
        labels.append(torch.cat([torch.full((len(lab), 1), float(k)), lab], 1))
    return {
        "raw_img": imgs,
        "raw_labels": torch.cat(labels, 0) if labels else torch.zeros((0, 6)),
        "im_file": [b[2] for b in batch],
    }


class GPUAugLoader(DataLoader):
    """DataLoader with the reset()/close() hooks the Ultralytics trainer calls."""

    def reset(self) -> None:  # mosaic state lives on the dataset and is read per batch
        pass


# ----------------------------------------------------------------------------- GPU ops


def rgb_to_hsv(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """img (B,3,H,W) RGB in [0,255] -> h in [0,180) (OpenCV scale), s in [0,1], v in [0,255]."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc, arg = img.max(1)
    minc = img.amin(1)
    delta = maxc - minc
    # Floors are fp16-safe (1/1e-2 = 100); pixel values are 0-255 so real deltas are >= 1e-2 anyway.
    s = delta / maxc.clamp(min=1e-2)
    inv = 1.0 / delta.clamp(min=1e-2)
    h = torch.where(arg == 0, (g - b) * inv, torch.where(arg == 1, 2.0 + (b - r) * inv, 4.0 + (r - g) * inv))
    h = (h * (180.0 / 6.0)) % 180.0  # gray pixels (delta=0) give h=0, which is harmless since s=0
    return h, s, maxc


_HSV_N = None


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Branch-free HSV->RGB: c_n = v - v*s*clamp(min(k, 4-k), 0, 1), k = (n + h/60) mod 6, n = 5,3,1."""
    global _HSV_N
    if _HSV_N is None or _HSV_N.device != h.device:
        _HSV_N = torch.tensor([5.0, 3.0, 1.0], device=h.device).view(1, 3, 1, 1)
    k = (_HSV_N + (h * (6.0 / 180.0)).unsqueeze(1)) % 6.0
    w = torch.minimum(k, 4.0 - k).clamp_(0.0, 1.0)
    v = v.unsqueeze(1)
    return v - v * s.unsqueeze(1) * w


def box_candidates(box1: np.ndarray, box2: np.ndarray, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


class GPUAugment:
    def __init__(self, hyp, imgsz: int = 640):
        self.s = int(imgsz)
        self.hyp = hyp

    def _mosaic(self, img: torch.Tensor, labels: list[np.ndarray]):
        """img (B,3,s,s) float on GPU; labels[k] (n,5) [cls, x1,y1,x2,y2] pixels."""
        B, s = img.shape[0], self.s
        S = 2 * s
        canvas = torch.full((B, 3, S, S), PAD_VALUE, device=img.device, dtype=img.dtype)
        partners = np.stack([np.arange(B)] + [np.random.permutation(B) for _ in range(3)], 1)
        out_labels = []
        for k in range(B):
            xc, yc = (int(random.uniform(s // 2, S - s // 2)) for _ in range(2))
            parts = []
            for t in range(4):
                j = int(partners[k, t])
                h = w = s
                if t == 0:
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif t == 1:
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, S), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif t == 2:
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(S, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                else:
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, S), min(S, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                canvas[k, :, y1a:y2a, x1a:x2a] = img[j, :, y1b:y2b, x1b:x2b]
                lab = labels[j].copy()
                if len(lab):
                    lab[:, [1, 3]] += x1a - x1b
                    lab[:, [2, 4]] += y1a - y1b
                parts.append(lab)
            lab = np.concatenate(parts, 0)
            lab[:, 1:] = lab[:, 1:].clip(0, S)
            keep = (lab[:, 3] > lab[:, 1]) & (lab[:, 4] > lab[:, 2])
            out_labels.append(lab[keep])
        return canvas, out_labels

    def _affine(self, canvas: torch.Tensor, labels: list[np.ndarray]):
        """Ultralytics RandomPerspective (perspective=0) via one batched grid_sample."""
        hyp, s = self.hyp, self.s
        B, _, Hin, Win = canvas.shape
        Ms, out_labels = [], []
        for k in range(B):
            C = np.eye(3)
            C[0, 2], C[1, 2] = -Win / 2, -Hin / 2
            R = np.eye(3)
            a = random.uniform(-hyp.degrees, hyp.degrees)
            sc = random.uniform(1 - hyp.scale, 1 + hyp.scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=sc)
            Sh = np.eye(3)
            Sh[0, 1] = math.tan(random.uniform(-hyp.shear, hyp.shear) * math.pi / 180)
            Sh[1, 0] = math.tan(random.uniform(-hyp.shear, hyp.shear) * math.pi / 180)
            T = np.eye(3)
            T[0, 2] = random.uniform(0.5 - hyp.translate, 0.5 + hyp.translate) * s
            T[1, 2] = random.uniform(0.5 - hyp.translate, 0.5 + hyp.translate) * s
            M = T @ Sh @ R @ C
            Ms.append(np.linalg.inv(M))

            lab = labels[k]
            if len(lab):
                b = lab[:, 1:]
                xy = np.ones((len(b) * 4, 3))
                xy[:, :2] = b[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)
                xy = (xy @ M.T)[:, :2].reshape(len(b), 8)
                x, y = xy[:, [0, 2, 4, 6]], xy[:, [1, 3, 5, 7]]
                nb = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], 1).clip(0, s)
                keep = box_candidates(b * sc, nb)
                lab = np.concatenate([lab[keep, :1], nb[keep]], 1).astype(np.float32)
            out_labels.append(lab)

        # Output pixel (u, v) samples input at Minv @ (u, v, 1); align_corners=True keeps cv2 pixel centers.
        Minv = torch.tensor(np.stack(Ms), device=canvas.device, dtype=torch.float32)
        if getattr(self, "_base", None) is None or self._base.device != canvas.device:
            v, u = torch.meshgrid(
                torch.arange(s, device=canvas.device, dtype=torch.float32),
                torch.arange(s, device=canvas.device, dtype=torch.float32),
                indexing="ij",
            )
            self._base = torch.stack([u, v, torch.ones_like(u)], -1).view(1, -1, 3)  # (1, s*s, 3)
        src = self._base @ Minv[:, :2].transpose(1, 2)  # (B, s*s, 2)
        gx = src[..., 0] * (2.0 / (Win - 1)) - 1.0
        gy = src[..., 1] * (2.0 / (Hin - 1)) - 1.0
        grid = torch.stack([gx, gy], -1).view(B, s, s, 2)
        out = F.grid_sample(canvas - PAD_VALUE, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return out + PAD_VALUE, out_labels

    def _hsv(self, img: torch.Tensor) -> torch.Tensor:
        hyp = self.hyp
        if not (hyp.hsv_h or hyp.hsv_s or hyp.hsv_v):
            return img
        B = img.shape[0]
        # fp16 is enough for 0-255 colour math (step 0.125 at 255) and halves memory traffic.
        img = img.half() if img.is_cuda else img
        gains = torch.empty((B, 3), device=img.device).uniform_(-1, 1)
        gains = (gains * torch.tensor([hyp.hsv_h, hyp.hsv_s, hyp.hsv_v], device=img.device) + 1).to(img.dtype)
        h, s, v = rgb_to_hsv(img)
        h = (h * gains[:, 0, None, None]) % 180.0
        s = (s * 255.0 * gains[:, 1, None, None]).clamp(0, 255) / 255.0
        v = (v * gains[:, 2, None, None]).clamp(0, 255)
        return hsv_to_rgb(h, s, v)

    @torch.no_grad()
    def __call__(self, raw_img: torch.Tensor, raw_labels: torch.Tensor, mosaic: bool):
        """raw_img (B,H,W,3) uint8 BGR on GPU; raw_labels (N,6) [bi, cls, x, y, w, h] normalized (CPU).

        Returns img (B,3,s,s) float [0,1] RGB, cls (N,1), bboxes (N,4) normalized xywh, batch_idx (N,).
        """
        s, hyp = self.s, self.hyp
        B = raw_img.shape[0]
        img = raw_img.permute(0, 3, 1, 2).flip(1).float()  # BGR -> RGB, (B,3,s,s)

        lab_np = raw_labels.numpy()
        labels = []
        for k in range(B):
            l = lab_np[lab_np[:, 0] == k, 1:]
            xyxy = np.empty_like(l)
            xyxy[:, 0] = l[:, 0]
            xyxy[:, 1] = (l[:, 1] - l[:, 3] / 2) * s
            xyxy[:, 2] = (l[:, 2] - l[:, 4] / 2) * s
            xyxy[:, 3] = (l[:, 1] + l[:, 3] / 2) * s
            xyxy[:, 4] = (l[:, 2] + l[:, 4] / 2) * s
            labels.append(xyxy)

        use_mosaic = mosaic and random.random() < hyp.mosaic
        canvas, labels = self._mosaic(img, labels) if use_mosaic else (img, labels)
        img, labels = self._affine(canvas, labels)
        img = self._hsv(img)

        flip = torch.rand(B) < hyp.fliplr
        if flip.any():
            idx = flip.nonzero().flatten().to(img.device)
            img[idx] = img[idx].flip(-1)
            for k in flip.nonzero().flatten().tolist():
                if len(labels[k]):
                    x1 = s - labels[k][:, 3]
                    x2 = s - labels[k][:, 1]
                    labels[k][:, 1], labels[k][:, 3] = x1, x2

        cls, boxes, bidx = [], [], []
        for k, l in enumerate(labels):
            if not len(l):
                continue
            xywh = np.stack(
                [(l[:, 1] + l[:, 3]) / 2 / s, (l[:, 2] + l[:, 4]) / 2 / s, (l[:, 3] - l[:, 1]) / s, (l[:, 4] - l[:, 2]) / s],
                1,
            )
            cls.append(l[:, :1])
            boxes.append(xywh)
            bidx.append(np.full(len(l), k, np.float32))
        dev = img.device
        if cls:
            cls_t = torch.from_numpy(np.concatenate(cls).astype(np.float32)).to(dev)
            box_t = torch.from_numpy(np.concatenate(boxes).astype(np.float32)).to(dev)
            bidx_t = torch.from_numpy(np.concatenate(bidx)).to(dev)
        else:
            cls_t, box_t, bidx_t = (torch.zeros((0, 1), device=dev), torch.zeros((0, 4), device=dev), torch.zeros(0, device=dev))
        return (img.float() / 255.0).clamp_(0, 1), cls_t, box_t, bidx_t
