"""Anti-UAV300 RGB test (91 sequences) -> data/yolo/antiuav. Spec §3. TEST ONLY.

Extracts every STRIDE-th frame of visible.mp4. exist=1 -> one `0 drone` box from gt_rect [x, y, w, h];
exist=0 -> empty label (background). Frames marked exist=1 with an all-zero rect are skipped.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from common import RAW, YOLO, label_path, manifest_row, write_label, write_manifest, yolo_box

SRC = RAW / "antiuav300" / "test"
DST = YOLO / "antiuav"
STRIDE = 5


def convert_seq(seq_dir: Path) -> tuple[list[dict], int, str | None]:
    ann = json.loads((seq_dir / "visible.json").read_text())
    exist, rects = ann["exist"], ann["gt_rect"]
    cap = cv2.VideoCapture(str(seq_dir / "visible.mp4"))
    rows, skipped, idx = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STRIDE == 0 and idx < len(exist):
            rect = rects[idx]
            if exist[idx] and (not rect or sum(rect) == 0):
                skipped += 1
            else:
                h, w = frame.shape[:2]
                dst_img = DST / "images" / seq_dir.name / f"{idx:06d}.jpg"
                if not dst_img.exists():
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst_img), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                boxes = []
                if exist[idx]:
                    x, y, bw, bh = rect
                    box = yolo_box(x, y, x + bw, y + bh, w, h)
                    if box:
                        boxes.append((0, box))
                write_label(label_path(dst_img), boxes)
                rows.append(manifest_row(dst_img, "antiuav", seq_dir.name, "test", "g2a", boxes, w, h))
        idx += 1
    cap.release()
    warn = None if idx == len(exist) else f"{seq_dir.name}: video {idx} frames vs json {len(exist)}"
    return rows, skipped, warn


def main() -> None:
    seqs = sorted(p for p in SRC.iterdir() if (p / "visible.mp4").exists())
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(convert_seq, seqs))
    for _, _, warn in results:
        if warn:
            print("WARN", warn)
    rows = [r for rs, _, _ in results for r in rs]
    print(f"antiuav: {len(seqs)} sequences, {len(rows)} frames, {sum(s for _, s, _ in results)} skipped (exist=1 without box)")
    write_manifest("antiuav", rows)


if __name__ == "__main__":
    main()
