"""seraphim -> data/yolo/seraphim (hardlinked images, validated labels). Spec §3.

Raw train -> split `train`, raw test -> split `val`. Single class `0 drone`.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import RAW, YOLO, clip_yolo_line, image_size, label_path, link_image, manifest_row, write_label, write_manifest

SRC = RAW / "seraphim"
DST = YOLO / "seraphim"
SPLITS = {"train": "train", "test": "val"}


def convert(args: tuple[Path, str]) -> tuple[dict, int]:
    src_img, raw_split = args
    dst_img = DST / "images" / raw_split / src_img.name
    link_image(src_img, dst_img)
    w, h = image_size(dst_img)
    boxes, dropped = [], 0
    src_lbl = SRC / raw_split / "labels" / (src_img.stem + ".txt")
    lines = src_lbl.read_text().splitlines() if src_lbl.exists() else []
    for line in lines:
        v = line.split()
        if not v:
            continue
        box = clip_yolo_line(*map(float, v[1:5]), w, h) if len(v) == 5 and v[0] == "0" else None
        if box is None:
            dropped += 1
        else:
            boxes.append((0, box))
    write_label(label_path(dst_img), boxes)
    row = manifest_row(dst_img, "seraphim", f"seraphim_{raw_split}_{src_img.stem}", SPLITS[raw_split], "mixed", boxes, w, h)
    return row, dropped


def main() -> None:
    jobs = [(p, s) for s in SPLITS for p in sorted((SRC / s / "images").iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(convert, jobs, chunksize=256))
    print(f"seraphim: {len(results)} images, {sum(d for _, d in results)} invalid label lines dropped")
    write_manifest("seraphim", [r for r, _ in results])


if __name__ == "__main__":
    main()
