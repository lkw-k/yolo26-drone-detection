"""DUT Anti-UAV (VOC xml) -> data/yolo/dut. Spec §3.

Official split: train -> train, val -> val, test -> test_aux. No sequence IDs, so group = dut_<split>.
"""

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import RAW, YOLO, image_size, label_path, link_image, manifest_row, write_label, write_manifest, yolo_box

SRC = RAW / "dut"
DST = YOLO / "dut"
SPLITS = {"train": "train", "val": "val", "test": "test_aux"}


def convert(args: tuple[Path, str]) -> tuple[dict, bool]:
    xml_path, raw_split = args
    root = ET.parse(xml_path).getroot()
    src_img = SRC / raw_split / "img" / root.findtext("filename")
    dst_img = DST / "images" / raw_split / src_img.name
    link_image(src_img, dst_img)
    w, h = image_size(dst_img)
    size_mismatch = (int(root.findtext("size/width")), int(root.findtext("size/height"))) != (w, h)
    boxes = []
    for obj in root.iter("object"):
        name = obj.findtext("name")
        if name != "UAV":
            raise ValueError(f"{xml_path}: unexpected class {name!r}")
        b = obj.find("bndbox")
        box = yolo_box(*(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")), w, h)
        if box:
            boxes.append((0, box))
    write_label(label_path(dst_img), boxes)
    return manifest_row(dst_img, "dut", f"dut_{raw_split}", SPLITS[raw_split], "g2a", boxes, w, h), size_mismatch


def main() -> None:
    jobs = [(p, s) for s in SPLITS for p in sorted((SRC / s / "xml").glob("*.xml"))]
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(convert, jobs, chunksize=64))
    print(f"dut: {len(results)} images, {sum(m for _, m in results)} xml sizes differ from the image (image size used)")
    write_manifest("dut", [r for r, _ in results])


if __name__ == "__main__":
    main()
