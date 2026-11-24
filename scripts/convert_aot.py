"""AOT subset -> data/yolo/aot (resized to 1920 long side). Spec §3, D1, D5.

Every frame from a flight that contains a Drone -> `test_aux` (air-to-air tiny-drone test).
Other flights -> train / val / test_fa = 70/15/15 by flight.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from common import RAW, YOLO, manifest_row, label_path, save_resized, split_groups, write_label, write_manifest, yolo_box

SRC = RAW / "aot"
DST = YOLO / "aot"
CLASS_ID = {"Drone": 0, "Bird": 1, "Airplane": 2, "Helicopter": 3}


def load_boxes(img_names: set[str]) -> pl.DataFrame:
    frames = []
    for part in ["part1", "part2", "part3"]:
        df = pl.read_csv(SRC / part / "ImageSets" / "groundtruth.csv", infer_schema_length=100_000)
        frames.append(df.filter(pl.col("img_name").is_in(img_names) & pl.col("id").is_not_null()))
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.with_columns(pl.col("id").str.replace(r"\d+$", "").str.to_titlecase().alias("cls"))


def convert_one(job: tuple[dict, list[tuple]]) -> dict:
    row, objs = job
    stem = Path(row["img_name"]).stem
    src_img = SRC / "images" / row["part"] / row["flight_id"] / f"{stem}.jpg"
    dst_img = DST / "images" / row["part"] / row["flight_id"] / f"{stem}.jpg"
    w, h = save_resized(src_img, dst_img)
    boxes = []
    for cls, sw, sh, l, t, r, b in objs:
        if cls not in CLASS_ID:
            raise ValueError(f"{row['img_name']}: unexpected class {cls!r}")
        sx, sy = w / sw, h / sh
        box = yolo_box(l * sx, t * sy, r * sx, b * sy, w, h)
        if box:
            boxes.append((CLASS_ID[cls], box))
    write_label(label_path(dst_img), boxes)
    return manifest_row(dst_img, "aot", row["flight_id"], row["split"], "a2a", boxes, w, h)


def main() -> None:
    subset = pl.read_csv(SRC / "subset_manifest.csv")
    drone_flights = set(subset.filter(pl.col("category") == "drone")["flight_id"])
    other = subset.filter(~pl.col("flight_id").is_in(drone_flights)).group_by("flight_id").len()
    assign = split_groups(dict(other.iter_rows()), {"train": 0.70, "val": 0.15, "test_fa": 0.15})
    assign |= dict.fromkeys(drone_flights, "test_aux")
    subset = subset.with_columns(pl.col("flight_id").replace_strict(assign).alias("split"))

    gt = load_boxes(set(subset["img_name"]))
    objs = {}
    for r in gt.select("img_name", "cls", "size_width", "size_height", "gt_left", "gt_top", "gt_right", "gt_bottom").iter_rows():
        objs.setdefault(r[0], []).append(r[1:])
    jobs = [(row, objs.get(row["img_name"], [])) for row in subset.to_dicts()]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(convert_one, jobs, chunksize=32))
    print(f"aot: {len(rows)} images, drone flights (test_aux): {len(drone_flights)}, other flights: {other.height}")
    write_manifest("aot", rows)


if __name__ == "__main__":
    main()
