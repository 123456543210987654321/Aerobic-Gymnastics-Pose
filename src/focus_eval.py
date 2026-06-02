from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


ACTION_NAMES = ["团身跳", "屈体分腿跳", "纵劈腿跳"]


@dataclass(frozen=True)
class ImageStat:
    split: str
    action: str
    image_path: str
    conf: float
    mean_kpt_conf: float
    min_kpt_conf: float
    kpt_ge_50: int
    kpt_ge_30: int


def iter_images(dataset_dir: Path, split: str, action: str) -> Iterable[Path]:
    images_dir = dataset_dir / "images" / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    # 文件名示例：团身跳_person_ID_10_1_frame_000028.jpg
    return sorted(images_dir.glob(f"{action}_*.jpg"))


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.min())


def compute_stats(
    *,
    split: str,
    action: str,
    image_path: Path,
    conf: float,
    result,
) -> ImageStat:
    if result.keypoints is None or result.keypoints.conf is None:
        kpt_conf = np.array([], dtype=np.float32)
    else:
        # shape: (num_instances, kpts)
        kpt_conf = result.keypoints.conf.cpu().numpy().astype(np.float32).reshape(-1)

    return ImageStat(
        split=split,
        action=action,
        image_path=str(image_path),
        conf=conf,
        mean_kpt_conf=safe_mean(kpt_conf),
        min_kpt_conf=safe_min(kpt_conf),
        kpt_ge_50=int((kpt_conf >= 0.50).sum()),
        kpt_ge_30=int((kpt_conf >= 0.30).sum()),
    )


def save_plot_image(out_path: Path, result) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()  # BGR uint8
    if plotted is None:
        return
    cv2.imwrite(str(out_path), plotted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--dataset", type=str, default="../YOLO_Dataset")
    parser.add_argument("--splits", type=str, default="val,test", help="comma-separated: val,test")
    parser.add_argument("--confs", type=str, default="0.25,0.15", help="comma-separated")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-per-action", type=int, default=0, help="0=all")
    parser.add_argument("--topk-hard", type=int, default=80)
    parser.add_argument("--outdir", type=str, default="runs/focus_eval")
    args = parser.parse_args()

    weights = Path(args.weights)
    dataset_dir = Path(args.dataset)
    outdir = Path(args.outdir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    confs = [float(x.strip()) for x in args.confs.split(",") if x.strip()]

    model = YOLO(str(weights))

    all_stats: list[ImageStat] = []

    for split in splits:
        for action in ACTION_NAMES:
            image_paths = list(iter_images(dataset_dir, split, action))
            if args.max_per_action and len(image_paths) > args.max_per_action:
                image_paths = image_paths[: args.max_per_action]

            for conf in confs:
                save_dir = outdir / f"{split}_{action}_conf{conf:.2f}".replace(".", "")
                # batch predict: keep deterministic order
                results = model.predict(
                    source=[str(p) for p in image_paths],
                    imgsz=args.imgsz,
                    conf=conf,
                    iou=0.7,
                    max_det=1,
                    verbose=False,
                )
                for p, r in zip(image_paths, results, strict=True):
                    stat = compute_stats(split=split, action=action, image_path=p, conf=conf, result=r)
                    all_stats.append(stat)
                    save_plot_image(save_dir / p.name, r)

    outdir.mkdir(parents=True, exist_ok=True)
    stats_csv = outdir / "stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "action",
                "image_path",
                "conf",
                "mean_kpt_conf",
                "min_kpt_conf",
                "kpt_ge_50",
                "kpt_ge_30",
            ]
        )
        for s in all_stats:
            writer.writerow(
                [
                    s.split,
                    s.action,
                    s.image_path,
                    f"{s.conf:.2f}",
                    f"{s.mean_kpt_conf:.6f}",
                    f"{s.min_kpt_conf:.6f}",
                    s.kpt_ge_50,
                    s.kpt_ge_30,
                ]
            )

    # hard-cases: mean_kpt_conf 最低优先，其次 min_kpt_conf
    for conf in confs:
        for split in splits:
            for action in ACTION_NAMES:
                subset = [
                    s
                    for s in all_stats
                    if s.conf == conf and s.split == split and s.action == action
                ]
                subset.sort(key=lambda s: (s.mean_kpt_conf, s.min_kpt_conf))
                hard = subset[: max(0, args.topk_hard)]
                hard_dir = outdir / "hard_cases" / f"conf{conf:.2f}".replace(".", "") / split / action
                hard_dir.mkdir(parents=True, exist_ok=True)
                index_path = hard_dir / "index.csv"
                with index_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["rank", "image_path", "mean_kpt_conf", "min_kpt_conf", "kpt_ge_50", "kpt_ge_30"])
                    for i, s in enumerate(hard, start=1):
                        w.writerow([i, s.image_path, f"{s.mean_kpt_conf:.6f}", f"{s.min_kpt_conf:.6f}", s.kpt_ge_50, s.kpt_ge_30])

                # 复制对应的可视化结果（便于快速人工检查）
                src_vis_dir = outdir / f"{split}_{action}_conf{conf:.2f}".replace(".", "")
                for s in hard:
                    src = src_vis_dir / Path(s.image_path).name
                    dst = hard_dir / src.name
                    if src.exists() and not dst.exists():
                        dst.write_bytes(src.read_bytes())

    print(f"[OK] wrote: {stats_csv}")
    print(f"[OK] visualizations under: {outdir}")


if __name__ == "__main__":
    main()

