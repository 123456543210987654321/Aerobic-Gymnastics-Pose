from ultralytics import YOLO
from pathlib import Path
import shutil
import sys
import pandas as pd
import os

# YOLOv8-Pose 训练脚本（支持断点续训）
print("-" * 60)
print("   Gymnastics Pose Estimation - 训练")
print("-" * 60)
sys.stdout.flush()

# 路径配置
# 项目根目录（src/ 的上级）
BASE_DIR = Path(__file__).resolve().parent.parent

pretrain_pt_path = str(BASE_DIR / "yolov8n-pose.pt")
data_yaml_path = str(BASE_DIR / "data.yaml")

# 自动查找最新的 last.pt（断点续训）
last_candidates = list(BASE_DIR.rglob("last.pt"))

if len(last_candidates) > 0:
    last_pt_path = str(sorted(last_candidates, key=lambda x: x.stat().st_mtime)[-1])
    model = YOLO(last_pt_path)
    print(f"   找到 last.pt -> 自动续训")
    print(f"   路径: {last_pt_path}")
    resume_flag = True
else:
    if Path(pretrain_pt_path).exists():
        model = YOLO(pretrain_pt_path)
        print("   未找到 last.pt -> 使用本地预训练权重开始训练")
        print(f"   权重路径: {pretrain_pt_path}")
        resume_flag = False
    else:
        print(f"   预训练权重不存在！请确认文件在：{pretrain_pt_path}")
        sys.exit(1)

sys.stdout.flush()

# 开始训练
results = model.train(
    data=data_yaml_path,
    epochs=150,
    imgsz=640,
    batch=64,
    device=0,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    warmup_epochs=10,
    cos_lr=True,
    mosaic=1.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    close_mosaic=20,
    patience=50,
    save=True,
    save_period=10,
    exist_ok=True,
    name="GymFinal_v7",
    project="runs/GymPose",
    verbose=True,
    plots=True,
    resume=resume_flag
)

# 训练结束后自动复制 best.pt 到项目根目录
best_pt = BASE_DIR / "runs" / "pose" / "runs" / "GymPose" / "GymFinal_v7" / "weights" / "best.pt"

if best_pt.exists():
    shutil.copy2(best_pt, str(BASE_DIR / "best.pt"))
    print(f"\n   最佳权重已复制 -> {BASE_DIR / 'best.pt'}")
else:
    print("\n   本次训练尚未产生 best.pt（继续跑就行）")

    # 获取当前 best epoch 信息
    csv_path = BASE_DIR / "runs" / "pose" / "runs" / "GymPose" / "GymFinal_v7" / "results.csv"
    best_info = "无法读取 results.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "metrics/mAP50-95(P)" in df.columns:
                best_row = df.loc[df["metrics/mAP50-95(P)"].idxmax()]
                epoch = int(best_row["epoch"])
                mAP = best_row["metrics/mAP50-95(P)"]
                best_info = f"当前 best.pt 来自第 {epoch} epoch\nPose mAP50-95 = {mAP:.4f}"
            else:
                best_info = "未找到 mAP 指标列"
        except Exception as e:
            best_info = f"读取 CSV 失败: {e}"

sys.stdout.flush()
print("-" * 60)
print("   训练脚本执行完毕！")
