# GymPose: Automatic Gymnastics Pose Estimation & Scoring

> **Vision-Based Technical Motion Analysis for Competitive Aerobics**  
> *Chongqing Jiaotong University*  
> 2024.11 – 2026.05

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Pose-0b9fef.svg)](https://github.com/ultralytics/ultralytics)

---

## Project Background

This project is part of the innovation and entrepreneurship training program at **Chongqing Jiaotong University**. It addresses the long-standing subjectivity and inconsistency in competitive aerobics judging by introducing an objective, vision-based automatic scoring pipeline.

We leverage **YOLOv8-Pose** for 17-keypoint human pose estimation on three core aerobics jumping maneuvers — **Tuck Jump** (团身跳), **Pike Straddle Jump** (屈体分腿跳), and **Split Leap** (纵劈腿跳) — and combine kinematic feature extraction (joint angles, buffer amplitude, symmetry, etc.) with an **Entropy-Weight TOPSIS** dual-ideal-solution algorithm to produce objective scores in the [8.0, 10.0] range.

### Key Features

| Feature | Approach |
|---------|----------|
| 17-Keypoint Detection | YOLOv8-nano-Pose (COCO format, 640×640) |
| 3-Class Action Evaluation | Tuck Jump / Pike Straddle Jump / Split Leap |
| Kinematic Feature Extraction | 7 joint angles (knee/hip/ankle/split) + buffer amplitude/duration/symmetry |
| Automated Scoring | Entropy-weight method + dual-ideal-solution TOPSIS (score mapped to 8–10) |
| Judge Consistency Validation | Pearson & Spearman correlation tests |

---

## Project Structure

```
GymPose/
├── README.md
├── requirements.txt              # Python dependencies
├── .gitignore                    # Exclude model weights / training outputs / private files
├── data.yaml                     # Dataset config (COCO 17-keypoint)
│
├── src/                          # Core code (training / evaluation)
│   ├── train.py                  #   YOLOv8-Pose training pipeline
│   └── focus_eval.py             #   Per-image focused evaluation + hard-case mining
│
├── scripts/                      # Scoring & analysis pipeline (run in numbered order)
│   ├── 01_topsis_scoring.py      #   Label parsing → joint angles → TOPSIS scoring
│   ├── 02_merge_scores.py        #   Merge human-judge scores with TOPSIS results
│   ├── 03_visualize_results.py   #   Correlation / error / distribution visualizations
│   └── 04_angle_charts.py        #   Joint angle academic charts (heatmaps / density)
│
├── outputs/                      # Output data
│   ├── training_results.csv      #   Raw training metrics
│   ├── gymnastics_joint_angles_full.csv  #   Full joint angle dataset
│   ├── evaluation_metrics.csv    #   Per-athlete evaluation indicators
│   ├── final_scores.xlsx         #   TOPSIS scoring results
│   └── final_scores_with_judges.csv  #   Merged with human-judge scores
│
├── figures/                      # Academic-paper figures
│   ├── training_results_total.png    #   Comprehensive training curves (6-in-1)
│   ├── confusion_matrix*.png         #   Confusion matrices (standard + normalized)
│   ├── test_confusion_matrix*.png    #   Test set confusion matrices
│   ├── scenario_*.jpg                #   Multi-scenario training samples (dim/bright/varying)
│   ├── detection_example.jpg         #   Detection examples
│   ├── val_batch*_pred.jpg           #   Validation batch predictions
│   ├── test_batch*_pred.jpg          #   Test batch predictions
│   ├── hardcases_*.jpg               #   Hard-case mining examples
│   ├── opt_threshold_table.png       #   Threshold optimization table
│   └── (scoring analysis figures)
│
└── runs/                         # Training run outputs (gitignored)
    ├── eval/
    │   ├── test_conf025_iou070/  #   Test set evaluation
    │   └── val_conf025_iou070/   #   Validation set evaluation
    └── opt_figures/              #   Optimization & hard-case analysis
```

---

## Dataset

### Statistics

| Action | Athletes | Train | Val | Test |
|--------|----------|-------|-----|------|
| Tuck Jump | 12 | 2159 | 235 | 673 |
| Pike Straddle Jump | 15 | 1685 | 720 | 440 |
| Split Leap | 14 | 1439 | 274 | 610 |
| **Total** | — | **5283** | **1229** | **1723** |

- Total samples: **8,235 images**
- Annotation format: YOLO Pose (`kpt_shape=[17, 3]`, COCO 17-keypoint)
- Split strategy: **athlete-based partitioning** to prevent intra-subject leakage
- Scene conditions: indoor gym with 3 lighting scenarios (dim / bright / varying + partial occlusion)

### Multi-Scenario Samples

| Dim Lighting | Bright Lighting | Varying + Occlusion |
|:------------:|:---------------:|:-------------------:|
| ![Dim](figures/scenario_dim.jpg) | ![Bright](figures/scenario_bright.jpg) | ![Varying](figures/scenario_varying.jpg) |

> *Training batch mosaics from 3 lighting conditions. The dataset covers diverse illumination to ensure robust deployment in real competition venues.*

### File Naming Convention

```
{action}_person_ID_{athlete_id}_{sequence}_frame_{frame_num}.jpg
```

Example: `tuck_jump_person_ID_10_1_frame_000028.jpg`

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/GymPose.git
cd GymPose

# Install dependencies
pip install -r requirements.txt

# (Recommended) Install PyTorch with CUDA for GPU training
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Dataset

Organize your dataset as follows and update the `path` field in `data.yaml`:

```
YOLO_Dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 3. Training

```bash
# Download pretrained weights to project root
# wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt

# Start training (150 epochs)
python src/train.py
```

### 4. Evaluation

```bash
# Standard evaluation (val + test, conf=0.25)
yolo pose val model=best.pt data=data.yaml split=val
yolo pose val model=best.pt data=data.yaml split=test

# Focused per-image evaluation (multi-threshold + hard-case mining)
python src/focus_eval.py --weights best.pt --splits val,test --confs 0.25,0.15
```

### 5. Automated Scoring Pipeline

```bash
# Execute the scoring pipeline in numbered order
cd scripts

python 01_topsis_scoring.py      # Joint angles → TOPSIS scoring
python 02_merge_scores.py        # Merge human-judge scores
python 03_visualize_results.py   # Scoring comparison visualizations
python 04_angle_charts.py        # Joint angle academic charts
```

---

## Results

### Keypoint Detection Performance

| Metric | Validation | Test |
|--------|-----------|------|
| Box mAP50-95 | 0.804 | 0.913 |
| Pose mAP50-95 | **0.592** | **0.794** |
| Pose Precision | 0.702 | 0.863 |
| Pose Recall | 0.709 | 0.864 |

> *Note:* The validation set contains more occlusion/motion-blur hard cases, yielding stricter metrics. Test set performance approaches practical deployment levels.

![Training Results](figures/training_results_total.png)

![Confusion Matrix (Validation)](figures/confusion_matrix.png)

![Confusion Matrix (Test)](figures/test_confusion_matrix.png)

### Qualitative Detection Examples

**Validation Set:**

![Val Batch Sample](figures/detection_example.jpg)

**Test Set:**

![Test Batch 0](figures/test_batch0_pred.jpg)

![Test Batch 1](figures/test_batch1_pred.jpg)

![Test Batch 2](figures/test_batch2_pred.jpg)

> *17-keypoint skeleton predictions on test set. Test set (8.0–10.0 level) shows cleaner poses and higher detection confidence compared to validation set (5.0–7.5 level) which contains more occlusion and motion blur.*

### Hard-Case Analysis

| Tuck Jump | Pike Straddle Jump | Split Leap |
|:---------:|:------------------:|:----------:|
| ![Tuck Jump Hardcases](figures/hardcases_tuck_jump.jpg) | ![Pike Straddle Hardcases](figures/hardcases_pike_straddle.jpg) | ![Split Leap Hardcases](figures/hardcases_split_leap.jpg) |

![Threshold Optimization](figures/opt_threshold_table.png)

> *Hard-case mining identifies low-confidence samples (occlusion, motion blur, extreme poses). Threshold optimization balances detection recall vs. false positives.*

### Scoring Consistency with Human Judges

| Method | Coefficient | Significance |
|--------|-------------|--------------|
| Pearson Correlation | ~0.78 | p < 0.001 (**highly significant**) |
| Spearman Rank Correlation | ~0.75 | p < 0.001 |
| Mean Absolute Error | ~0.12 pts | — |

> The Entropy-Weight TOPSIS scores show **highly significant positive correlation** with expert judge scores.

![Correlation Scatter](figures/1_correlation_scatter.png)

![Ranking Consistency](figures/6_ranking_consistency.png)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n-Pose |
| Input Size | 640×640 |
| Epochs | 150 (patience=50 early stopping) |
| Batch Size | 64 |
| Optimizer | AdamW (lr₀=0.0005, cosine annealing) |
| Augmentation | Mosaic(1.0) + HSV(h=0.015, s=0.5, v=0.4) |
| Device | NVIDIA GPU (single card) |

---

## Methodology

### Model Architecture

We use **YOLOv8n-Pose** as the baseline, outputting 17 COCO-format human keypoints. Key adaptations for competitive aerobics:

- **Multi-lighting augmentation**: HSV jitter to handle 3 indoor lighting conditions
- **Mosaic augmentation**: Improves generalization on complex pose combinations
- **Motion blur + Random Erasing**: Simulates mid-air ghosting and occlusion scenarios
- **Cosine annealing + Early stopping**: Prevents overfitting

### Scoring Algorithm: Entropy-Weight TOPSIS with Dual Ideal Solutions

1. **Kinematic feature extraction**: compute 7 joint angles (knee/hip/ankle/split) from 17 keypoints
2. **Evaluation indicator system**:
   - Maximize-type: split angle, jump height
   - Minimize-type: joint symmetry discrepancy, angle standard deviation
   - Intermediate-type: buffer amplitude, change rate, buffer duration
3. **Entropy-weight method**: objectively determine indicator weights
4. **Dual ideal-solution TOPSIS**:
   - Positive ideal solution: best-performing athlete as reference
   - Negative ideal solution: worst-performing athlete as contrast
   - Closeness coefficient mapped to [8.0, 10.0] scoring range

---

## Team

| Role | Major |
|------|-------|
| Project Lead | Computer Science & Technology |
| Member | Statistics |
| Member | Computer Science & Technology |

**Institution**: Chongqing Jiaotong University

**Contributed by**: [@123456543210987654321](https://github.com/123456543210987654321),[@Nolkee](https://github.com/Nolkee),

---

## Figures Gallery

### Per-Action Correlation Heatmaps

| Tuck Jump | Pike Straddle Jump | Split Leap |
|:---------:|:------------------:|:----------:|
| ![Tuck Jump](figures/3_tuck_jump_correlation_heatmap.png) | ![Pike Straddle](figures/3_pike_straddle_correlation_heatmap.png) | ![Split Leap](figures/3_split_leap_correlation_heatmap.png) |

### Joint Angle Analysis

![Knee Angle Density](figures/2_knee_angle_density.png)

![3-Joint Pairwise Distribution](figures/5_3joint_pairwise_distribution.png)

### Per-Sample Score Comparison

![Per-Sample Comparison](figures/2_per_sample_comparison.png)

---

## License

This repository is for academic research and educational purposes only. Dataset and model weights are copyrighted by the project team.

---

## References

1. Glenn Jocher, et al. *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
2. Hwang C L, Yoon K. *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag, 1981.
3. FIG 2025–2028 Code of Points – Aerobic Gymnastics
