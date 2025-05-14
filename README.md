# 🧠 BThesis---Barbara-Koch

Welcome! This repository contains the code and supporting materials for my **Bachelor’s thesis**, which investigates how modern deep learning architectures perform on the task of segmenting brain tumors in MRI scans. The project compares two state-of-the-art models: **nnU-Net** (based on convolutional neural networks) and **Swin UNETR** (a transformer-based model), using the **BraTS2020** dataset.

> 🧩 **Note**: This repository builds upon publicly available implementations from:
> - [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) – the official nnU-Net codebase
> - [Inc0mple/3D_Brain_Tumor_Seg_V2](https://github.com/Inc0mple/3D_Brain_Tumor_Seg_V2) – Swin UNETR training pipeline and structure

These frameworks provided the foundation for model training, testing, and comparison, which were adapted for the purposes of this thesis.

---

## 🎯 Project Aim

The goal of this thesis was to evaluate and compare the performance of CNN-based and transformer-based approaches in the segmentation of glioma subregions:

- **Whole Tumor (WT)**
- **Tumor Core (TC)**
- **Enhancing Tumor (ET)**

By training both models under the same resource constraints and without pretrained weights, the project explores their strengths, limitations, and suitability for different clinical objectives.

---

## 🗂 Repository Structure

```text

This repository contains the full codebase and training pipeline for the bachelor’s thesis project comparing **nnU-Net** and **Swin UNETR** architectures on the **BraTS2020** dataset for 3D multimodal brain tumor segmentation.

```text
BThesis---Barbara-Koch/
│
├── nnUnet/                           # nnU-Net training and evaluation pipeline
│   ├── automated_pretraining/       # Preprocessing outputs
│   │   ├── created_fingerprint.json         # Dataset fingerprint (spacing, size, intensity)
│   │   └── nnUNetPlans.json                # Training plans and architecture configuration
│   │
│   ├── scripts/                     # Shell scripts to run training and prediction
│   │   ├── train.sh
│   │   ├── predict.sh
│   │   └── evaluate
│   │
│   ├── testing/                     # Inference results and adapted prediction script
│   │   ├── predict_from_raw_data.py       # Adapted inference script from nnUNetv2
│   │   ├── evaluation.json                # Summary evaluation results (Dice/IoU)
│   │   └── progress.png                   # Training progress visualization
│   │
│   ├── train/                       # Training logic and config
│   │   ├── run_training.py                # Launch script for nnU-Net training
│   │   ├── train.json                     # Training configuration
│   │   └── trainer.py                     # Trainer class
│   │
│   ├── logs/                        # Training logs
│   │   └── training_log.txt                # Epoch-wise performance and loss log
│   │
│   ├── statistical_analysis/       # 📊 Plots and scripts for analyzing nnU-Net output
│   │   ├── all_stats.py                    # Computes full per-class metrics
│   │   ├── dice-iou_bar.py                # Barplot of Dice and IoU by region
│   │   ├── class_mean_summary_stats.csv   # Summary stats (mean, std) for each class
│   │   ├── per_case_metrics.csv           # Case-wise evaluation metrics
│   │   ├── error_rate_by_class.png        # Visualization of classification error per class
│   │   └── iou_distribution.png           # Distribution of IoU scores
│
├── SWIN_UNETR/                    # Swin UNETR fine-tuning and evaluation
│   ├── scripts/
│   │   └── train.sh                      # Training shell script
│   │
│   ├── utils/
│   │   ├── BratsDataset.py              # BraTS dataset loader
│   │   ├── Meter.py                     # Metric logger
│   │   ├── fold_data.csv                # Fold splits for cross-validation
│   │   └── viz_eval_utils.py            # Visualization utility functions
│   │
│   ├── logs/                            # Logs and output of evaluation
│   │   ├── swinUNETR_test_results.csv           # Per-case results
│   │   ├── SwinUNETR_summary_statistics.csv     # Summary statistics
│   │   ├── SwinUNETR_dice_vs_jaccard_scatter.pdf # Scatterplot Dice vs Jaccard
│   │   ├── SwinUNETR_score_distributions.pdf    # Boxplot distributions
│   │   ├── metrics_barplot.png                  # Barplot of performance
│   │   ├── swinUNETR_trainer_properties.txt     # Full model config summary
│   │   └── train_log_20250414-004602.csv        # Epoch-wise training log
│   │
│   ├── models.py                     # Swin UNETR architecture (MONAI-based)
│   ├── train.py                      # Training script for Swin UNETR
│   ├── generate_your_json.py        # Script to generate dataset split JSON
│   └── vizualize_test.py            # 🔹 Adapted from Inc0mple’s visualization notebook
│
├── testing_significance/           # 📊 Statistical comparison of both models
│   ├── boxplot_dice_iou_synthetic.py     # Dice/IoU boxplots on test sets
│   └── wilcoxon_synthetic_analysis.py    # Wilcoxon test for paired model performance
│
└── README.md                       # This file – full project overview and documentation
             
```
## 🧠 Pretrained Model Weights

The pretrained weights for both segmentation models used in this project are available below:

- 🌀 **Swin UNETR** (Transformer-based):
  [Download `your_best_model_20250414-004601.pth`](https://drive.google.com/file/d/1lEjkvGCFt4yLCkP-OvhpKjnd4zrHOVT-/view?usp=drive_link)

- 🧩 **nnU-Net** (CNN-based):
  [Download `nnunet_best_model.pth`](https://drive.google.com/file/d/1HCK1qAsZj2TgxeGd8Rg4gVG81Z8gZodV/view?usp=sharing)

After downloading, place the weights in a suitable directory (e.g., `./weights/`) and update your loading scripts accordingly:

```python
# Example
model.load_state_dict(torch.load("weights/your_best_model_20250414-004601.pth"))

