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
BThesis---Barbara-Koch/
│
├── nnUnet/                     # Training and evaluation of nnU-Net
├── SWIN_UNETR/                 # Fine-tuning Swin UNETR for BraTS2020
├── testing_significance/       # Scripts for statistical analysis (e.g., Wilcoxon test)
├── synthetic_visualization.py  # Script to simulate and visualize Dice/IoU distributions
└── README.md                   # You're here!
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

