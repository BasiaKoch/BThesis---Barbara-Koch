# BThesis---Barbara-Koch
Welcome! This repository contains the code and supporting materials for my Bachelor’s thesis, which investigates how modern deep learning architectures perform on the task of segmenting brain tumors in MRI scans. The project compares two state-of-the-art models: nnU-Net (based on convolutional neural networks) and Swin UNETR (a transformer-based model), using the BraTS2020 dataset.

🎯 Project Aim
The goal of this thesis was to evaluate and compare the performance of CNN-based and transformer-based approaches in the segmentation of glioma subregions:

Whole Tumor (WT)
Tumor Core (TC)
Enhancing Tumor (ET)

By training both models under the same resource constraints and without pretrained weights, the project explores their strengths, limitations, and suitability for different clinical objectives.

🗂 Repository Structure

BThesis---Barbara-Koch/
│
├── nnUnet/                  # Training and evaluation of nnU-Net
├── SWIN_Unetr/              # Fine-tuning Swin UNETR for BraTS2020
├── testing_significance/    # Scripts for statistical analysis (e.g., Wilcoxon test)
├── synthetic_visualization.py  # Script to simulate and visualize Dice/IoU distributions
└── README.md                # You're here!

🔍 What You’ll Find
This project combines practical implementation, quantitative evaluation, and statistical testing:

📈 Model training and evaluation: Run on a resource-limited setup using a fixed split of BraTS2020.

📊 Performance metrics: Dice and IoU scores computed per region and model.

🔬 Statistical analysis: Wilcoxon signed-rank tests are used to assess whether differences in performance are statistically significant.

🧪 Synthetic visualizations: Simulated boxplots based on reported means and standard deviations, helping illustrate result distributions.

