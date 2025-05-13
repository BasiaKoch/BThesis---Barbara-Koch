"""
Synthetic Performance Visualization for Tumor Segmentation

This script simulates Dice and IoU scores for two segmentation models — Swin UNETR and nnU-Net — 
based on reported means and standard deviations from experiments on the BraTS2020 dataset.

It generates synthetic distributions using Gaussian sampling and visualizes the results as boxplots 
to facilitate a direct comparison across three clinically relevant tumor subregions:
    - WT: Whole Tumor
    - TC: Tumor Core
    - ET: Enhancing Tumor

Key Features:
- Generates 73 synthetic samples per model/region using normally distributed values clipped to [0, 1]
- Visualizes performance variability and central tendency using boxplots
- Reproducibility ensured through a fixed random seed (np.random.seed(42))
- Organized subplot layout: top row for Dice scores, bottom row for IoU scores
"""

import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

n = 73  # number of synthetic samples

# Reported means and standard deviations for Dice and Jaccard
mean_std = {
    'Dice': {
        'SwinUNETR': {'WT': (0.8378, 0.1213), 'TC': (0.6559, 0.2626), 'ET': (0.$
        'nnU-Net':   {'WT': (0.7899, 0.1213), 'TC': (0.6514, 0.2625), 'ET': (0.$
    },
    'IoU': {
        'SwinUNETR': {'WT': (0.7237, 0.1445), 'TC': (0.5939, 0.2607), 'ET': (0.$
        'nnU-Net':   {'WT': (0.6902, 0.1445), 'TC': (0.5964, 0.2607), 'ET': (0.$
    }
}

# Generate synthetic scores clipped to [0,1]
def generate_scores(mean, std):
    return np.clip(np.random.normal(loc=mean, scale=std, size=n), 0, 1)

# Setup figure with 2 rows: Dice (top), IoU (bottom)
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey='row')

metrics = ['Dice', 'IoU']
regions = ['WT', 'TC', 'ET']

for row, metric in enumerate(metrics):
    for col, region in enumerate(regions):
        swin_scores = generate_scores(*mean_std[metric]['SwinUNETR'][region])
        nnunet_scores = generate_scores(*mean_std[metric]['nnU-Net'][region])

        axs[row, col].boxplot(
            [swin_scores, nnunet_scores],
            labels=['Swin UNETR', 'nnU-Net'],
            patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor='red', marker='o', markersize=4, li$
        )
        axs[row, col].set_title(f'{region} {metric} Comparison')
        axs[row, col].grid(True, linestyle='--', alpha=0.6)

