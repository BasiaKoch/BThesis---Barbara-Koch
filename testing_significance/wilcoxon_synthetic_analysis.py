"""
Wilcoxon Signed-Rank Test on Synthetic Segmentation Scores

This script simulates 73 paired samples of segmentation metrics (Dice and Jaccard)
for two models: Swin UNETR and nnU-Net. It approximates real model output using
summary statistics (mean, std, min, max) reported in the thesis. The Wilcoxon
signed-rank test is applied to assess statistical significance of performance differences
between models across three tumor regions (WT, TC, ET).

Note: This does not use real model predictions but synthetic approximations.
"""
import numpy as np
from scipy.stats import wilcoxon

np.random.seed(42)
n = 73

# Helper to generate synthetic samples within min/max bounds
def generate_scores(mean, std, min_val, max_val):
    scores = np.random.normal(loc=mean, scale=std, size=n)
    return np.clip(scores, min_val, max_val)

# Swin UNETR performance summary
swin = {
    'WT': {
        'Dice':    (0.8538, 0.1213, 0.3415, 0.9663),
        'Jaccard': (0.7353, 0.1445, 0.2076, 0.9343),
    },
    'TC': {
        'Dice':    (0.6817, 0.2626, 0.0004, 0.9440),
        'Jaccard': (0.5673, 0.2607, 0.0002, 0.8940),
    },
    'ET': {
        'Dice':    (0.7544, 0.1890, 0.1356, 0.9561),
        'Jaccard': (0.6354, 0.2029, 0.0727, 0.9158),
    },
}

# nnU-Net performance summary
nnunet = {
    'WT': {
        'Dice':    (0.8117, 0.1213, 0.3415, 0.9662),
        'Jaccard': (0.7060, 0.1445, 0.2076, 0.9343),
    },
    'TC': {
        'Dice':    (0.6945, 0.2625, 0.0004, 0.9440),
        'Jaccard': (0.5897, 0.2607, 0.0002, 0.8940),
    },
    'ET': {
        'Dice':    (0.7903, 0.1890, 0.1356, 0.9561),
        'Jaccard': (0.7035, 0.2029, 0.0727, 0.9158),
    },
}

# Perform Wilcoxon test
def run_wilcoxon(region, metric):
    swin_scores = generate_scores(*swin[region][metric])
    nnunet_scores = generate_scores(*nnunet[region][metric])
    stat, p = wilcoxon(swin_scores, nnunet_scores)
    print(f"{region} {metric}: statistic = {stat:.2f}, p-value = {p:.4f} {'*' if p < 0.05 else ''}")

print("Wilcoxon Signed-Rank Test Results\n--- Dice ---")
for region in ['WT', 'TC', 'ET']:
    run_wilcoxon(region, 'Dice')

print("\n--- Jaccard ---")
for region in ['WT', 'TC', 'ET']:
    run_wilcoxon(region, 'Jaccard')
