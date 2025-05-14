 # =====================================================================================
# This script was adapted from the original notebook:
# https://github.com/Inc0mple/3D_Brain_Tumor_Seg_V2/blob/master/VizEval_Single_Notebook.ipynb
#
# Original author: https://github.com/Inc0mple
# License: Apache License 2.0 (if applicable, otherwise follow original repo license)
#
# Modifications have been made for use in the BraTS2020 evaluation workflow
# and to suit the objectives of this thesis project.
# =====================================================================================

from tqdm import tqdm
import os
import time
from datetime import datetime
from random import randint
from PIL import Image
from skimage.transform import resize


import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import re
import warnings
from time import sleep

warnings.simplefilter("ignore")
from utils.Meter import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes

from utils.BratsDataset import BratsDataset

from utils.Meter import BCEDiceLoss

from utils.viz_eval_utils import get_dataloaders, compute_scores_per_classes, count_parameters
from models.SwinUNETR import SwinUNETR
modelDict = {
    "SwinUNETR": SwinUNETR(in_channels=4, out_channels=3, img_size=(128, 224, 224), depths=(1, 1, 1, 1), num_heads=(2, 4, 8, 16)).to('cuda'),
}
def chooseModel():
    availableActions = {str(i+1): k for (i, k)
                        in zip(range(len(modelDict)), modelDict.keys())}
    nl = '\n'
    # Takes in a dictionary with key/value pair corresponding with control/action
    availableActionsList = [(key, val)
                            for key, val in availableActions.items()]
    print(f"Use number keys to choose one of the models below: \n")
    print(
        f"Available Models: {nl.join(f'[{tup[0]}: {tup[1]}]' for tup in availableActionsList)}")
    sleep(1)
    while True:
        userInput = input("Enter your action: ")
        if userInput not in availableActions:
            print(
                f"{userInput} is an invalid action. Please try again.")
        else:
            break
    return availableActions[userInput]

def evalSingle(target="BraTS20_Training_004", threshold=0.5):
    model_name = "SwinUNETR_DoubleLayerDepth"
    print(f"'{model_name}' selected for evaluation and visualisation")

    model = modelDict[model_name]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Path to pretrained model
    checkpoint_path = "/home/u873859/BraTS2020_Dataset/Logs/SwinUNETR/your_best_model_20250414-004601.pth"

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        print(f"{model_name} loaded from: {checkpoint_path} — Parameters: {count_parameters(model)}")
    except Exception as e:
        print(f"❌ Error loading model from: {checkpoint_path}")
        print(e)
        return

    # Path to the CSV file used for splitting
    csv_path = "/home/u873859/3D_Brain_Tumor_Seg_V2/fold_data.csv"

    name, imgs, targets = None, None, None

    # Load test dataloader
    _, _, test_dataloader = get_dataloaders(
        dataset=BratsDataset,
        path_to_csv=csv_path,
        val_fold=0,
        test_fold=1,
        batch_size=1,
        do_resizing=True
    )

    print("Evaluating on test set...")
    start = datetime.now()
    dice_scores_per_classes, iou_scores_per_classes, ids = compute_scores_per_classes(
        model, test_dataloader, ['WT', 'TC', 'ET']
    )

    # Build metrics dataframe
    ids_df = pd.DataFrame(ids, columns=['Ids'])
    
    dice_df = pd.DataFrame({
        'WT dice': dice_scores_per_classes['WT'],
        'TC dice': dice_scores_per_classes['TC'],
        'ET dice': dice_scores_per_classes['ET'],
    })

    iou_df = pd.DataFrame({
        'WT jaccard': iou_scores_per_classes['WT'],
        'TC jaccard': iou_scores_per_classes['TC'],
        'ET jaccard': iou_scores_per_classes['ET'],
    })

    
    
    val_metrics_df = pd.concat([ids_df, dice_df, iou_df], axis=1)
    val_metrics_df = val_metrics_df.loc[:, ['Ids',
                                            'WT dice', 'WT jaccard',
                                            'TC dice', 'TC jaccard',
                                            'ET dice', 'ET jaccard']]
    mean_metrics_df = val_metrics_df.drop(columns=["Ids"]).mean()
    val_metrics_df = val_metrics_df.sort_values(by="Ids").reset_index(drop=True)

    # Plot average Dice and Jaccard
    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=mean_metrics_df.index, y=mean_metrics_df, palette=palette, ax=ax)
    ax.set_title(f"{model_name} - Dice and Jaccard Coefficients (Test Set)", fontsize=20)
    for idx, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(100 * mean_metrics_df.values[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")
    
    plt.tight_layout()
    barplot_path = f"/home/u873859/BraTS2020_Dataset/Logs/SwinUNETR/metrics_barplot.png"
    fig.savefig(barplot_path)
    print(f"✅ Saved barplot to: {barplot_path}")

    
    print(f"Evaluation done! Inference time: {datetime.now() - start}")
    print("Now visualising predictions for target:", target)

    # Find and load sample from test set
    for i, data in enumerate(test_dataloader):
        if data['Id'][0] == target:
            name, imgs, targets = data['Id'][0], data['image'], data['mask']
            break

    if imgs is None:
        print(f"❌ Target sample '{target}' not found in test set.")
        return

    with torch.no_grad():
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)

        predictions = (probs >= threshold).float().cpu()
        targets = targets.cpu()
        img = imgs.cpu().squeeze()[0].numpy()
        gt = targets[0].squeeze().numpy()
        pred = predictions[0].squeeze().numpy()


    for i in range(40, 101, 5):
        # Resize each slice to 200x200
        img_resized = resize(img[i], (200, 200), preserve_range=True, anti_aliasing=True)
        wt_gt_resized = resize(gt[0][i], (200, 200), preserve_range=True, order=0)
        wt_pred_resized = resize(pred[0][i], (200, 200), preserve_range=True, order=0)
        tc_gt_resized = resize(gt[1][i], (200, 200), preserve_range=True, order=0)
        tc_pred_resized = resize(pred[1][i], (200, 200), preserve_range=True, order=0)
        et_gt_resized = resize(gt[2][i], (200, 200), preserve_range=True, order=0)
        et_pred_resized = resize(pred[2][i], (200, 200), preserve_range=True, order=0)

        # Plotting
        fig, axs = plt.subplots(1, 7, figsize=(22, 5))  # Wider layout for resized slices

        axs[0].imshow(img_resized, cmap="gray", aspect='equal')
        axs[0].set_title('img')

        axs[1].imshow(wt_gt_resized, cmap="viridis", aspect='equal')
        axs[1].set_title('WT_GT')

        axs[2].imshow(wt_pred_resized, cmap="viridis", aspect='equal')
        axs[2].set_title('WT_Pred')
    
        axs[3].imshow(tc_gt_resized, cmap="cividis", aspect='equal')
        axs[3].set_title('TC_GT')

        axs[4].imshow(tc_pred_resized, cmap="cividis", aspect='equal')
        axs[4].set_title('TC_Pred')

        axs[5].imshow(et_gt_resized, cmap="plasma", aspect='equal')
        axs[5].set_title('ET_GT')

        axs[6].imshow(et_pred_resized, cmap="plasma", aspect='equal')
        axs[6].set_title('ET_Pred')

        # Adjust and save
        fig.suptitle(f"{model_name} | Sample: {name} | Slice: {i}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        slice_path = f"/home/u873859/BraTS2020_Dataset/Logs/SwinUNETR/visual_slice_{i:03d}.png"
        fig.savefig(slice_path)
        print(f"✅ Saved slice {i} to: {slice_path}")
        plt.close(fig)

   
    

    return val_metrics_df


# Run the evaluation
val_metrics_df = evalSingle()

val_metrics_df = evalSingle()

# Save results to CSV
results_path = "/home/u873859/BraTS2020_Dataset/Logs/SwinUNETR/test_results.csv"
if val_metrics_df is not None:
    val_metrics_df.to_csv(results_path, index=False)
    print(f"✅ Results saved to: {results_path}")
else:
    print("⚠️ No results to save. evalSingle() returned None.")
