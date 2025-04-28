## download all necessary libraries 
from tqdm import tqdm
import os
import time
from datetime import datetime
from random import randint

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



import warnings
warnings.simplefilter("ignore")
from utils.Meter import Meter, DiceLoss, BCEDiceLoss, dice_coef_metric_per_classes, jaccard_coef_metric_per_classes

from utils.BratsDataset import BratsDataset

from utils.Meter import BCEDiceLoss


from utils.viz_eval_utils import get_dataloaders, compute_scores_per_classes

from models.SwinUNETR import SwinUNETR


model = SwinUNETR(in_channels=4, out_channels=3, img_size=(128, 224, 224), depths=(2, 2, 2, 2), num_heads=(2,4,8,16)).to('cuda')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(torch.cuda.is_available())
print(f"Parameter count: {count_parameters(model)}")
print(model)

class GlobalConfig:
    root_dir = '/home/u873859/BraTS2020_Dataset'
    train_root_dir = os.path.join(root_dir, 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData')
    test_root_dir = None  # Set this if you have a test dataset
    path_to_csv = os.path.join(root_dir, 'fold_data.csv')
    train_logs_path = os.path.join(root_dir, 'Logs', 'SwinUNETR')
    pretrained_model_path = os.path.join(train_logs_path, 'your_best_model_20250414-004601.pth')
    seed = 42



def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_img(file_path):
    data = nib.load(file_path)
    data = np.asarray(data.dataobj)
    return data

config = GlobalConfig()
seed_everything(config.seed)
if not os.path.isdir(config.train_logs_path):
    os.mkdir(config.train_logs_path)



##Preprocessing
survivalInfoPath = f"{config.train_root_dir}/survival_info.csv"
nameMappingPath = f"{config.train_root_dir}/name_mapping.csv"
survival_info_df = pd.read_csv(survivalInfoPath)
name_mapping_df = pd.read_csv(nameMappingPath)

name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 


df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")
df = df[["Brats20ID"]]
paths = []
for _, row  in df.iterrows():
    
    id_ = row['Brats20ID']
    path = os.path.join(config.train_root_dir, id_)
    paths.append(path)
    
df['path'] = paths

train_data = df

kf = KFold(n_splits=7, random_state=config.seed, shuffle=True)
for i, (train_index, val_index) in enumerate(kf.split(train_data)):
    # assign all rows at val_index to the ith fold
    train_data.loc[val_index, "fold"] = i


train_data.to_csv(config.path_to_csv, index=False)



###Trainer Class

class Trainer:
    """
    Factory for training proccess.
    Args:
        display_plot: if True - plot train history after each epoch.
        net: neural network for mask prediction.
        criterion: factory for calculating objective loss.
        optimizer: optimizer for weights updating.
        phases: list with train and validation phases.
        dataloaders: dict with data loaders for train and val phases.
        path_to_csv: path to csv file.
        meter: factory for storing and updating metrics.
        batch_size: data batch size for one step weights updating.
        num_epochs: num weights updation for all data.
        accumulation_steps: the number of steps after which the optimization step can be taken
                    (https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614).
        lr: learning rate for optimizer.
        scheduler: scheduler for control learning rate.
        losses: dict for storing lists with losses for each phase.
        jaccard_scores: dict for storing lists with jaccard scores for each phase.
        dice_scores: dict for storing lists with dice scores for each phase.
    """
    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 val_fold: int,
                 test_fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 display_plot: bool = True,
                 do_resizing: bool = True,
                 optimizer: torch.optim = Adam
                ):

        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",  # Reduces learning rate when a metric has stopped improving
                                           patience=2, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs
        train_dl, val_dl, test_dl = get_dataloaders(
            dataset=dataset,
            path_to_csv=path_to_csv,
            val_fold=val_fold,
            test_fold=test_fold,
            batch_size=batch_size,
            num_workers=4,
            do_resizing=do_resizing,
        )
        self.dataloaders = {
            "train": train_dl,
            "val": val_dl,
            "test": test_dl
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.last_completed_run_time = None
        self.parameter_count = count_parameters(self.net)
         
    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits
        
    def _do_epoch(self, epoch: int, phase: str):
        # print(f"{phase} epoch: {epoch + 1} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        t_dataloader = tqdm(enumerate(dataloader), unit="batch", total=total_batches)
        for itr, data_batch in t_dataloader:
            t_dataloader.set_description(f"{phase} epoch: {epoch + 1} | time: {time.strftime('%H:%M:%S')}")
            images, targets = data_batch['image'], data_batch['mask']
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            t_dataloader.set_postfix(loss=loss.item())
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(),
                         targets.detach().cpu()
                        )
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)

        return epoch_loss
        
    def run(self):
        start = datetime.now()
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()
                
            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                now = datetime.now().strftime("%Y%m%d-%H%M%S")
                checkpoint_filename = f"your_best_model_{now}.pth"
                torch.save(self.net.state_dict(), os.path.join(
                    config.train_logs_path, checkpoint_filename))
            print()
        self.last_completed_run_time = str(datetime.now() - start)
        self._save_train_history()
            
    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,
            
            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """, 
                  
            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """,
        ]
        
        clear_output(True)
    # Remove the seaborn-dark-palette style context
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        for i, ax in enumerate(axes):
            ax.plot(data[i]['val'], c=colors[0], label="val")
            ax.plot(data[i]['train'], c=colors[-1], label="train")
            ax.set_title(labels[i])
            ax.legend(loc="upper right")
    
        plt.tight_layout()
        plt.show()

            
    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")
        
    def _save_train_history(self):
        """writing model weights and training logs to files."""
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_filename = f"your_last_epoch_model_{now}.pth"
        torch.save(self.net.state_dict(),os.path.join(
            config.train_logs_path, checkpoint_filename))

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                         for key in logs_[i]]
        log_names = [key+log_names_[i] 
                     for i in list(range(len(logs_))) 
                     for key in logs_[i]
                    ]
        # train_logs_path = './BraTS2020Logs/train_log.csv'
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(os.path.join(config.train_logs_path, f"train_log_{now}.csv"), index=False)


###Initiate trainer class - here the model was trained with and wihout loading the pretrained weights

DO_RESIZING = True

trainer = Trainer(net=model,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=4,
                  batch_size=1,
                  val_fold=0,
                  test_fold=1,
                  num_epochs=50,
                  path_to_csv=config.path_to_csv,
                  do_resizing=DO_RESIZING,
                  optimizer=Adam)

#######################################################################
# """UNCOMMENT THE FOLLOWING 2 LINES IF RELOADING MODEL CHECKPOINT"""
if config.pretrained_model_path is not None:
    trainer.load_predtrain_model(config.pretrained_model_path)
#######################################################################
dataloader, _, test_dataloader = get_dataloaders(
    dataset=BratsDataset, path_to_csv=config.path_to_csv, val_fold=0, test_fold=1, batch_size=1, do_resizing=DO_RESIZING)
len(dataloader)

print(len(dataloader))
print(len(_))
print(len(test_dataloader))

data = next(iter(dataloader))
data['Id'], data['image'].shape, data['mask'].shape
# size = (batch_size, channels, depth, width, height)

img_tensor = data['image'].squeeze()[0].cpu().detach().numpy() 
mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))
print(img_tensor.shape)
image = np.rot90(montage(img_tensor))
mask = np.rot90(montage(mask_tensor)) 

fig, ax = plt.subplots(1, 1, figsize = (20, 20))
ax.imshow(image, cmap ='bone')
ax.imshow(np.ma.masked_where(mask == False, mask),
           cmap='cool', alpha=0.6)

trainer.run()


with open(os.path.join(config.train_logs_path, 'trainer_properties.txt'), 'w') as f:
  for param, value in trainer.__dict__.items():
      if not param.startswith("__"):
        f.write(f"{param}:{value}\n")
# %%time
model.eval()
dice_scores_per_classes, iou_scores_per_classes, ids = compute_scores_per_classes(
    model, test_dataloader, ['WT', 'TC', 'ET']
    )

ids_df = pd.DataFrame(ids)
ids_df.columns = ['Ids']

dice_df = pd.DataFrame(dice_scores_per_classes)
dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

iou_df = pd.DataFrame(iou_scores_per_classes)
iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
val_metics_df = pd.concat([ids_df, dice_df, iou_df], axis=1, sort=True)
val_metics_df = val_metics_df.loc[:, ['Ids', 'WT dice', 'WT jaccard',
                                      'TC dice', 'TC jaccard',
                                      'ET dice', 'ET jaccard']]
val_metics_df.sample(5)
val_metrics_df.select_dtypes(include='number').mean()


##save the results to logs 
val_metrics_df.to_csv(os.path.join(
    config.train_logs_path, "test_results.csv"), index=False)

mean_metrics.to_frame().to_csv(os.path.join(
    config.train_logs_path, "test_results_mean.csv"), index=True)



###Visualise test results 
val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT jaccard',
                                      'TC dice', 'TC jaccard',
                                      'ET dice', 'ET jaccard']]

colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
palette = sns.color_palette(colors, 6)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=mean_metrics.index, y=mean_metrics.values, palette=palette, ax=ax)
ax.set_xticklabels(mean_metrics.index, fontsize=14, rotation=15)
ax.set_title("Dice and Jaccard Coefficients from Test Set", fontsize=20)

for idx, p in enumerate(ax.patches):
    percentage = '{:.1f}%'.format(100 * mean_metrics.values[idx])
    x = p.get_x() + p.get_width() / 2 - 0.15
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")


fig.savefig(os.path.join(config.train_logs_path, "result.png"), format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
fig.savefig(os.path.join(config.train_logs_path, "result.svg"), format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')


