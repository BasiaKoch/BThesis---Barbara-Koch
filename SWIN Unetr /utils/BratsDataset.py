import pandas as pd
import numpy as np
import nibabel as nib
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose
from monai.transforms import Compose, RandFlipd, RandShiftIntensityd
from monai.transforms import Resized



class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "train", do_resizing: bool = True):
        self.df = df
        self.phase = phase
        self.augmentations = self.get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.do_resizing = do_resizing


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = '/home/u873859/BraTS2020_Dataset'

        # Build the full path using the CSV 'path' column to get the relative path to the subdirectories
        sub_dir = self.df.loc[idx, 'path']

        # Remove the 'BraTS2020/' prefix from the path
        sub_dir = sub_dir.replace('./BraTS2020/', '')

        images = []
        for modality in self.data_types:
            # Construct the full image path by joining root_path, sub_dir, and modality
            img_path = os.path.join(root_path, sub_dir, id_ + modality)
            img = self.load_img(img_path)
            if self.do_resizing:
                img = self.resize(img, is_mask=False)
            img = self.normalize(img)
            images.append(img)

        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))  # (C, D, H, W)

        # Load the mask
        mask_path = os.path.join(root_path, sub_dir, id_ + '_seg.nii')
        mask = self.load_img(mask_path)
        if self.do_resizing:
            mask = self.resize(mask, is_mask=True)
        mask = self.preprocess_mask_labels(mask)
        mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)

        augmented = self.augmentations({"image": img.astype(np.float32), "mask": mask.astype(np.float32)})
        img = augmented['image']
        mask = augmented['mask']

        return {"Id": id_, "image": img, "mask": mask}


    def get_augmentations(self, phase):
        return Compose([])


    def load_img(self, file_path):
        return np.asarray(nib.load(file_path).dataobj)
    
    def normalize(self, data: np.ndarray):
        data_min, data_max = np.min(data), np.max(data)
        return (data - data_min) / (data_max - data_min + 1e-5)

    def resize(self, data: np.ndarray, is_mask: bool = False):
        """
        Resize 3D image or mask to a shape compatible with Swin UNETR (divisible by 16).
        Uses MONAI's Resized transform.
        """
        # Add channel dimension: (1, D, H, W)
        data = np.expand_dims(data, axis=0)

        # Set desired shape (D, H, W) – must be divisible by 16
        target_shape = (128, 224, 224)
        transformer = Resized(keys=["img"], spatial_size=target_shape, mode="nearest" if is_mask else "trilinear")

        # Apply resizing
        data = transformer({"img": data})["img"]

        # Remove channel dimension
        data = np.squeeze(data, axis=0)
        return data


    def crop_3d_array(self, arr, crop_shape):
        """
        Crop a 3D array to the specified shape (D, H, W).
        Assumes input is in (D, H, W) format.
        """
        assert len(crop_shape) == 3, "crop_shape must be a 3-element tuple"
        assert all([c <= a for c, a in zip(crop_shape, arr.shape)]), \
            f"Cannot crop to shape {crop_shape} from {arr.shape}"

        slices = []
        for dim_size, target_size in zip(arr.shape, crop_shape):
            start = (dim_size - target_size) // 2
            end = start + target_size
            slices.append(slice(start, end))

        cropped = arr[slices[0], slices[1], slices[2]]
        assert cropped.shape == crop_shape, f"Expected crop {crop_shape}, got {cropped.shape}"
        return cropped


    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = np.where(np.isin(mask, [1, 2, 4]), 1, 0)
        mask_TC = np.where(np.isin(mask, [1, 4]), 1, 0)
        mask_ET = np.where(mask == 4, 1, 0)
        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        return mask

    def crop_3d_array(self, arr, crop_shape):
        d, h, w = arr.shape
        cd, ch, cw = crop_shape
        start_d = max((d - cd) // 2, 0)
        start_h = max((h - ch) // 2, 0)
        start_w = max((w - cw) // 2, 0)
        end_d = start_d + cd
        end_h = start_h + ch
        end_w = start_w + cw
        cropped = arr[start_d:end_d, start_h:end_h, start_w:end_w]
        assert cropped.shape == crop_shape, f"Expected crop {crop_shape}, got {cropped.shape}"
        return cropped

