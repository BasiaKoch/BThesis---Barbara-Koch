import os
import nibabel as nib
import numpy as np

# Directories containing the labels
directories = [
    "/home/u873859/NNunet/nnUNet_raw/Dataset001_BraTS2020/labelsTr",
    "/home/u873859/NNunet/nnUNet_raw/Dataset001_BraTS2020/labelsTs"
]

# Function to remap labels from 4 to 3
def remap_labels(file_path):
    # Load the segmentation file (nii format)
    img = nib.load(file_path)
    data = img.get_fdata()

    # Print unique labels before remapping
    print(f"Checking {file_path}...")
    print(f"Original labels in {file_path}: {np.unique(data)}")

    # If label 4 exists, remap it to label 3
    if 4 in np.unique(data):
        data[data == 4] = 3
        print(f"Labels after remapping in {file_path}: {np.unique(data)}")

        # Save the modified image back in .nii format (not compressed)
        nib.save(nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header), file_path)
        print(f"✅ Modified: {file_path}")
    else:
        print(f"⚠️ No label 4 found in {file_path}")

# Loop over all directories and process each .nii file
for directory in directories:
    for fname in os.listdir(directory):
        if fname.endswith(".nii"):  # Only process .nii files
            file_path = os.path.join(directory, fname)
            remap_labels(file_path)

print("Label remapping completed!")
