#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --gres=gpu:1

# Setup conda
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

# Activate the [swin] environment
conda activate NMT

# Ensure pip is installed in case it's missing
conda install pip

# Install albumentations if it's not installed
pip install albumentations

# Go to the training directory
cd /home/u873859/3D_Brain_Tumor_Seg_V2/

# Convert the Jupyter notebook to a Python script
jupyter nbconvert --to script Train_Notebook.ipynb --output Train_Notebook

# Run the converted script
python Train_Notebook.py
