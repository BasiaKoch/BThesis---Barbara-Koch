#!/bin/bash
#SBATCH -p GPU  # partition (queue)
#SBATCH -N 1  # number of nodes
#SBATCH -t 0-36:00  # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR
#SBATCH --gres=gpu:1  # request 1 GPU

# Load conda 
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

conda activate nnunetv2


nnUNetv2_train Dataset001_BraTS2020 3d_fullres 0
nnUNetv2_train Dataset001_BraTS2020 3d_fullres 1
nnUNetv2_train Dataset001_BraTS2020 3d_fullres 2
nnUNetv2_train Dataset001_BraTS2020 3d_fullres 3
nnUNetv2_train Dataset001_BraTS2020 3d_fullres 4

