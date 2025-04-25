#!/bin/bash
#SBATCH -p GPU                    # GPU partition
#SBATCH -N 1                      # Number of nodes
#SBATCH -t 0-36:00                # Max walltime (36 hours)
#SBATCH -o slurm.%N.%j.out       # STDOUT
#SBATCH -e slurm.%N.%j.err       # STDERR
#SBATCH --gres=gpu:1             # Request 1 GPU

# Load and activate conda
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

conda activate nnunetv2

# Set nnUNet environment variables
export nnUNet_raw="/home/u873859/nnUNet/NNunet/nnUNet_raw"
export nnUNet_preprocessed="/home/u873859/nnUNet/NNunet/nnUNet_preprocessed"
export nnUNet_results="/home/u873859/nnUNet/NNunet/nnUNet_results"

# Run full prediction pipeline using nnUNetTrainer
nnUNetv2_predict \
  -d 001 \
  -i $nnUNet_raw/imagesTs \
  -o $nnUNet_results/Dataset001_BraTS2020/nnUNetTrainer__fold0_val \
  -f 0 \
  -c 3d_fullres \
  -tr nnUNetTrainer \
  -p nnUNetPlans \
  --disable_tta

echo "✅ Predictions with nnUNetTrainer (fold 0) saved to: nnUNetTrainer__fold0_val"
