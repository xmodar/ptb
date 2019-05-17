#!/bin/bash
#SBATCH --job-name IBP
#SBATCH --array=0-79
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --mem 10GB

# donwnload miniconda3 in your slurm cluster
# install it in your home direcotry witout setting it up with the shell
# use it to create a new environment called ibp and install all requirements
# pip install ibp-neurips
# create a logs folder
# now submit this script with sbatch
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ibp

export CUDA_VISIBLE_DEVICES=0

ibp pgd -r -i $SLURM_ARRAY_TASK_ID