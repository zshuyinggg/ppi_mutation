#!/bin/bash 

#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --job-name=esm_infer
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zshuying@tamu.edu
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --error=esm_infer.err --output=esm_infer.out
#SBATCH --nodes=3
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/user/zshuying/.conda/envs/yuning/lib/
module load Anaconda3 cuDNN/8.4.1.50-CUDA-11.7.0 GCC/10.3.0
source /sw/eb/sw/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate yuning
python --version
cd $SCRATCH/ppi_mutation/
which python

srun --export=ALL /scratch/user/zshuying/.conda/envs/yuning/bin/python scripts/esm/ESM_infer.py

