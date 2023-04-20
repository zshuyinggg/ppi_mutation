#!/bin/bash 
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --job-name=single_protein
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zshuying@tamu.edu
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=baseline0.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/user/zshuying/.conda/envs/shuying/lib/
module load Anaconda3 cuDNN/8.4.1.50-CUDA-11.7.0 GCC/10.3.0	Python/3.10.8
source /sw/eb/sw/Anaconda3/2021.11/etc/profile.d/conda.sh
conda activate shuying
python --version
cd $SCRATCH/ppi_mutation/



srun --export=ALL python scripts/esm/ESM_baseline0.py
