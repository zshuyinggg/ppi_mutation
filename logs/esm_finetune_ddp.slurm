#!/bin/bash 

#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --job-name=esm_finetune_ddp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zshuying@tamu.edu
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --error=esm_finetune_ddp_crop512_reloadevery2.err --output=esm_finetune_ddp_crop512_reloadevery2.out
#SBATCH --nodes=5
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2

module load WebProxy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/user/zshuying/.conda/envs/yuning/lib/
module load Anaconda3 cuDNN/8.4.1.50-CUDA-11.7.0 GCC/10.3.0
source /sw/eb/sw/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate yuning
python --version
cd $SCRATCH/ppi_mutation/
which python

srun --export=ALL /scratch/user/zshuying/.conda/envs/yuning/bin/python scripts/myesm/ESM_finetune_ddp.py --esm esm2_t36_3B_UR50D --numdevices 2 --numnodes 5 --unfreeze 10

