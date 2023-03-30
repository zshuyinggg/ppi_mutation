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
#SBATCH --output=single_protein.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/user/zshuying/.conda/envs/ProLMEnv/lib/
module load Anaconda3 cuDNN/8.4.1.50-CUDA-11.6.0 GCC/10.3.0	Python/3.9.15
source /sw/eb/sw/Anaconda3/2021.11/etc/profile.d/conda.sh
conda activate ProLMEnv
python --version
cd $SCRATCH/ppi_mutation/ProteinEncoder-LM-main


for i in {1..74}; do
    srun --export=ALL python scripts/main.py \
        run_eval \
        transformer \
        embed_seq \
        '/scratch/user/zshuying/ppi_mutation/trained_models/rp15_pretrain_1_models'\
        --batch_size 4\
        --data_dir '/scratch/user/zshuying/ppi_mutation/data/single_protein_seq/' \
        --metrics save_embedding \
        --split "batch150_$i" \
        --embed_modelNm 'rp15_pretrain_1_models' &
done
wait




