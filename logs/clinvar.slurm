#!/bin/bash 
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --job-name=clinvar
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zshuying@tamu.edu
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --output=clinvar.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24

module restore dl

source /sw/eb/sw/Anaconda3/2021.11/etc/profile.d/conda.sh
conda activate shuying
python --version
cd $SCRATCH/ppi_mutation/scripts

srun --export=ALL python scripts/clinvar_deal_big.py


