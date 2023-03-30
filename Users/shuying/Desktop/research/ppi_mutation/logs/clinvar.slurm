#!/bin/bash 

#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --job-name=clinvar
#SBATCH --ntasks=28
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zshuying@tamu.edu
#SBATCH --mem=30G
#SBATCH --time=4:00:00
#SBATCH --output=clinvar.out
#SBATCH --nodes=1

module restore shuying

source /sw/eb/sw/Anaconda3/2021.05/etc/profile.d/conda.sh
conda activate shuying
python --version
cd $SCRATCH/ppi_mutation/scripts

srun --export=ALL python /scratch/user/zshuying/ppi_mutation/scripts/clinvar_variant_sum.py


