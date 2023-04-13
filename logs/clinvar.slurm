#!/bin/bash 

#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --job-name=clinvar
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zshuying@tamu.edu
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH --output=clinvar.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
module restore shuying

source /sw/eb/sw/Anaconda3/2021.05/etc/profile.d/conda.sh
module load WebProxy
conda activate cpu
python --version
cd $SCRATCH/ppi_mutation/scripts

srun --export=ALL python /scratch/user/zshuying/ppi_mutation/scripts/esm/ESM_baseline0.py


