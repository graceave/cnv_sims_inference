#!/bin/bash
#SBATCH --job-name=snvWSWM
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --array=0-9
#SBATCH --mem=8GB
#SBATCH --mail-type=END
#SBATCH -o wswmSNV.%N.%j.out        # STDOUT
#SBATCH -e wswmSNV.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu

module load anaconda3/2019.10 
source activate del-env

#this is WSWM for SNVs, all combos for CNVs

#WSWM cnv
python cnv_sim_delfi.py -cs 0.001 -cu 1e-7 -ss 0.001 -su 1e-7 -s $SLURM_ARRAY_TASK_ID -o "snvWSWM_cnvWSWM$SLURM_ARRAY_TASK_ID"
#SSWM
python cnv_sim_delfi.py -cs 0.1 -cu 1e-7 -ss 0.001 -su 1e-7 -s $SLURM_ARRAY_TASK_ID -o "snvWSWM_cnvSSWM$SLURM_ARRAY_TASK_ID"
#WSSM
python cnv_sim_delfi.py -cs 0.001 -cu 1e-5 -ss 0.001 -su 1e-7 -s $SLURM_ARRAY_TASK_ID -o "snvWSWM_cnvWSSM$SLURM_ARRAY_TASK_ID"
#SSSM
python cnv_sim_delfi.py -cs 0.1 -cu 1e-5 -ss 0.001 -su 1e-7 -s $SLURM_ARRAY_TASK_ID -o "snvWSWM_cnvSSSM$SLURM_ARRAY_TASK_ID"

conda deactivate


