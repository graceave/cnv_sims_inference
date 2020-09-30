#!/bin/bash
#SBATCH --job-name=TpyABC
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --array=0
#SBATCH --mem=500
#SBATCH --mail-type=END
#SBATCH -o pyABC_test.%N.%j.out        # STDOUT
#SBATCH -e pyABC_test.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu

module load anaconda3/2019.10 
source activate del-env

# s_snv=0.002  # Venkataram et al. 2016
# m_snv=1.67e-10 #SNV mutation rate: 1.67 x 10^-10 per base per generation (Zhu et al. 2014)

python infer_pyABC.py -m "WF" -cs 0.001 -cu 1e-7 -ss 0.002 -su 1.67e-10 -s $SLURM_ARRAY_TASK_ID -o "WF_pyABC_test$SLURM_ARRAY_TASK_ID"

conda deactivate


