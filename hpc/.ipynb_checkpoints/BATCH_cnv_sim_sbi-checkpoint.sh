#!/bin/bash -e
#SBATCH --job-name=sbiC100
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3
#SBATCH --mem=2200
#SBATCH --mail-type=END
#SBATCH -o sbi_Chemo100000.%N.%j.out        # STDOUT
#SBATCH -e sbi_Chemo100000.%N.%j.err     # STDERR
##SBATCH --mail-user=ga824@nyu.edu

MODEL="Chemo"
N_PRESIM=100000
echo "SNPE"
echo ${MODEL}
echo ${N_PRESIM}
echo $SLURM_ARRAY_TASK_ID

python infer_sbi.py -m ${MODEL} -pd "${MODEL}_presimulated_data_${N_PRESIM}_${SLURM_ARRAY_TASK_ID}.csv" -pt "${MODEL}_presimulated_theta_${N_PRESIM}_${SLURM_ARRAY_TASK_ID}.csv" -obs "${MODEL}_simulated_data.csv" -o "${MODEL}_SNPE_${N_PRESIM}_${SLURM_ARRAY_TASK_ID}" -d "/scratch/ga824/cnv_sim_inf/" -s 1 -g "generations.csv"

rm *${MODEL}_SNPE_${N_PRESIM}_${SLURM_ARRAY_TASK_ID}.pdf

