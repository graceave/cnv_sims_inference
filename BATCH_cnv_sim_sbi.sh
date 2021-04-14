#!/bin/bash -e
#SBATCH --job-name=100Csbi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3
#SBATCH --mem=1000
#SBATCH --mail-type=END
#SBATCH -o sbi_Chemo100000.%N.%j.out        # STDOUT
#SBATCH -e sbi_Chemo100000.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu


echo "Chemo_presimulated_data_100000"
echo $SLURM_ARRAY_TASK_ID

python infer_sbi.py -m "Chemo" -pd "Chemo_presimulated_data_100000_${SLURM_ARRAY_TASK_ID}.csv" -pt "Chemo_presimulated_theta_100000_${SLURM_ARRAY_TASK_ID}.csv" -obs "Chemo_simulated_data.csv" -o "Chemo_SNPE_100000_${SLURM_ARRAY_TASK_ID}" -d "/scratch/ga824/cnv_sim_inf/" -s 1

rm *Chemo_SNPE_100000_${SLURM_ARRAY_TASK_ID}.pdf

