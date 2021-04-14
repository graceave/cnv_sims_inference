#!/bin/bash -e
#SBATCH --job-name=100LWFsbi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3
#SBATCH --mem=2000
#SBATCH --mail-type=END
#SBATCH -o sbi_LauerWF100000.%N.%j.out        # STDOUT
#SBATCH -e sbi_LauerWF100000.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu


echo "Lauer_data_100000_WF"
echo $SLURM_ARRAY_TASK_ID

python infer_sbi_Lauer.py -m "WF" -pd "WF_presimulated_data_100000_${SLURM_ARRAY_TASK_ID}.csv" -pt "WF_presimulated_theta_100000_${SLURM_ARRAY_TASK_ID}.csv" -obs "PopPropForABC.csv" -o "WF_SNPE_100000_${SLURM_ARRAY_TASK_ID}" -d "/scratch/ga824/cnv_sim_inf/Lauer/" -s 1

rm *WF_SNPE_100000_${SLURM_ARRAY_TASK_ID}.pdf

