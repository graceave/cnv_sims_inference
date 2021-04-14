#!/bin/bash
#SBATCH --job-name=100WFLpyABC
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --array=1-3
#SBATCH --mem=9000
#SBATCH --mail-type=END
#SBATCH -o pyABC_Lauer_100000WF.%N.%j.out        # STDOUT
#SBATCH -e pyABC_Lauer_100000WF.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu

echo "WF_particle_size_100000"
echo $SLURM_ARRAY_TASK_ID

python infer_pyABC_Lauer.py -m "WF" -obs "PopPropForABC.csv" -p 100000 -s $SLURM_ARRAY_TASK_ID -o "WF_pyABC_100000_Lauer_$SLURM_ARRAY_TASK_ID" -d "/scratch/ga824/cnv_sim_inf/Lauer/" -c 5

rm *WF_pyABC_100000_Lauer_${SLURM_ARRAY_TASK_ID}.pdf