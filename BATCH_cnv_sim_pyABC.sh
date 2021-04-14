#!/bin/bash
#SBATCH --job-name=.1CpyABC
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --array=1
#SBATCH --mem=4000
#SBATCH --mail-type=END
#SBATCH -o pyABC_100Chemo.%N.%j.out        # STDOUT
#SBATCH -e pyABC_100Chemo.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu

echo "Chemo_particle_size_1000"
echo $SLURM_ARRAY_TASK_ID

python infer_pyABC.py -m "Chemo" -obs "Chemo_simulated_data.csv" -p 100 -s $SLURM_ARRAY_TASK_ID -o "Chemo_pyABC_100_$SLURM_ARRAY_TASK_ID" -d "/scratch/ga824/cnv_sim_inf/" -c 30

rm *Chemo_pyABC_100_${SLURM_ARRAY_TASK_ID}.pdf