#!/bin/bash
#SBATCH --job-name=1CpyABC
#SBATCH --time=100:00:00
#SBATCH --nodes=2
##SBATCH --partition=cs
##SBATCH --cpus-per-task=48
##SBATCH --ntasks-per-node=48
#SBATCH --array=2
#SBATCH --mem=30000
#SBATCH --mail-type=END
#SBATCH -o pyABC_1000Chemo.%N.%j.out        # STDOUT
#SBATCH -e pyABC_1000Chemo.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu

MODEL="Chemo"
N_PARTICLES=1000
echo "pyABC"
echo ${MODEL}
echo ${N_PARTICLES}
echo $SLURM_ARRAY_TASK_ID

python infer_pyABC.py -m ${MODEL} -obs "${MODEL}_simulated_data.csv" -p ${N_PARTICLES} -s $SLURM_ARRAY_TASK_ID -o "${MODEL}_pyABC_${N_PARTICLES}_$SLURM_ARRAY_TASK_ID" -d "/scratch/ga824/cnv_sim_inf/" -c 96

rm *${MODEL}_pyABC_${N_PARTICLES}_${SLURM_ARRAY_TASK_ID}.pdf