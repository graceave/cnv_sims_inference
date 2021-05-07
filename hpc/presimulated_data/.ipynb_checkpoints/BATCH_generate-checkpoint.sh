#!/bin/bash -e
#SBATCH --job-name=chemo1000
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=400
#SBATCH --array=1-3
#SBATCH --mail-type=END
#SBATCH -o chemo_sim1000.%N.%j.out        # STDOUT
#SBATCH -e chemo_sim1000.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu


python generate_presimulated_data.py -n $SLURM_ARRAY_TASK_ID



