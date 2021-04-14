#!/bin/bash -e
#SBATCH --job-name=chemo10000
#SBATCH --time=44:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=350
#SBATCH --array=3-6
#SBATCH --mail-type=END
#SBATCH -o chemo_sim.%N.%j.out        # STDOUT
#SBATCH -e chemo_sim.%N.%j.err     # STDERR
#SBATCH --mail-user=ga824@nyu.edu


#for inference on 2 observations
# Job ID: 2941458
# Cluster: greene
# User/Group: ga824/ga824
# State: COMPLETED (exit code 0)
# Cores: 1
# CPU Utilized: 00:02:34
# CPU Efficiency: 95.65% of 00:02:41 core-walltime
# Job Wall-clock time: 00:02:41
# Memory Utilized: 231.97 MB
# Memory Efficiency: 30.93% of 750.00 MB

python generate_presimulated_data.py -n $SLURM_ARRAY_TASK_ID



