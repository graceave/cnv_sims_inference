#!/bin/bash                     
#SBATCH --job-name=bc-comb
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

## load bartender package
module purge
module load bartender/intel/1.1-20210106

## combine barcodes for all timepoints
## remove clusters with size <10
bartender_combiner_com -f F_bc02_g5_pcr_cluster.csv,F_bc02_g5_pcr_quality.csv,F_bc02_g37_pcr_cluster.csv,F_bc02_g37_pcr_quality.csv,F_bc02_g62_pcr_cluster.csv,F_bc02_g62_pcr_quality.csv,F_bc02_g99_pcr_cluster.csv,F_bc02_g99_pcr_quality.csv,F_bc02_g124_pcr_cluster.csv,F_bc02_g124_pcr_quality.csv,F_bc02_g149_pcr_cluster.csv,F_bc02_g149_pcr_quality.csv -o bc02 -c 10 


bartender_combiner_com -f F_bc04_g5_pcr_cluster.csv,F_bc04_g5_pcr_quality.csv,F_bc04_g37_pcr_cluster.csv,F_bc04_g37_pcr_quality.csv,F_bc04_g62_pcr_cluster.csv,F_bc04_g62_pcr_quality.csv,F_bc04_g99_pcr_cluster.csv,F_bc04_g99_pcr_quality.csv,F_bc04_g124_pcr_cluster.csv,F_bc04_g124_pcr_quality.csv,F_bc04_g149_pcr_cluster.csv,F_bc04_g149_pcr_quality.csv -o bc04 -c 10



