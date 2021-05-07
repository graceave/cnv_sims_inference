#!/bin/bash                     
#SBATCH --job-name=bc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11
#SBATCH --mem=2GB


## divide all fastq into R1 and R2
R1_1=($(ls /scratch/cgsb/gencore/out/Gresham/2020-12-15_HNYY3DRXX/2/*n01*.fastq.gz)) 

##this is going to assign the variables to file names
F_ID_R1_1=${R1_1[$SLURM_ARRAY_TASK_ID]}

NAME=${F_ID_R1_1:75:-9} ##sample name, eg. bc02_g124

#gunzip -c $F_ID_R1_1 > "${NAME}_R1_1.fastq" 

#R1_2=($(ls /scratch/cgsb/gencore/out/Gresham/2021-04-27_H5CYGDRXY/merged/*n01*.fastq.gz))
#F_ID_R1_2=${R1_2[$SLURM_ARRAY_TASK_ID]}

#gunzip -c $F_ID_R1_2 > "${NAME}_R1_2.fastq" 

#merge fastqs
#cat ${NAME}_R1_1.fastq ${NAME}_R1_2.fastq > ${NAME}_R1.fastq

## load bartender package
module purge
module load bartender/intel/1.1-20210106

##forward strand "BC2" extraction with quality cutoff of 30
bartender_extractor_com -f ${NAME}_R1.fastq -o mini_F_${NAME} -p TACC[5]AA[5]AA[5]TT[5]ATAA -u 0,8 -d f -q ?

###clustering (combinining off-by-1 barcodes with exact match barcodes)
## use this line of code for cases in which we did not expect low coverage 
bartender_single_com -f mini_F_${NAME}_barcode.txt -o F_${NAME} -z -1  ##-z-1 denotes off-by-1




