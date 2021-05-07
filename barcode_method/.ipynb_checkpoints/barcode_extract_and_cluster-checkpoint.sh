#!/bin/bash                     
#SBATCH --job-name=extract_and_cluster
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11
#SBATCH --mem=8GB


## divide all fastq into R1 and R2
R1=($(ls /scratch/cgsb/gencore/out/Gresham/2020-12-15_HNYY3DRXX/2/*n01*.fastq.gz)) 
R2=($(ls /scratch/cgsb/gencore/out/Gresham/2020-12-15_HNYY3DRXX/2/*n02*.fastq.gz)) 

##this is going to assign the variables to file names
F_ID_R1=${R1[$SLURM_ARRAY_TASK_ID]}
F_ID_R2=${R2[$SLURM_ARRAY_TASK_ID]}

NAME=${F_ID_R1:75:-9} ##sample name, eg. bc02_g124

## make directories
mkdir ${NAME} 
cp $F_ID_R1 $NAME/ ##copy each fastq to its newly made {NAME} directory
gunzip -c $NAME/*.gz > $NAME/"${NAME}_R1.fastq" ##unzip all the gz files inside the NAME directory, output the result into a new file named "{NAME}_R1.fastq" stored within the NAME dir 

## load bartender package
module purge
module load bartender/intel/1.1-20210106

##forward strand "BC2" extraction with quality cutoff of 30
bartender_extractor_com -f $NAME/${NAME}_R1.fastq -o $NAME/miniBar_F_$NAME -p TACC[5]AA[5]AA[5]TT[5]ATAA -u 0,8 -d f -q ?

###clustering (combinining off-by-1 barcodes with exact match barcodes)
## use this line of code for cases in which we did not expect low coverage 
bartender_single_com -f $NAME/miniBar_F_${NAME}_barcode.txt -o F_$NAME -z -1  ##-z-1 denotes off-by-1

