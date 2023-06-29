#!/bin/bash

#SBATCH --job-name=rebin_cooler
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio220004p
#SBATCH --mail-type=FAIL

# Number of cores requested
#SBATCH --ntasks-per-node=8

# Partition
#SBATCH -p RM-shared

# Request time
#SBATCH -t 5:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

module load anaconda3
conda activate refhic

#tissue=$1
tissue='liver'
resource_dir='/ocean/projects/bio210019p/stevesho/resources'

# try using hicConvertFormat to force integers
hicConvertFormat -m ${tissue}.cool --inputFormat cool --outputFormat cool --outFileName ${tissue}_int.cool --enforce_integer

# make 5kb binfile 
cooler makebins -o bins.5000kb.bed /ocean/projects/bio210019p/stevesho/resources/hg19.chrom.sizes.txt 5000

# get pixels from cooler
cooler dump -t pixels -o ${tissue}.pixels ${tissue}_int.cool

# rebin pixels @ 5kb
# cooler load --field count:dtype=float -f coo bins.5000kb.bed ${tissue}.pixels ${tissue}_5kb.cool
cooler load -f coo bins.5000kb.bed ${tissue}.pixels ${tissue}_5kb.cool

# convert to mcool w/ 5kb and 10kb resolutions
cooler zoomify -r 5000,10000 -o ${tissue}_5kb_10kb.mcool ${tissue}_5kb.cool

# try balancing
cooler balance ${tissue}_5kb_10kb.mcool::/resolutions/5000

# conver to bcool
refhic util cool2bcool -u 3000000 --resol 5000 ${tissue}_5kb_10kb.mcool ${tissue}_5kb.bcool
refhic util cool2bcool -u 3000000 --resol 5000 ${tissue}_5kb_10kb.mcool2 ${tissue}_nobal_5kb.bcool

# make predictions 
refhic loop pred --cpu True --chrom chr17 ${tissue}_5kb.bcool ${tissue}_chr17_loopCandidates.bedpe
refhic loop pred --cpu True --chrom chr17 ${tissue}_nobal_5kb.bcool ${tissue}_chr17_loopCandidates_nobal.bedpe



# # try converting to h5
# hicConvertFormat -m ${tissue}_5kb_10kb.mcool::/resolutions/5000 --inputFormat cool --outputFormat h5 --outFileName ${tissue}_5kb.h5


# liftover to hg38
conda activate hiclift
HiCLift \
    --input ${tissue}.cool \
    --input-format cooler \
    --out-pre ${tissue}_hg38 \
    --output-format cool \
    --chain-file ${resource_dir}/hg19ToHg38.over.chain.gz \
    --out-chromsizes ${resource_dir}/hg38.chrom.sizes.txt \
    --in-assembly hg19 \
    --out-assembly hg38 \
    --logFile hiclift.log