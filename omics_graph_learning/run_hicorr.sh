#!/bin/bash

#SBATCH --job-name=hicorr_pre
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=bio210019p

# Number of cores requested
#SBATCH --ntasks-per-node=16

# Partition
#SBATCH -p RM-shared

# Time
#SBATCH -t 48:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

module load anaconda3
module load bowtie2/2.4.4
module load samtools/1.13.0

conda activate /ocean/projects/bio210019p/stevesho/ogl

name=$1
workingdir=$2
maplen=$3
hg19=/ocean/projects/bio210019p/stevesho/resources/h38_bowtie
hg19fai=/ocean/projects/bio210019p/stevesho/resources/h38_bowtie
lib=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/HiCorr/bin/preprocess
bed=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/DPNII_HiCorr_ref/hg38.DPNII.frag.bed


# name='4DNFIXIDXXGV'
# workingdir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/process_deeploop/hmec
# maplen=96

### run: ##########################################################################################

# # 5. categorize reads pair and map them to Airma fragment pairs
# samtools view ${workingdir}/${name}.bam \
#     | cut -f2-8 \
#     | $lib/bam_to_temp_HiC.pl \
#     > ${workingdir}/${name}.temp

# # convert cis pairs to fragment pairs
# $lib/reads_2_cis_frag_loop.pl \
#     $bed \
#     $maplen \
#     ${workingdir}/${name}.loop.inward \
#     ${workingdir}/${name}.loop.outward \
#     ${workingdir}/${name}.loop.samestrand \
#     ${workingdir}/${name} \
#     ${workingdir}/${name}.temp 

# # get trans fragment pairs
# $lib/reads_2_trans_frag_loop.pl \
#     $bed \
#     $maplen \
#     ${workingdir}/${name}.loop.trans \
#     ${workingdir}/${name}.temp 
# wait

# # sort fragment pairs
# for file in ${workingdir}/${name}.loop.inward ${workingdir}/${name}.loop.outward ${workingdir}/${name}.loop.samestrand; do
#         cat $file | $lib/summary_sorted_frag_loop.pl $bed  > ${file}_sorted
# done
# cat ${workingdir}/${name}.loop.trans | $lib/summary_sorted_trans_frag_loop.pl - > ${workingdir}/${name}.loop.trans_sorted
# wait

# # resort by fragment id
# for file in ${name}.loop.inward ${name}.loop.outward ${name}.loop.samestrand ${name}.loop.trans; do
#         python $lib/resort.py $bed ${file}_sorted $lib ${workingdir}
# done
# wait

# # 6. filter the fragment pairs:
# awk '{if($4>1000)print $0}' ${workingdir}/${name}.loop.inward_sorted.combined \
#     > ${workingdir}/${name}.loop.inward.filter
# awk '{if($4>5000)print $0}' ${workingdir}/${name}.loop.outward_sorted.combined \
#     > ${workingdir}/${name}.loop.outward.filter
# wait

# # 7. merge fragment pairs (Note if you have multiple biological reps, run the first 6 steps for each rep, and merge in step 7)
# $lib/merge_sorted_frag_loop.pl \
#     ${workingdir}/${name}.loop.samestrand_sorted.combined \
#     ${workingdir}/${name}.loop.inward.filter \
#     ${workingdir}/${name}.loop.outward.filter \
#     > ${workingdir}/frag_loop.${name}.cis
# $lib/merge_sorted_frag_loop.pl \
#     ${workingdir}/${name}.loop.trans_sorted \
#     > ${workingdir}/frag_loop.${name}.trans
# wait

# echo ${workingdir}/${name} "trans:" `cat ${workingdir}/frag_loop.${name}.trans | awk '{sum+=$3}END{print sum/2}'` "cis:" `cat ${workingdir}/frag_loop.${name}.cis | awk '{sum+=$3}END{print sum/2}'` "cis2M:" `cat ${workingdir}/frag_loop.${name}.cis | awk '{if($4<=2000000)print}' | awk '{sum+=$3}END{print sum/2}'`

# # 8. clean UP, frag_loop.$expt.cis and frag_loop.${workingdir}/${name}.trans are the input files for HiCorr 
# # rm -f temp.${workingdir}/${name}.loop.inward.filter temp.${workingdir}/${name}.loop.outward.filter temp.${workingdir}/${name}.loop.inward temp.${workingdir}/${name}.loop.outward

bash /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/HiCorr/HiCorr_DPNII.sh \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/DPNII_HiCorr_ref \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/HiCorr/bin/DPNII \
    ${workingdir}/frag_loop.${name}.cis \
    ${workingdir}/frag_loop.${name}.trans \
    ${workingdir}/${name}_hicorr_output \
    hg38