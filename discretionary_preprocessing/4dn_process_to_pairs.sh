#!/bin/bash

#SBATCH --job-name=4dn_process_to_pairs
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of cores requested
#SBATCH --ntasks-per-node=30

# Partition
#SBATCH -p RM-shared

# Request time
#SBATCH -t 48:00:00

# output to a designated folder
#SBATCH -o /ocean/projects/bio210019p/stevesho/raw_tissue_hic/slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

fastq_prefix=$1
seqtk=/ocean/projects/bio210019p/stevesho/resources/software/seqtk/seqtk
script_dir=/ocean/projects/bio210019p/stevesho/conda/docker-4dn-hic/scripts
resource_dir=/ocean/projects/bio210019p/stevesho/resources/4dn_resources
working_dir=/ocean/projects/bio210019p/stevesho/raw_tissue_hic
hg38_fasta=4DNFI823L888.fasta.gz
hg38_index=4DNFIZQZ39L9.bwaIndex.tgz
chrom_sizes=4DNFI823LSII.chrom.sizes

sra_dir=${working_dir}/sra_data
renamed_dir=${working_dir}/renamed_fastqs
outdir=${working_dir}/${fastq_prefix}
mkdir -p ${renamed_dir}
mkdir -p ${outdir}


# We run the processing out of 4DN's HI-C processing docker container. To build
# the environment, run:
# `singularity pull --disable-cache docker://4dndcic/4dn-hic:v44`

# Our data only involves sequencing replicates, not biological replicates. The
# pipeline is run to pair generation and stopped. Merging, marking, deduping,
# and matrix generation happen in the next script.


# =============================================================================
# Preprocess the fastqs to have proper naming convention for bwa mem
# =============================================================================
function preprocess_fastq {
    local sra_dir=$1  # directory where the fastq files are located
    local renamed_dir=$2  # directory where the renamed fastq files will be stored
    local fastq_prefix=$3  # prefix of the fastq files
    # decompress
    gunzip -c ${sra_dir}/${fastq_prefix}_1.fastq.gz > ${renamed_dir}/${fastq_prefix}_1.fastq
    gunzip -c ${sra_dir}/${fastq_prefix}_2.fastq.gz > ${renamed_dir}/${fastq_prefix}_2.fastq

    sed "s/\(${fastq_prefix}\.[0-9]*\)\.\([12]\)/\1\/\2/"  ${renamed_dir}/${fastq_prefix}_1.fastq > ${renamed_dir}/${fastq_prefix}_1.renamed.fastq
    sed "s/\(${fastq_prefix}\.[0-9]*\)\.\([12]\)/\1\/\2/" ${renamed_dir}/${fastq_prefix}_2.fastq > ${renamed_dir}/${fastq_prefix}_2.renamed.fastq

    rm ${renamed_dir}/${fastq_prefix}_1.fastq
    rm ${renamed_dir}/${fastq_prefix}_2.fastq
}

preprocess_fastq \
    ${sra_dir} \
    ${renamed_dir} \
    ${fastq_prefix}


# =============================================================================
# Step 1: BWA MEM alignment
# Args:
#   1 - fastq1
#   2 - fastq2
#   3 - index_file
#   4 - outdir
#   5 - prefix
#   6 - num_threads
# Output:
#   BAM file of the name `${outdir}/${fastq_prefix}.bam`
# =============================================================================
singularity exec --bind ${script_dir},${renamed_dir}/,${resource_dir},${outdir} \
   /ocean/projects/bio210019p/stevesho/conda/4dn-hic_v44.sif \
   bash ${script_dir}/run-bwa-mem.sh \
   ${renamed_dir}/${fastq_prefix}_1.renamed.fastq \
   ${renamed_dir}/${fastq_prefix}_2.renamed.fastq \
   ${resource_dir}/${hg38_index} \
   ${outdir} \
   ${fastq_prefix} \
   24

# delete fastq files after bam creation
rm ${renamed_dir}/${fastq_prefix}_1.renamed.fastq
rm ${renamed_dir}/${fastq_prefix}_2.renamed.fastq

# delete copied fastqs
rm ${outdir}/fastq1
rm ${outdir}/fastq2

# =============================================================================
# Step 2: Generate pairs
# BAM file from the previous step is used here
# Args:
#   1 - bamfile
#   2 - chr_sizes
#   3 - outdir
#   4 - out_prefix
#   5 - threads
#   6 - compression program
# Output:
#   Pairs file of the name `${outdir}/${fastq_prefix}.sam.pairs.gz`
# =============================================================================
singularity run --bind ${script_dir},${resource_dir},${outdir} \
   /ocean/projects/bio210019p/stevesho/conda/4dn-hic_v44.sif \
   bash ${script_dir}/run-pairsam-parse-sort.sh \
   ${outdir}/${fastq_prefix}.bam \
   ${resource_dir}/${chrom_sizes} \
   ${outdir} \
   ${fastq_prefix} \
   24 \
   lz4c


# =============================================================================
# Step 6: Clean up intermediate files
# =============================================================================
if [ -f "${outdir}/${fastq_prefix}.sam.pairs.gz" ]; then
    echo "Pairs file successfully generated. Cleaning up!"
    for file in ${fastq_prefix}.bam GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.amb GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.ann GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.bwt GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.pac GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.sa; do
        if [ -f "${outdir}/${file}" ]; then
            rm ${outdir}/${file}
        fi
    done
else
    echo "Pairs file not generated, no clean up performed"
fi


# sra_accessions=(
#     "SRR1501138"
#     "SRR1501139"
#     "SRR1501140"
#     "SRR1501141"
#     "SRR1501142"
#     "SRR1501143"
#     "SRR1501144"
#     "SRR1501145"
#     "SRR1501146"
#     "SRR1501147"
#     "SRR1501148"
#     "SRR1501149"
#     "SRR1501150"
#     "SRR1501151"
#     "SRR4271980"
#     "SRR4271981"
#     "SRR4271982"
#     "SRR4271983"
#     "SRR4271997"
#     "SRR4271998"
#     "SRR4271999"
#     "SRR4272000"
#     "SRR4272004"
#     "SRR4272005"
#     "SRR4272006"
#     "SRR4272007"
#     "SRR4272008"
#     "SRR4272009"
#     "SRR4272010"
#     "SRR4272015"
#     "SRR4272016"
#     "SRR4272018"
#     "SRR4272019"
#     "SRR4272020"
#     "SRR4272031"
#     "SRR4272032"
#     "SRR4272033"
#     "SRR4272034"
#     "SRR4272035"
#     "SRR4272036"
#     "SRR4272037"
# )
# # because the file writes to temp.gz and temp1.gz in the directory that the script is run from and is difficult to change due to being in a container, we copy the script to each subdirectory and run it from there instead.
# processing_dir=/ocean/projects/bio210019p/stevesho/raw_tissue_hic
# for accession in ${sra_accessions[@]}; do
#     mkdir -p ${processing_dir}/${accession}
#     cp 4dn_process_to_pairs.sh ${processing_dir}/${accession}/4dn_process_to_pairs.sh
#     cd ${processing_dir}/${accession}
#     sbatch 4dn_process_to_pairs.sh ${accession}
#     cd ${processing_dir}
# done
