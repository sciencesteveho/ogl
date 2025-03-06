#!/bin/bash

#SBATCH --job-name=4dn_to_matrices
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of cores requested
#SBATCH --ntasks-per-node=36

# Partition
#SBATCH -p RM-shared

# Request time
#SBATCH -t 48:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

# An array to associate each tissue with its accessions. According to 4DN's
# processing pipeline, sequencing replicates are merged after initial pair
# generation but before duplicate removal. Reads resulting from a single PCR
# duplication even might exist in both replicates, so deduping is performed
# after merging (merging done via pairsam-merge, not merge-pairs).

# set up the tissue accessions array to associated each tissue with its list of
# sequencing replicates
declare -A tissue_accessions=(
   [hippocampus]="SRR4271997 SRR4271998 SRR4271999 SRR4272000"
   [lung]="SRR4272004 SRR4272005 SRR4272006"
   [pancreas]="SRR4272015 SRR4272016"
   [psoas]="SRR4272018 SRR4272019 SRR4272020"
   [small_intestine]="SRR4272031 SRR4272032 SRR4272033 SRR4272034"
   [liver]="SRR1501147 SRR1501148 SRR1501149 SRR1501150 SRR1501151"
   [aorta]="SRR1501138 SRR1501139 SRR1501140 SRR1501141"
   [left_ventricle]="SRR1501142 SRR1501143 SRR1501144 SRR1501145 SRR1501146"
   [spleen]="SRR4272035 SRR4272036 SRR4272037"
   [ovary]="SRR4272007 SRR4272008 SRR4272009 SRR4272010"
   [adrenal]="SRR4271980 SRR4271981 SRR4271982 SRR4271983"
)


# Define command-line arguments
tissue=$1  # (str) name of your tissue or cell line
merge_pairs=$2  # (bool) true or false to merge pairs
restriction_file=$3  # (str) name of the restriction file to use

# If no restriction file is provided, the code will default to useing HindIII.
if [ -z "$3" ]; then
    restriction_file="4DNFI823MBKE.txt"
else
    restriction_file="$3"
fi

# Define paths and other vars

script_dir=/ocean/projects/bio210019p/stevesho/conda/docker-4dn-hic/scripts
resource_dir=/ocean/projects/bio210019p/stevesho/resources/4dn_resources
working_dir=/ocean/projects/bio210019p/stevesho/raw_tissue_hic
chrom_sizes=4DNFI823LSII.chrom.sizes

sra_dir=${working_dir}/sra_data
outdir=${working_dir}/${tissue}
mkdir -p ${outdir}


# =============================================================================
# Simple function to use the accessions to get a list of pairsams for merging
# with absolute filepaths
# =============================================================================
function pair_getter () {
    local tissue=$1  # name of the tissue
    local accessions=()  # array of absolute filepaths to the pairsams to be returned
    for accession in ${tissue_accessions[$tissue]}; do
        accessions+=(${working_dir}/${accession}/${accession}.sam.pairs.gz)
    done
    echo ${accessions[@]}
}


# run pairsam-merge --> marking --> filter
# filtered pairsam --> cooler --> individual .cool
# filtered pairsam --> add fragment information --> juicer pre --> .hic
# =============================================================================
# Step 3: Merge pairs
# Args:
#   1 - output_prefix
#   2 - threads
#   3 - pairsam
# Output:
#   Pairs file of the name `${outdir}/${tissue}.merged.sam.pairs.gz`
# =============================================================================

# Check if merge_pairs is true or false. If true, will get accession list and
# run the first function. If not, continues.
if [ ${merge_pairs} == "true" ]; then
    echo "Merging pairs"
    accessions=$(pair_getter ${tissue})  # get accessions
    # run merging command
    singularity exec --bind ${script_dir},${outdir} \
        /ocean/projects/bio210019p/stevesho/conda/4dn-hic_v44.sif \
        bash ${script_dir}/run-pairsam-merge.sh \
        ${outdir}/${tissue} \
        24 \
        ${accessions[@]}
else
    echo "Skipping merging pairs"
fi


# =============================================================================
# Step 4: Mark duplicates
# Args:
#   1 - pairsam
#   2 - output_prefix
# Output:
#   Pairs file of the name `${outdir}/${tissue}.marked.sam.pairs.gz`
# =============================================================================
singularity exec --bind ${script_dir},${outdir} \
   /ocean/projects/bio210019p/stevesho/conda/4dn-hic_v44.sif \
   bash ${script_dir}/run-pairsam-markasdup.sh \
   ${outdir}/${tissue}.merged.sam.pairs.gz \
   ${outdir}/${tissue}


# =============================================================================
# Step 5: Filter pairs
# Args:
#   1 - pairsam
#   2 - output_prefix
#   3 - chr_sizes
# Output:
#   Pairs files of the following names: `${outdir}/${tissue}.unmapped.sam.pairs.gz`, `${outdir}/${tissue}.dedup.pairs.gz`, `${outdir}/${tissue}.lossless.bam`
# =============================================================================
singularity exec --bind ${script_dir},${resource_dir},${outdir} \
   /ocean/projects/bio210019p/stevesho/conda/4dn-hic_v44.sif \
   bash ${script_dir}/run-pairsam-filter.sh \
   ${outdir}/${tissue}.marked.sam.pairs.gz \
   ${outdir}/${tissue}.filtered \
   ${resource_dir}/${chrom_sizes}

# remove intermediate files
rm ${outdir}/${tissue}.filtered.lossless.bam
rm ${outdir}/${tissue}.filtered.unmapped.sam.pairs.gz
rm ${outdir}/${tissue}.marked.sam.pairs.gz
rm ${outdir}/${tissue}.merged.sam.pairs.gz
rm ${outdir}/temp.gz
rm ${outdir}/temp1.gz



# =============================================================================
# Step 8: Incorporate fragments into pairs file
# Args:
#   1 - deduped_pairs
#   2 - restriction_file
#   3 - output_prefix
# Output:
#   Pairs file with fragment information, called `${output_dir}/${tissue}.ff.pairs.gz` along with a .px2 index `${output_dir}/${tissue}.ff.pairs.gz.px2`
# =============================================================================
singularity exec --bind ${script_dir},${outdir},${resource_dir} \
    /ocean/projects/bio210019p/stevesho/conda/4dn-hic_v44.sif \
    bash ${script_dir}/run-addfrag2pairs.sh \
    -i ${outdir}/${tissue}.filtered.dedup.pairs.gz \
    -r ${resource_dir}/${restriction_file} \
    -o ${outdir}/${tissue}


# =============================================================================
# Step 10: Make a .hic file with juicer
#   1 - input pairs
#   2 - chromsize file
#   3 - out_prefix
#   4 - mapqfilter
#   5 - maxmem
#   6 - min_res
# Output:
#   Multi-res hi-c file with the name `${output_dir}/${tissue}.hic`
# =============================================================================
singularity exec --bind ${script_dir},${outdir},${resource_dir} \
    /ocean/projects/bio210019p/stevesho/conda/4dn-hic_v44.sif \
    bash ${script_dir}/run-juicebox-pre.sh \
    -i ${outdir}/${tissue}.ff.pairs.gz \
    -c ${resource_dir}/${chrom_sizes} \
    -o ${outdir}/${tissue} \
    -r 5000 \
    -m 48g \
    -q 0 \
    -u 5000,10000,20000,40000


# =============================================================================
# Step 9: Make cooler file. This is with cooler 0.9.3, so does not include the
# pixels duplication bug from cooler 0.8.3 currently present in the 4dn portal.
# Additionally adds the normalization vectors from juicer to the cooler file
# using hic2cool.
# Args:
#   1 - pairs file, bgzipped with a .px2 index
#   2 - chrsize_file
#   3 - binsize
#   4 - num_cores
#   5 - output_prefix
#   6 - max_split
# Output:
#   Cooler with the name `${output_dir}/${tissue}.cool`
# =============================================================================
module load anaconda3
conda activate /jet/home/stevesho/.conda/envs/hiclift
resolutions=(5000 10000 20000 40000)
for resolution in ${resolutions[@]}; do
    cooler cload pairix -p 24 -s 2 ${resource_dir}/${chrom_sizes}:${resolution} ${outdir}/${tissue}.ff.pairs.gz ${outdir}/${tissue}_${resolution}.cool
    cp ${outdir}/${tissue}_${resolution}.cool ${outdir}/${tissue}_${resolution}_normalized.cool
    hic2cool extract-norms -e ${outdir}/${tissue}.hic ${outdir}/${tissue}_${resolution}_normalized.cool
done


# ### For posterity, we ran the script with the following command:
# for tissue in hippocampus lung pancreas psoas small_intestine liver aorta spleen ovary adrenal; do
#     # cp 4dn_merge_pairs_output_matrices.sh ${tissue}/4dn_merge_pairs_output_matrices.sh
#     cd /ocean/projects/bio210019p/stevesho/raw_tissue_hic/${tissue}
#     sbatch 4dn_merge_pairs_output_matrices.sh ${tissue} true
# done
# # for tissue inleft_ventricle; do
