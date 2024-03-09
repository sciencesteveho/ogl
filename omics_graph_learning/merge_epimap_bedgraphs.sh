#!/bin/bash

#SBATCH --job-name=merge_epimap
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of nodes requested
#SBATCH --ntasks-per-node=18

# Partition
#SBATCH -p RM-shared

# Time
#SBATCH -t 24:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

module load anaconda3
conda activate /ocean/projects/bio210019p/stevesho/gnn

# This script is used to convert epimap bigwigs to narrow or broad peak bedfiles
# with liftover to hg38. Broad peaks are called for histone marks and narrow
# peaks for TFs. Histones and TFs are then merged across the three samples to
# act as "representative" potential tracks for the given tissue.
# SECONDS=0

# =============================================================================
# liftover hg19 to hg38
# Arguments:
#   $1 - directory to liftOver
#   $2 - bigwig tissue directory
#   $3 - filename without extension
# =============================================================================
function _merge_bedgraphs () {
    working_dir=$1
    
    function _liftover () {
        $1/liftOver \
        ${2}_merged.narrow.bed \
        $1/hg19ToHg38.over.chain.gz \
        ${2}_merged.narrow.hg38.bed \
        ${2}.unlifted

        # cleanup
        rm ${2}.unlifted
    }

    # set up cutoff array
    mkdir $1/merged_bedgraph
    cd $1/tmp
    rm *narrow.bed*

    declare -A cutoffs
    # cutoffs provided by C. Boix
	cutoffs["DNase-seq"]=1.9
	cutoffs["H3K27ac"]=2.2
	cutoffs["H3K27me3"]=1.2
	cutoffs["H3K36me3"]=1.7
	cutoffs["H3K4me1"]=1.7
	cutoffs["H3K4me2"]=2
	cutoffs["H3K4me3"]=1.7
	cutoffs["H3K79me2"]=2.2
	cutoffs["H3K9ac"]=1.6
	cutoffs["H3K9me3"]=1.1
    # these following cutoffs default to 2
	cutoffs["ATAC-seq"]=2
	cutoffs["CTCF"]=2
	cutoffs["POLR2A"]=2
	cutoffs["RAD21"]=2
	cutoffs["SMC3"]=2

    # for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    for feature in CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        # Collect files into an array
        files=( $(ls | grep "${feature}") )
        
        # Check that we have exactly three files
        if [ "${#files[@]}" -ne 3 ]; then
            echo "Error: Expected 3 files for feature ${feature}, but found ${#files[@]}."
            continue
        fi

        # Sort files once and store them
        for i in {0..2}; do
            sorted_files[$i]="${files[$i]}.sorted"
            LC_ALL=C sort --parallel=12 -S 80% -k1,1 -k2,2n "${files[$i]}" > "${sorted_files[$i]}"
        done
        
        # Sum the coverage values using bedtools unionbedg and awk
        echo "Merging ${feature} bedgraphs"
        bedtools unionbedg -i "${sorted_files[@]}" \
            | awk '{sum=$4+$5+$6; print $1, $2, $3, sum / 3}' > "${feature}_merged.bedGraph"
        echo "Merged ${feature} bedgraphs"

        echo "calling peaks with cutoff ${cutoffs[$feature]}"
        macs2 bdgpeakcall \
            -i ${feature}_merged.bedGraph \
            -o $1/merged_bedgraph/${feature}_merged.narrow.bed \
            -c ${cutoffs[$feature]} \
            -l 73 \
            -g 100

        cd $1/merged_bedgraph
        tail -n +2 ${feature}_merged.narrow.bed > tmpfile && mv tmpfile ${feature}_merged.narrow.bed
        echo "liftover ${feature} to hg38"
        _liftover \
            /ocean/projects/bio210019p/stevesho/resources \
            ${feature}
        cd $1/tmp
    done
}

_merge_bedgraphs $1