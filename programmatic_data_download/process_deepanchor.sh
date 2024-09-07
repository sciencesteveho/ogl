#!/bin/bash

#SBATCH --job-name=process_deepanchor
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of nodes requested
#SBATCH --ntasks-per-node=8

# Partition
#SBATCH -p RM-shared

# Time
#SBATCH -t 24:00:00

# output to a designated folder
#SBATCH -o %x_%j.out

#echo commands to stdout
set -x

module load anaconda3
conda activate /ocean/projects/bio210019p/stevesho/gnn

# =============================================================================
# Setting up variables to track time
# =============================================================================
SECONDS=0

function convertsecs() {
    local total_seconds=
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d\n" hours minutes seconds
}


# Function to echo script progress to stdout
log_progress() {
    echo -e "[$(date +%Y-%m-%dT%H:%M:%S%z)] "
}


# =============================================================================
# Function to convert .bigInteract files from DeepAnchor over to .bedpe files.
# Additionally liftovers the coordinates from hg19 to hg38
# =============================================================================
function _format_deepanchor_loops () {
    # Declare variables as local
    local resource_dir=$1
    local raw_chr_loop_dir=$2
    local genome_file=$3  # genome / chr size file
    local prefix=$4  # prefix for .bigInteract file
    local output_dir=$5  # final directory to place lifted and formatted calls
    local output_prefix=$6  # tissue naming for final file
    
    "$resource_dir/bigBedToBed" \
        "$raw_chr_loop_dir/${prefix}.bigInteract" \
        "$raw_chr_loop_dir/${prefix}.bedpe" 

    # Split bedpe and extend regions to approximate 5kb resolution, and to
    # determine a cutoff for anchor overlap to create edges to regulatory
    # elements. The first anchor is extended upstream, and the second anchor is
    # extended downstream.
    awk -v OFS='\t' '{print $9,$10,$11,NR}' "$raw_chr_loop_dir/${prefix}.bedpe" \
        | bedtools slop \
        -i stdin \
        -g "$genome_file" \
        -r 4981 \
        -l 0 \
        > "$raw_chr_loop_dir/${prefix}.bedpe_1"
    
    awk -v OFS='\t' '{print $14,$15,$16,NR}' "$raw_chr_loop_dir/${prefix}.bedpe" \
        | bedtools slop \
        -i stdin \
        -g "$genome_file" \
        -r 0 \
        -l 4981 \
        > "$raw_chr_loop_dir/${prefix}.bedpe_2"

    # liftover hg19 to hg38
    for file in bedpe_1 bedpe_2;
    do
        local outfile="${raw_chr_loop_dir}/${prefix}.${file}.hg38"
        local unmappedfile="${raw_chr_loop_dir}/${prefix}.${file}.unmapped"
        "$resource_dir/liftOver" \
            "$raw_chr_loop_dir/${prefix}.${file}" \
            "$resource_dir/hg19ToHg38.over.chain.gz" \
            "$outfile" \
            "$unmappedfile"
        
        sort -k4,4 -o "$outfile" "$outfile"
    done

    # join bedpe by matching anchor number
    join \
        -j 4 \
        -o 1.1,1.2,1.3,2.1,2.2,2.3 \
        "${raw_chr_loop_dir}/${prefix}.bedpe_1.hg38" \
        "${raw_chr_loop_dir}/${prefix}.bedpe_2.hg38" \
        | sed 's/ /\t/g' \
        | sort -k8,8n \
        > "${output_dir}/${output_prefix}_deepanchor.bedpe.hg38"
}


# =============================================================================
# Process deepanchor loops
# =============================================================================
function deepanchor_processing_main () {
    local resource_dir=$1
    local raw_chr_loop_dir=$2
    local final_dir=$3

    local -A loop_files=(
        ["ENCFF000RWO"]="k562"
        ["ENCFF000YCK"]="imr90"
        ["ENCFF001HHX"]="gm12878"
        ["GSM749715"]="hepg2"
        ["ENCFF000ONR"]="h1_esc"
        ["ENCFF001HPM"]="hmec"
        ["GSM749707"]="nhek"
        ["ENCFF000CFW"]="hippocampus"
        ["ENCFF000RYN"]="lung"
        ["ENCLB993ADU"]="pancreas"
        ["ENCLB645ZVC"]="skeletal_muscle"
        ["ENCLB020WHN"]="small_intestine"
        ["ENCFF002EXB"]="liver"
        ["ENCLB432SKR"]="aorta"
        ["ENCLB120LGG"]="skin"
        ["ENCLB303UFA"]="left_ventricle"
        ["ENCFF001HPM"]="mammary"
        ["ENCFF000SEC"]="spleen"
        ["ENCLB026GMS"]="ovary"
        ["ENCLB733WTO"]="adrenal"
    )

    download_prefix=https://github.com/Xuhang01/LoopAnchor/raw/main/loopanchor/data/loops/

    for file in "${!loop_files[@]}"; do
        log_progress "Downloading ${file} loops"
        wget -P "${raw_chr_loop_dir}" "${download_prefix}/${file}.bigInteract"

        log_progress "Processing ${file} loops"
        _format_deepanchor_loops \
            ${resource_dir} \
            ${raw_chr_loop_dir} \
            ${resource_dir}/hg19.chrom.sizes \
            ${file} \
            ${final_dir} \
            ${loop_files[$file]}
        log_progress "Finished processing ${file} loops"
        # echo "Total time: $(convertsecs SECONDS)"
    done
}

deepanchor_processing_main \
    /ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/shared_data/references \
    /ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/raw_tissue_data/chromatin_loops \
    /ocean/projects/bio210019p/stevesho/raw_tissue_hic/deepanchor