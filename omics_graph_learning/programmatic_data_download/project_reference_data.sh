#!/bin/bash
#
# Code to download and parse reference base data for the project.

# =============================================================================
# Setting up variables to track time and progress
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
# Prepare directory folders and subfolders as necessary
# Arguments:
#   $1: project root directory
# =============================================================================
function _prepare_directory_structure () {
    local root_dir=$1  # project root directory

    local directories=(
        "raw_tissue_data/chromatin_loops/processed_loops"
        "shared_data/interaction"
        "shared_data/local"
        "shared_data/reference"
        "shared_data/regulatory_elements"
        "shared_data/targets/expression"
        "shared_data/targets/tpm"
        "shared_data/targets/matrices"
    )

    for directory in "${directories[@]}"; do
        mkdir -p "$root_dir/$directory"
    done
}


# =============================================================================
# Convert gencode v26 GTF to bed, remove micro RNA genes and only keep canonical
# "gene" entries. Additionally, make a lookup table to convert from gencode to
# genesymbol.
# wget https://storage.googleapis.com/gtex_analysis_v8/reference/gencode.v26.GRCh38.genes.gtf 
# Arguments:
# =============================================================================
function _gencode_bed () {
    gtf2bed <  $1/$2 | awk '$8 == "gene"' | grep -v miR > $3/local/gencode_v26_genes_only_with_GTEx_targets.bed
    cut -f4,10 $3/local/gencode_v26_genes_only_with_GTEx_targets.bed | sed 's/;/\t/g' | cut -f1,5 | sed -e 's/ gene_name //g' -e 's/\"//g' > $3/interaction/gencode_to_genesymbol_lookup_table.txt
}


# =============================================================================
# run main_func function! 
# =============================================================================
function main () {
    log_progress "Setting up function params"
    local root_dir=$1  # project root directory

    log_progress "Preparing directory structure"
    _prepare_directory_structure "$root_dir"


}

main_func \
     \  # root_directory
    
    

echo "Total time: $(convertsecs SECONDS)"