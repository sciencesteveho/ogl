#!/bin/bash
#
# Code for symlinking and moving chromatin loops from different sources.
# Eventually will integrate the actual processing, but for now the processing is
# done in different scripts and this module is simply for combining them.

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
log_progress () {
    echo -e "[$(date +%Y-%m-%dT%H:%M:%S%z)] "
}

# =============================================================================
# Combine chromatin loops from two different files
# Assumes files are in standard .bedpe format
# =============================================================================
function _combine_chr_loops () {
    local input_file1=$1  # first loop file
    local input_file2=$2  # second loop file
    local output_dir=$3  # directory to store combined loops
    local prefix=$4  # name of tissue for combined loop file
    local suffix=$5  # file prefix or num loops
    
    cat \
        "$input_file1" \
        "$input_file2" \
        | sort -k1,1 -k2,2n \
        > "$output_dir/${prefix}_${suffix}.hg38.combined_loops"
}


# =============================================================================
# Make symlinks for peakachu derived chr loops
# =============================================================================
function _combine_chr_loops () {
    local input_file1=$1  # name of loop first
    local src_dir=$2  # directory to store combined loops
    local dst_dir=$3  # name of tissue for combined loop file

    ln -s \
        "${src_dir}/${input_file1}" \
        "${dst_dir}/${input_file1}"
}


# =============================================================================
# Main function to perform centralized processing
# =============================================================================
function _main () {
    # parse input arguments
    local root_dir=$1
    local deepanchor_dir=$2
    local deeploop_dir=$3
    local fdr_dir=$4
    local coarsegrain_dir=$5

    # set up directories
    local shared_data_dir=${root_dir}/shared_data
    local processed_loops_dir=${shared_data_dir}/processed_loops
    local reference_dir=${shared_data_dir}/references

    # for folder in deepanchor deeploop peakachu combined_loop_callers fdr_filtered_hic adaptive_coarsegrain hic_combined loops_and_fdrhic loops_and_coarsegrain all_loops_all_hic; do
    for folder in deepanchor deeploop peakachu combined_loop_callers fdr_filtered_hic adaptive_coarsegrain; do
        mkdir -p "${processed_loops_dir}/${folder}"
    done

    for folder in 100000 200000 300000 gte1 gte2; do
        mkdir -p "${processed_loops_dir}/deeploop/${folder}"
    done

    # set up tissue array
    # declare -a tissues=(
    #     "adrenal" "aorta" "gm12878" "h1_esc" "hepg2" "hippocampus" "hmec" "imr90" "k562" "left_ventricle" "liver" "lung" "mammary" "nhek" "ovary" "pancreas" "skeletal_muscle" "skin" "small_intestine" "spleen"
    # )
    declare -a tissues=(
        "adrenal" "aorta" "gm12878" "h1_esc" "hepg2" "hippocampus" "hmec" "imr90" "k562" "left_ventricle" "liver" "lung" "nhek" "ovary" "pancreas" "skeletal_muscle" "small_intestine" "spleen"
    )

    local -A peakachu_files=(
        ["adrenal"]="Schmitt_2016.Adrenal.hg38.peakachu-merged.loops"
        ["aorta"]="Leung_2015.Aorta.hg38.peakachu-merged.loops"
        ["gm12878"]="Rao_2014.GM12878.hg38.peakachu-merged.loops"
        ["h1_esc"]="Dixon_2015.H1-ESC.hg38.peakachu-merged.loops"
        ["hepg2"]="ENCODE3.HepG2.hg38.peakachu-merged.loops"
        ["hippocampus"]="Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops"
        ["hmec"]="Rao_2014.HMEC.hg38.peakachu-merged.loops"
        ["imr90"]="Rao_2014.IMR90.hg38.peakachu-merged.loops"
        ["k562"]="Rao_2014.K562.hg38.peakachu-merged.loops"
        ["left_ventricle"]="Leung_2015.VentricleLeft.hg38.peakachu-merged.loops"
        ["liver"]="Leung_2015.Liver.hg38.peakachu-merged.loops"
        ["lung"]="Schmitt_2016.Lung.hg38.peakachu-merged.loops"
        ["mammary"]="Rao_2014.HMEC.hg38.peakachu-merged.loops"
        ["nhek"]="Rao_2014.NHEK.hg38.peakachu-merged.loops"
        ["ovary"]="Schmitt_2016.Ovary.hg38.peakachu-merged.loops"
        ["pancreas"]="Schmitt_2016.Pancreas.hg38.peakachu-merged.loops"
        ["skeletal_muscle"]="Schmitt_2016.Psoas.hg38.peakachu-merged.loops"
        ["skin"]="Rao_2014.NHEK.hg38.peakachu-merged.loops"
        ["small_intestine"]="Schmitt_2016.Bowel_Small.hg38.peakachu-merged.loops"
        ["spleen"]="Schmitt_2016.Spleen.hg38.peakachu-merged.loops"
    )

    # symlink peakachu loops
    for tissue in "${!peakachu_files[@]}"; do
        ln -s \
            "${reference_dir}/peakachu_loops/${peakachu_files[$tissue]}" \
            "${processed_loops_dir}/peakachu/${tissue}_loops.bedpe"
    done

    # symlink deepanchor loops
    for tissue in "${tissues[@]}"; do
        ln -s \
            "${deepanchor_dir}/${tissue}_deepanchor.bedpe.hg38" \
            "${processed_loops_dir}/deepanchor/${tissue}_loops.bedpe"
    done

    # symlink deeploop loops
    for folder in 100000 200000 300000 gte1 gte2; do
        for tissue in "${tissues[@]}"; do
            ln -s \
                "${deeploop_dir}/${folder}/${tissue}_${folder}.pixels" \
                "${processed_loops_dir}/deeploop/${folder}/${tissue}_loops.bedpe"
        done
    done

    # symlink fdr filtered hic loops
    for tissue in "${tissues[@]}"; do
        for fdr in 0.1 0.01 0.001; do
            mkdir -p "${processed_loops_dir}/fdr_filtered_hic/${fdr}"
            ln -s \
                "${fdr_dir}/${tissue}/${tissue}_${fdr}_filtered.txt" \
                "${processed_loops_dir}/fdr_filtered_hic/${fdr}/${tissue}_contacts.bedpe"
        done
    done

    # adaptive coarsegrain 
    for tissue in "${tissues[@]}"; do
        for k in 100000 300000 500000; do
            mkdir -p "${processed_loops_dir}/adaptive_coarsegrain/${k}"
            ln -s \
                "${coarsegrain_dir}/${tissue}_contacts_${k}.bed" \
                "${processed_loops_dir}/adaptive_coarsegrain/${k}/${tissue}_contacts.bedpe"
        done
    done

    # # combine loop calls
    # for key in "${!peakachu_file[@]}"; do
    #     cat "${deepanchor_peakachu_dir}/${file}" \
    #         "${deeploop_only_dir}/${combine_files[$file]}" \
    #         | sort -k1,1 -k2,2n \
    #         > "${final_dir}/${tissue}_alloops.bed"
    # done
}


# =============================================================================
# run main_func function! 
# =============================================================================
_main \
    /ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing \
    /ocean/projects/bio210019p/stevesho/raw_tissue_hic/deepanchor \
    /ocean/projects/bio210019p/stevesho/raw_tissue_hic/pixel_processing/top_n_pixels \
    /ocean/projects/bio210019p/stevesho/raw_tissue_hic/contact_matrices/fdr_filtered \
    /ocean/projects/bio210019p/stevesho/raw_tissue_hic/contact_matrices/coolers/topk

echo "Total time: $(convertsecs SECONDS)"