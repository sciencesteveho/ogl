#!/bin/bash
#
# This is a description for your code to do stuff. It's a good idea to write
# something for the sake of reproducibility, ok. Don't give me that look, you're
# going to do it. Good habits don't build themselves (yet)!

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



function _main () {
    local -A peakachu_files=(
        ["hippocampus"]="Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops"
        ["left_ventricle"]="Leung_2015.VentricleLeft.hg38.peakachu-merged.loops"
        ["liver"]="Leung_2015.Liver.hg38.peakachu-merged.loops"
        ["lung"]="Schmitt_2016.Lung.hg38.peakachu-merged.loops"
        ["mammary"]="Rao_2014.HMEC.hg38.peakachu-merged.loops"
        ["pancreas"]="Schmitt_2016.Pancreas.hg38.peakachu-merged.loops"
        ["skeletal_muscle"]="Schmitt_2016.Psoas.hg38.peakachu-merged.loops"
        ["skin"]="Rao_2014.NHEK.hg38.peakachu-merged.loops"
        ["small_intestine"]="Schmitt_2016.Bowel_Small.hg38.peakachu-merged.loops"
        ["aorta"]="Leung_2015.Aorta.hg38.peakachu-merged.loops"
    )

    local -A combine_files=(
        ["aorta_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_Aorta.top300K_300000_loops.bedpe.hg38"
        ["hippocampus_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_Hippocampus.top300K_300000_loops.bedpe.hg38"
        ["left_ventricle_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_LeftVentricle.top300K_300000_loops.bedpe.hg38"
        ["liver_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_Liver.top300K_300000_loops.bedpe.hg38"
        ["lung_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_Lung.top300K_300000_loops.bedpe.hg38"
        ["pancreas_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_Pancreas.top300K_300000_loops.bedpe.hg38"
        ["skeletal_muscle_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_Psoas_Muscle.top300K_300000_loops.bedpe.hg38"
        ["small_intestine_peakachu_deepanchor.hg38.combined_loops"]="GSE167200_Small_Intenstine.top300K_300000_loops.bedpe.hg38"
    )
    
    for file in "${!combine_files[@]}"; do
        tissue=$(echo "${file}" | sed 's/_peakachu_deepanchor.hg38.combined_loops//g')
        cat "${deepanchor_peakachu_dir}/${file}" "${deeploop_only_dir}/${combine_files[$file]}" \
            | sort -k1,1 -k2,2n \
            > "${final_dir}/${tissue}_alloops.bed"
    done

}


# =============================================================================
# run main_func function! 
# =============================================================================
main_func \
     \  # description for arg1
     \  # description for arg2


# Find unique cutoff values
cutoffs=$(ls ovary_chr*_balanced_coarse_grain_*.bedpe | rev | cut -d_ -f1 | rev | sort -u)

# Iterate over each cutoff value
for cutoff in $cutoffs; do
    # The output file for this cutoff
    output_file="all_chrs_balanced_coarse_grain_${cutoff}.bedpe"

    # Find all files with this cutoff value and concatenate them
    cat ovary_chr*"_balanced_coarse_grain_${cutoff}" > "$output_file"
done


# mkdirs 
# processed_loops / deepanchor
# processed_loops / peakachu
# processed_loops / deeploop
# processed_loops / hi-c
# processed_loops / combined_catalogue
# 
# /ocean/projects/bio210019p/stevesho/raw_tissue_hic/deepanchor
# /ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/shared_data/references/peakachu_loops
# /ocean/projects/bio210019p/stevesho/raw_tissue_hic/pixel_processing/top_n_pixels
# /ocean/projects/bio210019p/stevesho/raw_tissue_hic/contact_matrices/fdr_filtered
# Adaptive coarsegrain - /ocean/projects/bio210019p/stevesho/raw_tissue_hic/contact_matrices/coolers


echo "Total time: $(convertsecs SECONDS)"