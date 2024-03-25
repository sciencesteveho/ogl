#!/bin/bash
#
# Scripts to retrieve top pixel contacts from enhanced or denoised hi-c data
# from deeploop. Note that some of the data was available for download and some
# we processed ourselves, so the scripts are not universal and need to be
# adjusted according. Keeping that in mind, if you plan to use these scripts,
# pay attention to where directories are hardcoded, and be aware that not all of
# our directories were created programatically.
#
# All of the tissue data need to be lifted over to hg38.
# The cell lines: H1, IMR90, and GM12878 need to be lifted over to hg38.
# The cell lines: HepG2, HMEC, K562, and NHEK are already in hg38, but need start and end positions to be adjust by -1.


# =============================================================================
# Setting up variables to track time
# =============================================================================
SECONDS=0

function convertsecs() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d\n" $hours $minutes $seconds
}


# =============================================================================
# Download processed deeploop contacts for cell lines. We do our own processing
# for K562, HepG2, HMEC, and NHEK
# =============================================================================
function _download_loops () {
    wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FH1%2Eraw%5FHiCorr%5FLoopDenoise%2Etxt%2Egz  # H1
    wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FIMR90%2Eraw%5FHiCorr%5FLoopDenoise%2Etxt%2Egz  # IMR90
    wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FGM12878%2Ein%5Fsitu%2Eraw%5FHiCorr%5FLoopDenoise%2Etxt%2Egz  # GM12878
    gunzip *.gz*
    echo "Downloaded deeploop contacts!"
}


# =============================================================================
# Format deeploop pixels to bedpe for downloaded contacts
# =============================================================================
function _deeploop_txt_to_bedpe () {
    local input_dir=$1  # directory of deeploop files
    local input_prefix=$2  # filename of downloaded deeploop contacts
    local output_dir=$3  # directory to place processed file
    local output_prefix=$4  # filename without extension of processed file

    # Using the local input/output directories and prefixes
    sed \
        -e 's/:/\t/g' \
        -e 's/-/\t/g' \
        "$input_dir/$input_prefix".txt \
        | cut -f1,2,3,4,5,6,9 \
        > "$output_dir/$output_prefix".pixels
}


# =============================================================================
# Liftover deeploop files
# =============================================================================
function _liftover_deeploop_bedpe () {
    local input_file=$1  # directory of deeploop files
    local resource_dir=$2  # directory of liftover and liftover chain
    local output_dir=$3  # directory to place processed file
    local chain_file="${resource_dir}/hg19ToHg38.over.chain"

    local tissue_file=$(basename "${input_file}")
    local input_dir=$(dirname "${input_file}")

    mkdir -p "${output_dir}"

    # split bedpe into two files for lifting
    local lifted_1="${output_dir}/${tissue_file}_1.hg38"
    local lifted_2="${output_dir}/${tissue_file}_2.hg38"
    local unmapped_1="${output_dir}/${tissue_file}_1.unmapped"
    local unmapped_2="${output_dir}/${tissue_file}_2.unmapped"

    awk -v OFS='\t' '{print $1,$2,$3,$7,NR}' "${input_dir}/${tissue_file}" > "${output_dir}/${tissue_file}_1"
    awk -v OFS='\t' '{print $4,$5,$6,$7,NR}' "${input_dir}/${tissue_file}" > "${output_dir}/${tissue_file}_2"

    # liftover each anchor
    for file in _1 _2; do
        "${resource_dir}/liftOver" \
            "${output_dir}/${tissue_file}${file}" \
            "${chain_file}" \
            "${output_dir}/${tissue_file}${file}.hg38" \
            "${output_dir}/${tissue_file}${file}.unmapped"

        sort -k5,5 -o "${output_dir}/${tissue_file}${file}.hg38" "${output_dir}/${tissue_file}.${file}.hg38"
    done

    # join the lifted files
    join -j 5 \
        -o 1.1,1.2,1.3,2.1,2.2,2.3,1.4 \
        "${lifted_1}" \
        "${lifted_2}" \
        | sed 's/ /\t/g' \
        > "${input_dir}/${tissue_file}.hg38"

    # cleanup
    rm -f "${output_dir}/${tissue_file}"._* "${lifted_1}" "${lifted_2}" "${unmapped_1}" "${unmapped_2}"
}


# =============================================================================
# Extract top pixels from cooler files.
# =============================================================================
function _extract_top_pixels () {
    local tissue=$1  # tissue name
    local working_dir=$2  # one level above /path/to/data
    local extract_cooler=$3  # bool to extract cooler if necessary
    local zero_index=$4  # bool to adjust start and end positions by -1
    local liftover=$5  # bool to liftover to hg38

    local pixels_dir="${working_dir}/pixels"
    local coolers_dir="${working_dir}/coolers"
    local liftover_resources_dir="/ocean/projects/bio210019p/stevesho/resources"
    local tmp_dir="${working_dir}/tmp"

    mkdir -p "${pixels_dir}" "${coolers_dir}" "${tmp_dir}"

    local pixels_file="${pixels_dir}/${tissue}.pixels"
    local sorted_pixels_file="${pixels_file}_sorted"

    if [[ "$extract_cooler" == true ]]; then
        cooler dump \
            -t pixels \
            --join \
            -o "${pixels_file}" \
            "${coolers_dir}/${tissue}.cool"
    fi

    local intermediate_file="${pixels_file}"
    if [[ "$zero_index" == true ]]; then
        intermediate_file+=".zero_indexed"
        awk -v OFS='\t' '{print $1,$2-1,$3,$4,$5,$6,$7}' "${pixels_file}" > "${intermediate_file}"
    fi

    if [[ "$liftover" == true ]]; then
        _liftover_deeploop_bedpe \
            "${intermediate_file}" \
            "${liftover_resources_dir}" \
            "${tmp_dir}"
        local intermediate_file+=".hg38"
    fi

    # Sort by e/o ratio to extract top pixels
    sort --parallel=8 -S 80% -k7,7nr "${intermediate_file}" > "${sorted_pixels_file}"
    
    local thresholds=(50000 100000 150000 200000 300000 500000 1000000)
    for threshold in "${thresholds[@]}"; do
        local top_n_file="${pixels_dir}/${tissue}_${threshold}.pixels"
        head -n "${threshold}" "${sorted_pixels_file}" > "${top_n_file}"
    done

    local gte1_file="${pixels_dir}/${tissue}_gte1.pixels"
    awk '$7 >= 1' "${sorted_pixels_file}" > "${gte1_file}"
}


# =============================================================================
# Main functions!
# =============================================================================
# format deeploop pixels for the cell-lines with already processed loops
function _format_downloaded_deeploop () {
    base_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic"
    output_dir="${base_dir}/pixels"
    declare -a datasets=("h1" "imr90" "gm12878")
    for dataset in "${datasets[@]}"; do
        file_prefix="GSE167200_${dataset^^}"
        if [ "$dataset" == "gm12878" ]; then
            file_prefix+=".in_situ"
        fi
        file_suffix=".raw_HiCorr_LoopDenoise"

        _deeploop_txt_to_bedpe \
            "${base_dir}" \
            "${file_prefix}${file_suffix}" \
            "${output_dir}" \
            "${dataset}"
        echo "Processed ${dataset}!"
    done
}

# run main function
_extract_top_pixels \
    ${1} \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic \
    ${2} \
    ${3} \
    ${4}


echo "Finished in $(convertsecs "${SECONDS}")"

"""
For posterity, we ran this script with the following:
#sbatch top_pixels.sh tissue extract-cooler zero-index liftover
#bash /ocean/projects/bio210019p/stevesho/data/preprocess/omics_graph_learning/omics_graph_learning/discretionary_preprocessing/deeploop_top_pixels.sh aorta false false true

# sbatch top_pixels.sh aorta false false true
# sbatch top_pixels.sh gm12868 false true true
# sbatch top_pixels.sh h1 false true true
# sbatch top_pixels.sh hepg2 false true false
# sbatch top_pixels.sh hippocampus false false true
# sbatch top_pixels.sh hmec false true false
# sbatch top_pixels imr90 false true true
# sbatch top_pixels.sh k562 false true false
# sbatch top_pixels.sh left_ventricle false false true
# sbatch top_pixels.sh liver false false true
# sbatch top_pixels.sh lung false false true
# sbatch top_pixels.sh nhek false true false
# sbatch top_pixels.sh pancreas false false true
# sbatch top_pixels.sh skeletal_muscle false false true
# sbatch top_pixels.sh small_intestine false false true
"""