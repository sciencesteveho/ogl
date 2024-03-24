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
# Format deeploop pixels to bedpe for downloaded contacts
# Arguments:
#   $1 - directory of deeploop files
#   $2 - filename of downloaded deeploop contacts
#   $3 - directory to place processed file
#   $4 - filename without extension of processed file
# =============================================================================
function _deeploop_txt_to_bedpe () {
    local input_dir=$1
    local input_prefix=$2
    local output_dir=$3
    local output_prefix=$4

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
# Arguments:
#   $1 - directory of deeploop files
#   $2 - filename without extension
#   $3 - directory of liftover and liftover chain
#   $4 - tmp directory to process files
# =============================================================================
function _liftover_deeploop_bedpe () {
    local input_dir=$1
    local tissue_file=$2
    local resource_dir=$3
    local output_dir=$4
    local chain_file="${resource_dir}/hg19ToHg38.over.chain.gz"

    # split bedpe into two files for lifting
    awk -v OFS='\t' '{print $1,$2,$3,$7,NR}' "${input_dir}/${tissue_file}" > "${output_dir}/${tissue_file}._1"
    awk -v OFS='\t' '{print $4,$5,$6,$7,NR}' "${input_dir}/${tissue_file}" > "${output_dir}/${tissue_file}._2"

    # liftover each anchor
    for file in ._1 ._2; do
        "${resource_dir}/liftOver" \
            "${output_dir}/${tissue_file}.${file}" \
            "${chain_file}" \
            "${output_dir}/${tissue_file}.${file}.hg38" \
            "${output_dir}/${tissue_file}.${file}.unmapped"

        sort -k5,5 -o "${output_dir}/${tissue_file}.${file}.hg38" "${output_dir}/${tissue_file}.${file}.hg38"
    done

    # join the lifted files
    join -j 5 \
        -o 1.1,1.2,1.3,2.1,2.2,2.3,1.4 \
        "${output_dir}/${tissue_file}._1.hg38" \
        "${output_dir}/${tissue_file}._2.hg38" \
        | sed 's/ /\t/g' \
        | awk -v OFS='\t' '{print $1,$2,$3,$4,$5,$6,$7}' \
        > "${input_dir}/${tissue_file}.hg38"

    # cleanup
    for file in ._1 ._2; do
        rm "${output_dir}/${tissue_file}.${file}" "${output_dir}/${tissue_file}.${file}.hg38" "${output_dir}/${tissue_file}.${file}.unmapped"
    done
}


# =============================================================================
# Extract top pixels from cooler files.
# Arguments:
#   $1 - tissue name
#   $2 - one level above /path/to/data
#   $3 - bool to extract cooler if necessary
#   $4 - bool to adjust start and end positions by -1
# =============================================================================
function _extract_top_pixels () {
    local tissue=$1
    local working_dir=$2
    local extract_cooler=$3
    local zero_index=$4
    local liftover=$5

    local pixels_dir="${working_dir}/pixels"
    local coolers_dir="${working_dir}/coolers"
    local liftover_resources_dir="/ocean/projects/bio210019p/stevesho/resources"
    local tmp_dir="${working_dir}/tmp"
    local pixels_file="${pixels_dir}/${tissue}.pixels"
    local sorted_pixels_file="${pixels_file}_sorted"

    mkdir -p "${pixels_dir}" "${coolers_dir}" "${tmp_dir}"

    if [[ "$extract_cooler" == true ]]; then
        cooler dump -t pixels --join -o "${pixels_file}" "${coolers_dir}/${tissue}.cool"
    fi

    # Adjust to zero_index and liftover 
    local intermediate_file="${pixels_file}"
    if [[ "$zero_index" == true ]]; then
        intermediate_file+=".zero_indexed"
        awk -v OFS='\t' '{print $1,$2-1,$3,$4,$5,$6,$7}' "${pixels_file}" > "${intermediate_file}"
    fi

    if [[ "$liftover" == true ]]; then
        _liftover_deeploop_bedpe \
            "${working_dir}" \
            "${intermediate_file}" \
            "${liftover_resources_dir}" \
            "${tmp_dir}"
        intermediate_file+=".hg38"
    fi

    # Sort by e/o ratio to extract top pixels
    sort --parallel=8 -S 80% -k7,7nr "${intermediate_file}" > "${sorted_pixels_file}"
    
    local thresholds=(50000 100000 150000 200000 300000 500000 1000000)
    local threshold
    for threshold in "${thresholds[@]}"; do
        local output_dir="${working_dir}/top_n_pixels/top_${threshold}"
        mkdir -p "${output_dir}"
        head -n "${threshold}" "${sorted_pixels_file}" > "${output_dir}/${tissue}_${threshold}.pixels"
    done

    local output_dir="${working_dir}/top_n_pixels/gte1"
    awk '$7 >= 1' "${sorted_pixels_file}" > "${output_dir}/${tissue}_gte1.pixels"
}


# =============================================================================
# run main functions!
# filenames=(Aorta Hippocampus LeftVentricle Liver Lung Pancreas Psoas_Muscle Small_Intestine)
# filenames=(aorta hippocampus leftventricle liver lung pancreas psoas_muscle small_intestine)
# for name in ${filenames[@]}; do
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