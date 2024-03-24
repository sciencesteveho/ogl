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
# setting up variables
# =============================================================================
SECONDS=0

function convertsecs() {
 ((h=${1}/3600))
 ((m=(${1}%3600)/60))
 ((s=${1}%60))
 printf "%02d:%02d:%02d\n" $h $m $s
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
    local zero_index=$4  # bool that tells the script to adjust start and end positions by -1

    local pixels_dir="${working_dir}/top_pixels"
    local coolers_dir="${working_dir}/coolers"
    local pixels_file="${pixels_dir}/${tissue}.pixels"
    local sorted_pixels_file="${pixels_file}_sorted"

    # check if dirs exist
    mkdir -p "${pixels_dir}"
    mkdir -p "${coolers_dir}"

    if [[ "$extract_cooler" == true ]]; then
        # get pixels from cooler
        cooler dump \
            -t pixels \
            --join \
            -o "${pixels_file}" \
            "${coolers_dir}/${tissue}.cool"
    fi

    # sort by intensity
    if [[ "$zero_index" == true ]]; then
        sort --parallel=8 -S 80% -k7,7nr "${pixels_file}" | awk -v OFS='\t' '{print $1,$2-1,$3,$4,$5,$6,$7}' > "${sorted_pixels_file}"
    else
        sort --parallel=8 -S 80% -k7,7nr "${pixels_file}" > "${sorted_pixels_file}"
    fi
    
    # take top N pixels
    for i in 50000 100000 150000 200000 300000 500000 1000000; do
        head -n $i "${sorted_pixels_file}" > "${pixels_dir}/${tissue}_${i}.pixels"
    done

    # filter for counts >= 1 (should this be >=1 or >=2? It was inconsistent with the comments)
    awk '$7 >= 1' "${sorted_pixels_file}" > "${pixels_dir}/${tissue}_gte1.pixels"
}



# =============================================================================
# liftover deeploop files
# Arguments:
#   $1 - directory of deeploop files
#   $2 - filename without extension
#   $3 - directory of liftover and liftover chain
#   $4 - tmp directory to process files
# =============================================================================
function _liftover_deeploop_bedpe () {

    awk -v OFS='\t' '{print $1,$2,$3,NR}' $1/$2.pixels > $4/$2.pixels_1
    awk -v OFS='\t' '{print $4,$5,$6,NR}' $1/$2.pixels > $4/$2.pixels_2

    for file in pixels_1 pixels_2;
    do
        $3/liftOver \
            $4/$2.${file} \
            $3/hg19ToHg38.over.chain.gz \
            $4/$2.${file}.hg38 \
            $4/$2.${file}.unmapped

        sort -k4,4 -o $4/$2.${file}.hg38 $4/$2.${file}.hg38
    done
}


# =============================================================================
# format loops and keep top N interactions
# Arguments:
#   $1 - tmp dir to process files
#   $2 - filename without extension
#   $3 - directory to store final files
# =============================================================================
function _format_deeploop_bedpe () {
    join \
        -j 4 \
        -o 1.1,1.2,1.3,2.1,2.2,2.3 \
        $1/$2.pixels_1.hg38 \
        $1/$2.pixels_2.hg38 \
        | sed 's/ /\t/g' \
        | awk -v OFS='\t' '{print $1,$2,$3,$4,$5,$6}' \
        | sort --parallel=8 -S 50% -k1,1 -k2,2n \
        > $3/${2}_pixels.hg38
}


# =============================================================================
# cleanup files
# Arguments:
#   $2 - filename without extension
#   $3 - tmp dir to process files
# =============================================================================
function _cleanup_liftover () {
    # cleanup
    for file in $1.pixels_1 $1.pixels_1.hg38 $1.pixels_1.unmapped $1.pixels_2 $1.pixels_2.hg38 $1.pixels_2.unmapped;
    do
        rm $2/$file
    done
}


# =============================================================================
# main function!
# Arguments:
#   $1 - filename without extension
#   $2 - /path/to/pixels
#   $3 - /path/to/resources
#   $4 - final path to deposit chromatin contacts
# =============================================================================
deeploop_processing_main () {
    # set vars
    filename=$1
    loop_dir=$2
    resource_dir=$3
    deposit_dir=$4
    tmp_dir=${loop_dir}/tmp

    # make directory for processing
    if [ ! -d ${loop_dir}/tmp ]; then
        mkdir ${loop_dir}/tmp
    fi

    _liftover_deeploop_bedpe \
        $loop_dir \
        $filename \
        $resource_dir \
        $tmp_dir

    _format_deeploop_bedpe \
        $tmp_dir \
        $filename \
        $deposit_dir

    _cleanup_liftover \
        $filename \
        $tmp_dir
}


# =============================================================================
# run main functions!
# filenames=(Aorta Hippocampus LeftVentricle Liver Lung Pancreas Psoas_Muscle Small_Intestine)
# filenames=(aorta hippocampus leftventricle liver lung pancreas psoas_muscle small_intestine)
# for name in ${filenames[@]}; do
# =============================================================================
# format deeploop pixels for the cell-lines with top N txts available for download
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


_extract_top_pixels \
    ${1} \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic \
    false


tissue_name=$(echo $1 | awk '{print tolower($0)}')
for addendum in 1000000 2000000 3000000 4000000 5000000 10000000;
do
    echo "Processing ${name}_${addendum}..."
    deeploop_processing_main \
        ${tissue_name}_${addendum} \
        /ocean/projects/bio210019p/stevesho/hic/top_pixels \
        /ocean/projects/bio210019p/stevesho/resources \
        /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/processed_loops/deeploop_${addendum}
done
# done

end=`date +%s`
time=$((end-start))
echo "Finished! in $(convertsecs $time)"