#!/bin/bash
#
# Scripts to retrieve top pixel contacts from enhanced, denoised hi-c data from
# deeploop. Also convolves the matrix with a small Gaussian filter as an option.

# """
# 
# for tissue in Aorta Hippocampus LeftVentricle Liver Lung Pancreas Psoas_Muscle Small_Intestine;
# do
#     sbatch top_pixels.sh $tissue
# done
# """

# setting up variables
SECONDS=0

# Arguments:
#   $1 - tissue name
#   $2 - one level above /path/to/data
function _extract_from_cooler () {
    # set up vars
    tissue_name=$1
    tissue=$(echo $tissue_name | awk '{print tolower($0)}')

    # get pixels from cooler
    cooler dump \
        -t pixels \
        --join \
        -o $2/top_pixels/${tissue}.pixels \
        $2/coolers/${tissue_name}.cool
    
    # sort by intensity
    sort --parallel=8 -S 60% -k7,7nr $2/top_pixels/${tissue}.pixels > $2/top_pixels/${tissue}.pixels.sorted

    # take top 5M, 10M, 50M pixels
    for i in 5000000 10000000 50000000; do
        head -n ${i} $2/top_pixels/${tissue}.pixels.sorted > $2/top_pixels/${tissue}_${i}.pixels
    done

    # filter for counts >= 1
    awk '$7 >= 1' $2/top_pixels/${tissue}.pixels.sorted > $2/top_pixels/${tissue}_gte1.pixels

    # try balancing
    cp $2/coolers/${tissue_name}.cool $2/${tissue_name}.cool
    cooler balance \
        -p 12 \
        --force \
        $2/${tissue_name}.cool
}

# run main function!
_extract_from_cooler \
    $1 \
    $2