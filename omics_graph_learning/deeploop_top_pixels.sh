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

function convertsecs() {
 ((h=${1}/3600))
 ((m=(${1}%3600)/60))
 ((s=${1}%60))
 printf "%02d:%02d:%02d\n" $h $m $s
}

# Arguments:
#   $1 - tissue name
#   $2 - one level above /path/to/data
function _extract_from_cooler () {
    # set up vars
    tissue_name=$1
    tissue=$(echo $tissue_name | awk '{print tolower($0)}')

    # # get pixels from cooler
    # cooler dump \
    #     -t pixels \
    #     --join \
    #     -o $2/top_pixels/${tissue}.pixels \
    #     $2/coolers/${tissue_name}.cool

    # sort by intensity
    sort --parallel=8 -S 80% -k7,7nr $2/top_pixels/${tissue}.pixels > $2/top_pixels/${tissue}.pixels.sorted

    # take top 5M, 10M, 50M pixels
    for i in 1000000 2000000 3000000 4000000 5000000 10000000; do
        head -n ${i} $2/top_pixels/${tissue}.pixels.sorted > $2/top_pixels/${tissue}_${i}.pixels
    done

    # filter for counts >= 1
    awk '$7 >= 1' $2/top_pixels/${tissue}.pixels.sorted > $2/top_pixels/${tissue}_gte1.pixels

    # # try balancing
    # cp $2/coolers/${tissue_name}.cool $2/${tissue_name}.cool
    # cooler balance \
    #     -p 12 \
    #     --force \
    #     $2/${tissue_name}.cool
}

# liftover deeploop files
# Arguments:
#   $1 - directory of deeploop files
#   $2 - filename without extension
#   $3 - directory of liftover and liftover chain
#   $4 - tmp directory to process files
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

# format loops and keep top N interactions
# Arguments:
#   $1 - tmp dir to process files
#   $2 - filename without extension
#   $3 - directory to store final files
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

# cleanup files
# Arguments:
#   $2 - filename without extension
#   $3 - tmp dir to process files
function _cleanup_liftover () {
    # cleanup
    for file in $1.pixels_1 $1.pixels_1.hg38 $1.pixels_1.unmapped $1.pixels_2 $1.pixels_2.hg38 $1.pixels_2.unmapped;
    do
        rm $2/$file
    done
}

# main function!
# Arguments:
#   $1 - filename without extension
#   $2 - /path/to/pixels
#   $3 - /path/to/resources
#   $4 - final path to deposit chromatin contacts
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

# run main functions!
# filenames=(Aorta Hippocampus LeftVentricle Liver Lung Pancreas Psoas_Muscle Small_Intestine)
_extract_from_cooler \
    ${1} \
    /ocean/projects/bio210019p/stevesho/hic

# filenames=(aorta hippocampus leftventricle liver lung pancreas psoas_muscle small_intestine)
# for name in ${filenames[@]}; do
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