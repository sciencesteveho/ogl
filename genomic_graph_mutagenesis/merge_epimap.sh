#!/bin/bash
#
# This script is used to convert epimap bigwigs to narrow or broad peak bedfiles
# with liftover to hg38. Broad peaks are called for histone marks and narrow
# peaks for TFs. Histones and TFs are then merged across the three samples to
# act as "representative" potential tracks for the given tissue.
SECONDS=0

# convert bigwig to bed
# wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bigWigToBedGraph
# Arguments:
#   $1 - pseudo bool, true if histone, false if TF
#   $2 - directory to bigWigToBedGraph
#   $3 - bigwig tissue directory
#   $4 - filename without extension
# function _bigWig_to_peaks () {
#     local histone="${1:-false}"

#     $2/bigWigToBedGraph ${3}/${4}.bigWig ${3}/tmp/${4}.bedGraph
#     if "$histone"; then
#         macs2 bdgbroadcall \
#             -i ${3}/tmp/${4}.bedGraph \
#             -o ${3}/tmp/${4}.bed \
#             -c 2 \
#             -l 73 \
#             -g 100

#     else
#         macs2 bdgpeakcall \
#             -i ${3}/tmp/${4}.bedGraph \
#             -o ${3}/tmp/${4}.bed \
#             -c 2 \
#             -l 73 \
#             -g 100
#     fi
#     # cleanup 
#     tail -n +2 ${3}/tmp/${4}.bed > tmpfile && mv tmpfile ${3}/tmp/${4}.bed
# }

function _bigWig_to_peaks () {
    local histone="${1:-false}"

    for pval in 3 4 5;
    do
        $2/bigWigToBedGraph ${3}/${4}.bigWig ${3}/tmp/${4}.bedGraph
        macs2 bdgbroadcall \
            -i ${3}/tmp/${4}.bedGraph \
            -o ${3}/tmp/${4}.broad.${pval}.bed \
            -c ${pval} \
            -l 73 \
            -g 100

        macs2 bdgpeakcall \
            -i ${3}/tmp/${4}.bedGraph \
            -o ${3}/tmp/${4}.narrow.${pval}.bed \
            -c ${pval} \
            -l 73 \
            -g 100

        for peak in broad narrow;
        do
            tail -n +2 ${3}/tmp/${4}.${peak}.${pval}.bed > tmpfile && mv tmpfile ${3}/tmp/${4}.${peak}.${pval}.bed
        done
    done

    # cleanup 

}


# liftover hg19 to hg38
# Arguments:
#   $1 - directory to liftOver
#   $2 - bigwig tissue directory
#   $3 - filename without extension
function _liftover_19_to_38 () {
    $1/liftOver \
        $2/tmp/${3}.bed \
        $1/hg19ToHg38.over.chain.gz \
        $2/peaks/${3}._lifted_hg38.bed \
        $2/tmp/${3}.unlifted

    # cleanup
    rm $2/tmp/${3}.unlifted
}

# merge epimap peaks
# Arguments:
#   $1 - directory to parsed peaks
#   $2 - directory to save merged peaks
function _merge_epimap_features () {
    cd $1
    for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        for peak in broad narrow;
        do
            for num in 3 4 5;
            do
                files=$(ls $1 | grep ${feature} | grep "\.${num}\." | grep $peak)
                bedops -m $1/${files} | awk -v FS='\t' '{print $1, $2, $3}' > $2/${feature}_${peak}_${num}_merged.bed
            done
        done
    done
}

# main function to perform processing in a given tissue!
#   $1 - directory to bigwigs
#   $2 - name of tissue
#   $3 - directory to liftOver
function main_func () {
    # make directories if they don't exist
    for dir in merged peaks tmp crms;
    do
        if [ ! -d $1/$2/$dir ]; then
            mkdir $1/$2/$dir
        fi
    done

    # loop through bigwigs and convert to peaks, based on which epi feature in
    # filename
    for file in $1/$2/*;
    do
        if [ -f $file ]; then
            name=$(echo $(basename ${file}) | sed 's/\.bigWig//g')
            case $name in
                *H3K27me3* | *H3K36me3* | *H3K4me1* | *H3K79me2*)
                    _bigWig_to_peaks \
                        true \
                        $3 \
                        $1/$2 \
                        $name
                    ;;
                *ATAC-seq* | *CTCF* | *DNase-seq* | *POLR2A* | *RAD21* | *SMC3* | *H3K27ac* | *H3K4me2* | *H3K4me3* | *H3K9ac* | *H3K9me3*)
                    _bigWig_to_peaks \
                        false \
                        $3 \
                        $1/$2 \
                        $name
                    ;;
            esac

            # liftover to hg38
            for peak in broad narrow;
            do
                for pval in 3 4 5;
                do
                    _liftover_19_to_38 \
                        $3 \
                        $1/$2 \
                        $name.${peak}.${pval}
                done
            done
        fi
    done

    # merge epimap features across 3 samples
    _merge_epimap_features \
        $1/$2/peaks \
        $1/$2/merged
}

# run main_func function! 
#   $1 - bigwig directory, one level before tissues
#   $2 - name of tissue
#   $3 - directory to liftover and bigwig conversion file
main_func \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs \
    $1 \
    /ocean/projects/bio210019p/stevesho/resources


t=$SECONDS
printf 'Time elapsed: %d days, %d minutes\n' "$(( t/86400 ))" "$(( t/60 - 1440*(t/86400) ))"

# get concensus from epimap peaks
# peakmerge_dir=$2
peakmerge_dir=/ocean/projects/bio210019p/stevesho/remap2022
working_dir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs

function _get_crms_from_epimap () {
    minsize=1
    for file in ${1}/*narrow.5*;
    do
        file_size=$(wc -l $file | cut -d' ' -f1)
        if [ $file_size -gt $minsize ]; then
            cp $file ${2}
        fi
    done

    python ${peakmerge_dir}/peakMerge.py \
        "/ocean/projects/bio210019p/stevesho/resources/hg38.chrom.sizes.autosomes.txt" \
        ${2} \
        "narrowPeak" \
        ${3}
}

for tissue in hela hippocampus k562 left_ventricle liver lung mammary npc pancreas skeletal_muscle skin small_intestine;
do
    _get_crms_from_epimap \
        ${working_dir}/${tissue}/peaks \
        ${working_dir}/${tissue}/crms_processing \
        ${working_dir}/${tissue}/crms/
done