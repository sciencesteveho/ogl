#!/bin/bash
#
# This script is used to convert epimap bigwigs to narrow or broad peak bedfiles
# with liftover to hg38. Broad peaks are called for histone marks and narrow
# peaks for TFs. Histones and TFs are then merged across the three samples to
# act as "representative" potential tracks for the given tissue.
# SECONDS=0

# convert bigwig to bed
# wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bigWigToBedGraph
# If tracks are imputed AND track is one of the marks seen in the cutoffs array,
# then the cutoff value is changed to match the cutoff values provided by C.
# Boix
# Arguments:
#   $1 - pseudo bool, true if histone, false if TF
#   $2 - directory to bigWigToBedGraph
#   $3 - bigwig tissue directory
#   $4 - filename without extension
function _bigWig_to_peaks () {
    $1/bigWigToBedGraph ${2}/${3}.bigWig ${2}/tmp/${3}.bedGraph

    echo "calling peaks on ${3} with cutoff ${4}"
    macs2 bdgpeakcall \
        -i ${2}/tmp/${3}.bedGraph \
        -o ${2}/tmp/${3}.narrow.${4}.bed \
        -c ${4} \
        -l 73 \
        -g 100

    # cleanup
    tail -n +2 ${2}/tmp/${3}.narrow.bed > tmpfile && mv tmpfile ${2}/tmp/${3}.narrow.bed
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
        files=$(ls $1 | grep ${feature})
        bedops -m $1/${files} \
            | awk -vOFS='\t' -v feature=$feature '{print $1, $2, $3, feature}' \
            > $2/${feature}_narrow_merged.bed
    done
}

# Arguments:
#   $1 - directory to converted and lifted peaks
#   $2 - directory to save peak copies
#   $3 - full path to hg38 chrom sizes
#   $3 - working directory
function _get_crms_from_epimap () {
    peakmerge_dir=/ocean/projects/bio210019p/stevesho/remap2022
    minsize=1

    for file in ${1}/*narrow*;
    do
        file_size=$(wc -l $file | cut -d' ' -f1)
        if [ $file_size -ge $minsize ]; then
            filename=$(basename $file)
            cp $file ${2}/$filename
        fi
    done

    python ${peakmerge_dir}/peakMerge.py \
        ${3} \
        ${2} \
        "narrowPeak" \
        ${4}
}

# main function to perform processing in a given tissue!
#   $1 - directory to bigwigs
#   $2 - name of tissue
#   $3 - directory to liftOver
function main_func () {

    # set up cutoff array
    declare -A cutoffs
	cutoffs["DNase-seq"] = 1.9
	cutoffs["H3K27ac"] = 2.2
	cutoffs["H3K27me3"] = 1.2
	cutoffs["H3K36me3"] = 1.7
	cutoffs["H3K4me1"] = 1.7
	cutoffs["H3K4me2"] = 2
	cutoffs["H3K4me3"] = 1.7
	cutoffs["H3K79me2"] = 2.2
	cutoffs["H3K9ac"] = 1.6
	cutoffs["H3K9me3"] = 1.1
    cutoffs["ATAC-seq"] = 2
    cutoffs["CTCF"] = 2
    cutoffs["POLR2A"] = 2
    cutoffs["RAD21"] = 2
    cutoffs["SMC3"] = 2

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
            # set variables
            prefix=$(echo $(basename ${file}) | cut -d'_' -f1)
            histone=$(echo $(basename ${file}) | cut -d'_' -f2)
            name=$(echo $(basename ${file}) | sed 's/\.bigWig//g')

            # get cutoffs for imputed tracks. Observed tracks get a cutoff of 5
            if [ $prefix == "impute"]; then
                pval=${cutoffs[$histone]}
            else
                pval=5
            fi

            _bigWig_to_peaks \
                $3 \
                $1/$2 \
                $name \
                $pval

            # liftover to hg38
            _liftover_19_to_38 \
                $3 \
                $1/$2 \
                ${name}.narrow
        fi
    done

    # merge epimap features across 3 samples
    _merge_epimap_features \
        $1/$2/peaks \
        $1/$2/merged

    # call crms via remap method
    _get_crms_from_epimap \
        $1/$2/peaks \
        $1/$2/crms_processing \
        $4 \
        $1/$2/crms/

    # copy crms to raw_files directory
    cp \
        $1/$2/crms/consensuses.bed \
        $5/$2/consensuses.bed

    # copy merged peaks to raw_files directory
    cp \
        $1/$2/merged/* \
        $5/$2/
    
}

# run main_func function!
#   $1 - bigwig directory, one level before tissues
#   $2 - name of tissue
#   $3 - directory to liftover and bigwig conversion file
main_func \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs \
    $1 \
    /ocean/projects/bio210019p/stevesho/resources \
    "/ocean/projects/bio210019p/stevesho/resources/hg38.chrom.sizes.autosomes.txt" \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files


# t=$SECONDS
# printf 'Time elapsed: %d days, %d minutes\n' "$(( t/86400 ))" "$(( t/60 - 1440*(t/86400) ))"