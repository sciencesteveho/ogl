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
raw_dir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files
peakmerge_dir=/ocean/projects/bio210019p/stevesho/remap2022
working_dir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs

function _get_crms_from_epimap () {
    minsize=1
    for file in ${1}/*narrow.5*;
    do
        file_size=$(wc -l $file | cut -d' ' -f1)
        if [ $file_size -ge $minsize ]; then
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

    # copy to raw_files
    cp \
        ${working_dir}/${tissue}/crms/consensuses.bed \
        ${raw_dir}/${tissue}/consensuses.bed

    # simple moving
    rawdir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files
    bigwigdir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs/
    for file in ATAC-seq_narrow_5_merged.bed CTCF_narrow_5_merged.bed DNase-seq_narrow_5_merged.bed H3K27ac_narrow_5_merged.bed H3K27me3_narrow_5_merged.bed H3K36me3_narrow_5_merged.bed H3K4me1_narrow_5_merged.bed H3K4me2_narrow_5_merged.bed H3K4me3_narrow_5_merged.bed H3K79me2_narrow_5_merged.bed H3K9ac_narrow_5_merged.bed H3K9me3_narrow_5_merged.bed POLR2A_narrow_5_merged.bed RAD21_narrow_5_merged.bed SMC3_narrow_5_merged.bed;
    do
        cp ${bigwigdir}/${tissue}/merged/${file} ${rawdir}/${tissue}/${file}
    done
done


# change H3K9me3 files from narrow -c 5 to narrow -c 4. Recall CRMs.
bigwigdir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs
rawdir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files
for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle skin small_intestine;
do
    echo $tissue
    files=$(ls ${bigwigdir}/${tissue}/peaks | grep H3K9me3 | grep narrow | grep "\.4\.")

    for file in $files;
    do
        cp ${bigwigdir}/${tissue}/peaks/${file} ${bigwigdir}/${tissue}/crms_processing/${file}
    done

    files_2=$(ls ${bigwigdir}/${tissue}/crms_processing | grep H3K9me3 | grep narrow | grep "\.5\.")
    for file2 in $files_2;
    do
        rm ${bigwigdir}/${tissue}/crms_processing/${file2}
    done

    cp \
        ${bigwigdir}/${tissue}/merged/H3K9me3_narrow_4_merged.bed \
        ${rawdir}/${tissue}/H3K9me3_narrow_4_merged.bed

    rm ${rawdir}/${tissue}/H3K9me3_narrow_5_merged.bed
done


"""
cp impute_BSS01869_H3K36me3.narrow.4._lifted_hg38.bed ../crms_processing/impute_BSS01869_H3K36me3.narrow.4._lifted_hg38.bed
bedops -m FINAL_H3K36me3_BSS00150.sub_VS_FINAL_WCE_BSS00150.pval.signal.bedgraph.gz.narrow.5._lifted_hg38.bed impute_BSS01869_H3K36me3.narrow.4._lifted_hg38.bed impute_BSS01871_H3K36me3.narrow.5._lifted_hg38.bed | awk -v FS='\t' '{print $1, $2, $3}' > ../merged/H3K36me3_narrow_5_merged.bed
cp ../merged/H3K36me3_narrow_5_merged.bed ../../lung/H3K36me3_narrow_5_merged.bed

cp impute_BSS00122_H3K36me3.narrow.4._lifted_hg38.bed ../crms_processing/impute_BSS00122_H3K36me3.narrow.4._lifted_hg38.bed
bedops -m FINAL_H3K36me3_BSS00123.sub_VS_FINAL_WCE_BSS00123.pval.signal.bedgraph.gz.narrow.5._lifted_hg38.bed impute_BSS00122_H3K36me3.narrow.4._lifted_hg38.bed impute_BSS00758_H3K36me3.narrow.5._lifted_hg38.bed | awk -v FS='\t' '{print $1, $2, $3}' > ../merged/H3K36me3_narrow_5_merged.bed
cp ../merged/H3K36me3_narrow_5_merged.bed ../../pancreas/H3K36me3_narrow_5_merged.bed

raw_dir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files
peakmerge_dir=/ocean/projects/bio210019p/stevesho/remap2022
working_dir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs

function _remv_epimap () {
    minsize=1
    for file in ${1}/*narrow.5*;
    do
        file_size=$(wc -l $file | cut -d' ' -f1)
        if [ $file_size -ge $minsize ]; then
            cp $file ${2}
        fi
    done
}

for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle skin small_intestine;
do
    _remv_epimap \
        ${working_dir}/${tissue}/peaks \
        ${working_dir}/${tissue}/crms_processing
done

bigwigdir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs
rawdir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files
for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle skin small_intestine;
do
    echo $tissue
    files=$(ls ${bigwigdir}/${tissue}/peaks | grep H3K9me3 | grep narrow | grep "\.4\.")

    for file in $files;
    do
        cp ${bigwigdir}/${tissue}/peaks/${file} ${bigwigdir}/${tissue}/crms_processing/${file}
    done

    files_2=$(ls ${bigwigdir}/${tissue}/crms_processing | grep H3K9me3 | grep narrow | grep "\.5\.")
    for file2 in $files_2;
    do
        rm ${bigwigdir}/${tissue}/crms_processing/${file2}
    done
done


for tissue in lung pancreas;
do
    echo $tissue
    ls /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs/${tissue}/crms_processing | wc -l 
    for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        echo $tissue $feature
        ls /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs/${tissue}/crms_processing | grep ${feature} | wc -l
    done
done



function _get_crms_from_epimap () {
    python ${peakmerge_dir}/peakMerge.py \
        "/ocean/projects/bio210019p/stevesho/resources/hg38.chrom.sizes.autosomes.txt" \
        ${2} \
        "narrowPeak" \
        ${3}
}

for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle skin small_intestine;
do
    _get_crms_from_epimap \
        ${working_dir}/${tissue}/peaks \
        ${working_dir}/${tissue}/crms_processing \
        ${working_dir}/${tissue}/crms/

    # copy to raw_files
    cp \
        ${working_dir}/${tissue}/crms/consensuses.bed \
        ${raw_dir}/${tissue}/consensuses.bed
done


"""