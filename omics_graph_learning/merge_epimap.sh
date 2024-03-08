#!/bin/bash
#
# This script is used to convert epimap bigwigs to narrow or broad peak bedfiles
# with liftover to hg38. Broad peaks are called for histone marks and narrow
# peaks for TFs. Histones and TFs are then merged across the three samples to
# act as "representative" potential tracks for the given tissue.
# SECONDS=0

# =============================================================================
# liftover hg19 to hg38
# Arguments:
#   $1 - directory to liftOver
#   $2 - bigwig tissue directory
#   $3 - filename without extension
# =============================================================================
function _liftover_19_to_38 () {
    $1/liftOver \
        $2/tmp/${3}.bed \
        $1/hg19ToHg38.over.chain.gz \
        $2/peaks/${3}._lifted_hg38.bed \
        $2/tmp/${3}.unlifted

    # cleanup
    rm $2/tmp/${3}.unlifted
}


function _merge_bedgraphs () {
    function _liftover () {
        $1/liftOver \
            ${2}_merged.narrow.bed \
            $1/hg19ToHg38.over.chain.gz \
            ${2}_merged.narrow.hg38.bed \
            ${2}.unlifted

        # cleanup
        rm ${2}.unlifted
    }

    # set up cutoff array
    mkdir merged_bedgraph
    cd tmp
    rm *narrow.bed*

    declare -A cutoffs
    # cutoffs provided by C. Boix
	cutoffs["DNase-seq"]=1.9
	cutoffs["H3K27ac"]=2.2
	cutoffs["H3K27me3"]=1.2
	cutoffs["H3K36me3"]=1.7
	cutoffs["H3K4me1"]=1.7
	cutoffs["H3K4me2"]=2
	cutoffs["H3K4me3"]=1.7
	cutoffs["H3K79me2"]=2.2
	cutoffs["H3K9ac"]=1.6
	cutoffs["H3K9me3"]=1.1
    # these following cutoffs default to 2
	cutoffs["ATAC-seq"]=2
	cutoffs["CTCF"]=2
	cutoffs["POLR2A"]=2
	cutoffs["RAD21"]=2
	cutoffs["SMC3"]=2

    # for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    for feature in CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        # Collect files into an array
        files=( $(ls | grep "${feature}") )
        
        # Check that we have exactly three files
        if [ "${#files[@]}" -ne 3 ]; then
            echo "Error: Expected 3 files for feature ${feature}, but found ${#files[@]}."
            continue
        fi
        
        # Sum the coverage values using bedtools unionbedg and awk
        echo "Merging ${feature} bedgraphs"

        bedtools unionbedg -i <(sort -k1,1 -k2,2n "${files[0]}") <(sort -k1,1 -k2,2n "${files[1]}") <(sort -k1,1 -k2,2n "${files[2]}") \
            | awk '{sum=$4+$5+$6; print $1, $2, $3, sum / 3}' > "${feature}_merged.bedGraph"
        echo "Merged ${feature} bedgraphs"

        echo "calling peaks with cutoff ${cutoffs[$feature]}"
        macs2 bdgpeakcall \
            -i ${feature}_merged.bedGraph \
            -o ../merged_bedgraph/${feature}_merged.narrow.bed \
            -c ${cutoffs[$feature]} \
            -l 73 \
            -g 100

        cd ../merged_bedgraph
        tail -n +2 ${feature}_merged.narrow.bed > tmpfile && mv tmpfile ${feature}_merged.narrow.bed
        echo "liftover ${feature} to hg38"
        _liftover \
            /ocean/projects/bio210019p/stevesho/resources \
            ${feature}
        cd -
    done
}

_merge_bedgraphs

# =============================================================================
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
# The output headers are the following: chr, start, end, name, score, strad, signalValue, pValue, qValue, peak summit relative to peak start
# =============================================================================
function _bigWig_to_peaks () {
    $1/bigWigToBedGraph ${2}/${3}.bigWig ${2}/tmp/${3}.bedGraph

    echo "calling peaks on ${3} with cutoff ${4}"
    macs2 bdgpeakcall \
        -i ${2}/tmp/${3}.bedGraph \
        -o ${2}/tmp/${3}.narrow.bed \
        -c ${4} \
        -l 73 \
        -g 100

    # cleanup
    tail -n +2 ${2}/tmp/${3}.narrow.bed > tmpfile && mv tmpfile ${2}/tmp/${3}.narrow.bed
}


# =============================================================================
# merge epimap peaks, naively 
# Arguments:
#   $1 - directory to parsed peaks
#   $2 - directory to save merged peaks
# =============================================================================
function _merge_epimap_features_naive () {
    cd $1
    for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        files=$(ls $1 | grep ${feature})
        bedops -m $1/${files} \
            | awk -vOFS='\t' -v feature=$feature '{print $1, $2, $3, feature}' \
            > $2/${feature}_narrow_merged.bed
    done
}


# =============================================================================
# merge epimap peaks heuristically 
# We first keep peaks with at least 50% overlap. Then we only keep peaks were the summits are within 300bp of eachother.
# Arguments:
#   $1 - directory to parsed peaks
#   $2 - directory to save merged peaks
# =============================================================================
# Function to merge two narrowPeak files with summit distance constraint
_merge_with_summit_dist() {
    inputFile1="$1"
    inputFile2="$2"
    outputMergedFile="$3"
    maxDist="$4"

    # Perform reciprocal overlapping with at least 50% overlap in both files
    bedtools intersect -a "$inputFile1" -b "$inputFile2" -f 0.5 -r -wo \
    | awk -v maxDist="$maxDist" '{
        # Calculate distance between peak summits
        summitDist = ($2 + $10) - ($13 + $16)
        if (summitDist < 0) {
            summitDist = -summitDist
        }
        
        # Apply the summit condition based on the variable
        if (summitDist <= maxDist) {
            print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9
        }
      }' > "$outputMergedFile"
}

function _merge_epimap_heuristically_summits () {
    # Define the maximum allowed distance between summits as an example
    peakdir="$1"
    mergeddir="$2"

    # set up peakdist array
    declare -A peakdist
    # cutoffs provided by C. Boix
	peakdist["DNase-seq"]=200
	peakdist["H3K27ac"]=300
	peakdist["H3K27me3"]=300
	peakdist["H3K36me3"]=300
	peakdist["H3K4me1"]=300
	peakdist["H3K4me2"]=300
	peakdist["H3K4me3"]=300
	peakdist["H3K79me2"]=300
	peakdist["H3K9ac"]=300
	peakdist["H3K9me3"]=300
    # these following cutoffs default to 2
	peakdist["ATAC-seq"]=200
	peakdist["CTCF"]=200
	peakdist["POLR2A"]=200
	peakdist["RAD21"]=200
	peakdist["SMC3"]=200
    
    cd $peakdir
    # Loop through each feature
    for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        # get max distance b/w summits
        maxDist=${peakdist[$feature]}

        # store all files in an array
        files=($(ls $peakdir | grep ${feature}))
        
        # Check if exactly 3 files are present
        if [ "${#files[@]}" -ne 3 ]; then
            echo "Error: Expected 3 files for feature ${feature}, but found ${#files[@]}." >&2
            continue
        fi
        
        # Merge the first two files
        intermediateOutput="${feature}_tmp_merged1_2.narrowPeak"
        _merge_with_summit_dist "${files[0]}" "${files[1]}" "$intermediateOutput" "$maxDist"
        
        # Merge the result with the third file
        finalOutput="${feature}_merged_final.narrowPeak"
        _merge_with_summit_dist "$intermediateOutput" "${files[2]}" "${2}/$finalOutput" "$maxDist"
        
        # Clean up the intermediate file
        rm "$intermediateOutput"

        # Output the result or move the files to a final destination as needed
        echo "Merged peaks for ${feature} saved as ${finalOutput}"
    done
}

_merge_with_reciprocal_overlap() {
    inputFile1="$1"
    inputFile2="$2"
    outputMergedFile="$3"

    # Perform reciprocal overlapping with at least 50% overlap in both files
    bedtools intersect -a "$inputFile1" -b "$inputFile2" -f 0.5 -r -wo \
    | cut -f 1-9 > "$outputMergedFile"
}

function _merge_epimap_heuristically_overlap () {
    # Define the maximum allowed distance between summits as an example
    peakdir="$1"
    mergeddir="$2"
    
    cd $peakdir
    # Loop through each feature
    for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        # store all files in an array
        files=($(ls $peakdir | grep ${feature}))
        
        # Check if exactly 3 files are present
        if [ "${#files[@]}" -ne 3 ]; then
            echo "Error: Expected 3 files for feature ${feature}, but found ${#files[@]}." >&2
            continue
        fi
        
        # Merge the first two files
        intermediateOutput="${feature}_tmp_merged1_2.narrowPeak"
        _merge_with_summit_dist "${files[0]}" "${files[1]}" "$intermediateOutput"
        
        # Merge the result with the third file
        finalOutput="${feature}_merged_final.narrowPeak"
        _merge_with_summit_dist "$intermediateOutput" "${files[2]}" "${2}/$finalOutput"
        
        # Clean up the intermediate file
        rm "$intermediateOutput"

        # Output the result or move the files to a final destination as needed
        echo "Merged peaks for ${feature} saved as ${finalOutput}"
    done
}

# =============================================================================
# Arguments:
#   $1 - directory to converted and lifted peaks
#   $2 - directory to save peak copies
#   $3 - full path to hg38 chrom sizes
#   $3 - working directory
# =============================================================================
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


# =============================================================================
# main function to perform processing in a given tissue!
#   $1 - directory to bigwigs
#   $2 - name of tissue
#   $3 - directory to liftOver
# =============================================================================
function main_func () {

    # set up cutoff array
    declare -A cutoffs
    # cutoffs provided by C. Boix
	cutoffs["DNase-seq"]=1.9
	cutoffs["H3K27ac"]=2.2
	cutoffs["H3K27me3"]=1.2
	cutoffs["H3K36me3"]=1.7
	cutoffs["H3K4me1"]=1.7
	cutoffs["H3K4me2"]=2
	cutoffs["H3K4me3"]=1.7
	cutoffs["H3K79me2"]=2.2
	cutoffs["H3K9ac"]=1.6
	cutoffs["H3K9me3"]=1.1
    # these following cutoffs default to 2
	cutoffs["ATAC-seq"]=2
	cutoffs["CTCF"]=2
	cutoffs["POLR2A"]=2
	cutoffs["RAD21"]=2
	cutoffs["SMC3"]=2

    # make directories if they don't exist
    for dir in merged peaks tmp crms crms_processing;
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
            histone=$(echo $(basename ${file}) | cut -d'_' -f3 | sed 's/\.bigWig//g')
            name=$(echo $(basename ${file}) | sed 's/\.bigWig//g')

            # get cutoffs for imputed tracks. Observed tracks get a cutoff of 5
            if [ $prefix == "impute" ]; then
                pval=${cutoffs[$histone]}
            else
                pval=5
            fi

            _bigWig_to_peaks \
                $3 \
                $1/$2 \
                $name \
                $pval
        fi
    done

    # merge epimap features across 3 samples
    _merge_epimap_heuristically \
        $1/$2/peaks \
        $1/$2/merged

    # liftover final peaks to hg38

    # # call crms via remap method
    # _get_crms_from_epimap \
    #     $1/$2/peaks \
    #     $1/$2/crms_processing \
    #     $4 \
    #     $1/$2/crms/

    # # copy crms to raw_files directory
    # cp \
    #     $1/$2/crms/consensuses.bed \
    #     $5/$2/consensuses.bed

    # copy merged peaks to raw_files directory
    cp \
        $1/$2/merged/* \
        $5/$2/

    # # temporarily delete old merged files
    # rm $5/$2/*_narrow_5*
    # rm $5/$2/*_narrow_4*
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