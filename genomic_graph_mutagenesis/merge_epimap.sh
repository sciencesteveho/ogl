# convert bigwig to bed
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
function _bigWig_to_peaks () {
    $1/bigWigToBedGraph ${2}/${3}.bigWig ${2}/tmp/${3}.bedGraph
    macs3 bdgpeakcall -i ${2}/tmp/${3}.bedGraph -o ${2}/tmp/${3}.bed
    # cleanup 
    tail -n +2 ${2}/tmp/${3}.bed > ${4}/${3}.bed 
}

# function to liftover 
# wget link/to/liftOvertool
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
function _liftover_19_to_38 () {
    $1/liftOver \
        $2/${3}.bed \
        $1/hg19ToHg38.over.chain.gz \
        $2/${3}._lifted_hg38.bed \
        $2/${3}.unlifted

    # cleanup
    rm $2/${3}.unlifted
}

# function to merge epimap peaks
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
function _merge_epimap_features () {
    for feature in ATAC-seq CTCF DNase-seq H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 POLR2A RAD21 SMC3;
    do
        bedops -m $1/*${feature}* | awk -v FS='\t' '{print $1, $2, $3, $4 }' > $2/${feature}_merged.bed
    done
}

# main function to perform processing in a given tissue!
function main () {
    mkdir $1/$2/tmp
    for file in $1/$2/*;
    do
        if [ -f $file ]; then
            name=$(echo $(basename ${file}) | sed 's/\.bigWig//g')
            _bigWig_to_peaks \
                /ocean/projects/bio210019p/stevesho/resources \
                $1/$2 \
                $name \
                $1/$2/peaks

            _liftover_19_to_38 \
                /ocean/projects/bio210019p/stevesho/resources \
                $1/$2/peaks \
                $name
        fi
    done

    if [ ! -d $1/$2/merged ]; then
        mkdir $1/$2/merged
    fi

    _merge_epimap_features \
        $1/$2/peaks \
        $1/$2/merged
}

# run main function! 
#   $1 - bigwig directory, one level before tissues
#   $2 - name of tissue
#   $3 - directory to liftover and bigwig conversion file
main \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/bigwigs \
    $1 \
    /ocean/projects/bio210019p/stevesho/resources \ 