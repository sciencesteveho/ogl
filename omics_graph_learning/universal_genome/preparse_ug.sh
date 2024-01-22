#!/bin/bash
#
# This is a description for your code to do stuff. It's a good idea to write
# something for the sake of reproducibility, ok. Don't give me that look, you're
# going to do it. Good habits don't build themselves (yet)!
SECONDS=0

# description for first function
# Arguments:
#    - 
#    - 
#    - 
function _histone_bigwig_to_peak () {
    pval=5
    $1/bigWigToBedGraph ${2}/${3}.bigWig ${2}/tmp/${3}.bedGraph

    for dir in tmp peaks;
    do
        if [ ! -d $2/$dir ]; then
            mkdir $2/$dir
        fi
    done

    macs2 bdgpeakcall \
        -i ${2}/tmp/${4}.bedGraph \
        -o ${2}/peaks/${4}.narrow.${pval}.bed \
        -c ${pval} \
        -l 73 \
        -g 100
}

_histone_bigwig_to_peak \
    /ocean/projects/bio210019p/stevesho/resources \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/universal_genome/bigwigs \
    $1  # name of file without .bigWig extension
    

bedtools shift -i <(awk -vOFS='\t' '{print $1,$2,$3}' TADMap_scaffold_hs.bed) -s 0 -g ../../../../resources/hg38.chrom.sizes.txt > tmp && mv tmp TADMap_scaffold_hs.bed 

# run main_func function! 
#    - 
#    - 
#    - 
main_func \
     \
     \
    
    

t=SECONDS
printf 'Time elapsed: %d days, %d minutes\n' "$(( t/86400 ))" "$(( t/60 - 1440*(t/86400) ))"