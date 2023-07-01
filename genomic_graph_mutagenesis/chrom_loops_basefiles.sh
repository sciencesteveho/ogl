#!/bin/bash
#
# Scripts to process chromatin loops. Base loops come from Salameh et al., 2020 and are ensembled. Lower coverage tissue samples are ensembled with the top 15K pixels from deeploop, which require liftover from hg19 to hg38. Higher coverage samples are ensembled with calls from refHiC. Loops are not merged because graphs are not directed, so overlap is not recounted.
start=`date +%s`

# convert bigwig to bed
# Arguments:
#    - pseudo bool, true if histone, false if TF
#    - directory to bigWigToBedGraph
#    - bigwig tissue directory
#    - filename without extension
function _placeholder_function () {
    for number in {0..100};
    do
        echo number
    done
}

# run main_func function! 
#    - 
#    - 
#    - 
main_func \
     \
     \
    


end=`date +%s`
time=$((end-start))
echo "Finished! in time seconds."