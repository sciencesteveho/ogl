#!/bin/bash
#
# Scripts to process chromatin loops. Base loops come from Salameh et al., 2020
# and are ensembled. Lower coverage tissue samples are ensembled with the top
# 15K pixels from deeploop, which require liftover from hg19 to hg38. Higher
# coverage samples are ensembled with calls from refHiC. Loops are not merged
# because graphs are not directed, so overlap is not recounted.

# setting up variables - folders were the loops are stored
start=`date +%s`
loop_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops"
base_dir=${loop_dir}/base
supp_dir=${loop_dir}/supp

if [ ! -d ${loop_dir}/tmp ]; then
    mkdir ${loop_dir}/tmp
fi

# extra function to liftover deeploop bedpe files
# only keep top 5K interactions
# Arguments:
#   $1 - filename
#   $2 - directory to process files
#   $3 - directory of liftover and liftover chain
function _liftover_deeploop_bedpe () {
    sed \
        -e 's/:/\t/g' \
        -e 's/-/\t/g' \
        $1.txt \
        > $2/$1.bedpe
        
    awk -v OFS='\t' '{print $1,$2,$3,NR}' $2/$1.bedpe > $2/$1.bedpe_1
    awk -v OFS='\t' '{print $4,$5,$6,NR,$7,$8}' $2/$1.bedpe > $2/$1.bedpe_2

    for file in bedpe_1 bedpe_2;
    do
        $3/liftOver \
            $2/$1.${file} \
            $3/hg19ToHg38.over.chain.gz \
            $2/$1.${file}.hg38 \
            $2/$1.${file}.unmapped
        
        sort -k4,4 -o $2/$1.${file}.hg38 $2/$1.${file}.hg38
    done

    join \
        -j 4 \
        -o 1.1,1.2,1.3,2.1,2.2,2.3,2.4,2.5,2.6 \
        $2/$1.bedpe_1.hg38 \
        $2/$1.bedpe_2.hg38 \
        | sed 's/ /\t/g' \
        | sort -k8,8n \
        | tail -n 5000 \
        > $2/$1.bedpe.hg38
    
    for file in $1.bedpe $1.bedpe_1 $1.bedpe_1.hg38 $1.bedpe_1.unmapped $1.bedpe_2 $1.bedpe_2.hg38 $1.bedpe_2.unmapped;
    do
        rm $2/$file
    done
}


function _combine_chromatin_loops_lowcov () {

}

function _combine_chromatin_loops_highcov () {

}

# process chromatin loops
# For low coverage, loops are lifted over from deeploop. We take the top 10K loop pixels from deeploop and add them to peakachu. For high coverage, we called loops from refhic. We combine any loops with 80% reciprocal overlap w/ peakachu and take the remaining union (we keep the refhic boundaries for the overlapping loops).
for tissue in GSE167200_Liver.top300K GSE167200_Hippocampus.top300K GSE167200_LeftVentricle.top300K GSE167200_Lung.top300K GSE167200_Pancreas.top300K GSE167200_Psoas_Muscle.top300K GSE167200_Small_Intenstine.top300K;
do
    _liftover_deeploop_bedpe \
        $tissue \
        liftover_processing \
        /ocean/projects/bio210019p/stevesho/resources
done

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