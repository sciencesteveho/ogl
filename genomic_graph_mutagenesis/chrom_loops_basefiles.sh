#!/bin/bash
#
# Scripts to process chromatin loops. Some of these initial steps are adjusted
# per sample, so be warned that parts are hard coded and not intended for public
# use. Base loops come from Salameh et al., 2020 and are ensembled. Lower
# coverage tissue samples are ensembled with the top pixels from deeploop,
# which require liftover from hg19 to hg38. Higher coverage samples are
# ensembled with calls from refHiC. Loops are not merged because graphs are not
# directed, so overlap is not recounted.
#
# The scripts are designed to take different number of n top loops from deeploop
# for concatenation to test which base configurations train the best.

# Put loop calls from deeploop and refhic in the /supp folder. Put peakachu
# calls in the base_dir. Combined loops will be put in loop_dir.
# loop_dir/
# ├── base_dir/
# │   ├── *put peakachu calls here*
# ├── supp/
# │   ├── *put deeploop + refhic calls here*
# └── tmp/

# setting up variables - folders were the loops are stored
SECONDS=0


# liftover deeploop files
# Arguments:
#   $1 - directory of deeploop files
#   $2 - filename without extension
#   $3 - tmp dir to process files
#   $4 - directory of liftover and liftover chain
function _liftover_deeploop_bedpe () {
    sed \
        -e 's/:/\t/g' \
        -e 's/-/\t/g' \
        $1/$2.txt \
        > $3/$2.bedpe
        
    awk -v OFS='\t' '{print $1,$2,$3,NR}' $3/$2.bedpe > $3/$2.bedpe_1
    awk -v OFS='\t' '{print $4,$5,$6,NR,$7,$8}' $3/$2.bedpe > $3/$2.bedpe_2

    for file in bedpe_1 bedpe_2;
    do
        $4/liftOver \
            $3/$2.${file} \
            $4/hg19ToHg38.over.chain.gz \
            $3/$2.${file}.hg38 \
            $3/$2.${file}.unmapped
        
        sort -k4,4 -o $3/$2.${file}.hg38 $3/$2.${file}.hg38
    done
}

# format loops and keep top N interactions
# Arguments:
#   $1 - directory of deeploop files
#   $2 - filename without extension
#   $3 - tmp dir to process files
#   $4 - number of top N loops to keep
function _format_deeploop_bedpe () {
    join \
        -j 4 \
        -o 1.1,1.2,1.3,2.1,2.2,2.3,2.4,2.5,2.6 \
        $3/$2.bedpe_1.hg38 \
        $3/$2.bedpe_2.hg38 \
        | sed 's/ /\t/g' \
        | sort -k8,8n \
        | tail -n $4 \
        | awk -v OFS='\t' '{print $1,$2,$3,$4,$5,$6}' \
        > $1/$2_$4_loops.bedpe.hg38
}

# cleanup files
# Arguments:
#   $2 - filename without extension
#   $3 - tmp dir to process files
function _cleanup_liftover () {
    # cleanup
    for file in $1.bedpe $1.bedpe_1 $1.bedpe_1.hg38 $1.bedpe_1.unmapped $1.bedpe_2 $1.bedpe_2.hg38 $1.bedpe_2.unmapped;
    do
        rm $2/$file
    done
}

# Arguments:
#   $1 - first loop file
#   $2 - second loop file
#   $3 - directory to store combined loops
#   $4 - name of tissue for combined loop file
function _combine_chr_loops () {
    cat \
        <(sort -k1,1 -k2,2n $1) \
        <(sort -k1,1 -k2,2n $2) \
        > $3/${4}_${5}.hg38.combined_loops
}

# process chromatin loops
# For low coverage, loops are lifted over from deeploop. We take the top 15K loop pixels from deeploop and add them to peakachu. For high coverage, we called loops from refhic. We combine any loops with 80% reciprocal overlap w/ peakachu and take the remaining union (we keep the refhic boundaries for the overlapping loops).
main_func () {
    # set vars
    loop_dir=/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops
    resource_dir=/ocean/projects/bio210019p/stevesho/resources
    base_dir=${loop_dir}/hg38_chromatin_loops_yue
    supp_dir=${loop_dir}/supp
    tmp_dir=${loop_dir}/tmp

    n_loops=(10000 20000 30000 40000 50000)

    declare -A loop_files
    loop_files["Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops"]="GSE167200_Hippocampus.top300K.bedpe.hg38"
    loop_files["Leung_2015.VentricleLeft.hg38.peakachu-merged.loops"]="GSE167200_LeftVentricle.top300K.bedpe.hg38"
    loop_files["Leung_2015.Liver.hg38.peakachu-merged.loops"]="GSE167200_Liver.top300K.bedpe.hg38"
    loop_files["Schmitt_2016.Lung.hg38.peakachu-merged.loops"]="GSE167200_Lung.top300K.bedpe.hg38"
    loop_files["Schmitt_2016.Pancreas.hg38.peakachu-merged.loops"]="GSE167200_Pancreas.top300K.bedpe.hg38"
    loop_files["Schmitt_2016.Psoas.hg38.peakachu-merged.loops"]="GSE167200_Psoas_Muscle.top300K.bedpe.hg38"
    loop_files["Schmitt_2016.Bowel_Small.hg38.peakachu-merged.loops"]="GSE167200_Small_Intenstine.top300K.bedpe.hg38"
    loop_files["peakachu.HeLa.-5kb-loops.0.99.bedpe"]="HeLa_loops.bedpe"
    loop_files["Rao_2014.HMEC.hg38.peakachu-merged.loops"]="HMEC_loops.bedpe"
    loop_files["Rao_2014.K562.hg38.peakachu-merged.loops"]="K562_loops.bedpe"
    loop_files["Rao_2014.NHEK.hg38.peakachu-merged.loops"]="NHEK_loops.bedpe" 

    low_cov=("GSE167200_Liver.top300K" "GSE167200_Hippocampus.top300K" "GSE167200_LeftVentricle.top300K" "GSE167200_Lung.top300K" "GSE167200_Pancreas.top300K" "GSE167200_Psoas_Muscle.top300K" "GSE167200_Small_Intenstine.top300K")

    # make directory for processing
    if [ ! -d ${loop_dir}/tmp ]; then
        mkdir ${loop_dir}/tmp
    fi

    # liftover low coverage loops
    for tissue in ${low_cov[@]};
    do
        _liftover_deeploop_bedpe \
            ${supp_dir} \
            ${tissue} \
            ${tmp_dir} \
            ${resource_dir} \
            ${num_loops}

        for num_loops in ${n_loops[@]};
        do
            _format_deeploop_bedpe \
                ${supp_dir} \
                ${tissue} \
                ${tmp_dir} \
                ${num_loops}
        done

        _cleanup_liftover \
            ${tissue} \
            ${tmp_dir} 
    done

    for key in ${!loop_files[@]};
    do
        tissue=$(echo ${key} | cut -d'.' -f2)
        prefix=$(echo ${loop_files[${key}]} | cut -d'.' -f1)
        for num_loops in ${n_loops[@]};
        do
            _combine_chr_loops \
                ${base_dir}/${key} \
                ${supp_dir}/${prefix}.top300K_${num_loops}_loops.bedpe.hg38 \
                ${loop_dir} \
                ${tissue} \
                ${num_loops}
        done
    done

    rm -r ${tmp_dir}
}


# run main_func function! 
#    - 
#    - 
#    - 
main_func 


end=`date +%s`
time=$((end-start))
echo "Finished! in time seconds."


# Extra code below are QOL scripts to help with processing. Be warned, they
# are hard coded and not intended for public use.

# move files from processing dir to dir for combining
# hic_dir=/ocean/projects/bio210019p/stevesho/hic
# for tissue in HeLa HMEC K562 NHEK;
# do
#     mv \
#         ${hic_dir}/${tissue}/${tissue}_loops.bedpe \
#         ${loop_dir}/supp
# done

# download deeploop files
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FHippocampus%2Etop300K%2Etxt%2Egz
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FLeftVentricle%2Etop300K%2Etxt%2Egz
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FLiver%2Etop300K%2Etxt%2Egz
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FLung%2Etop300K%2Etxt%2Egz
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FPancreas%2Etop300K%2Etxt%2Egz
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FPsoas%5FMuscle%2Etop300K%2Etxt%2Egz
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167200/suppl/GSE167200%5FSmall%5FIntenstine%2Etop300K%2Etxt%2Egz