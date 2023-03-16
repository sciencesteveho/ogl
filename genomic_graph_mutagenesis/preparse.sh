#!/bin/bash

# Scripts to filter bedfiles of common attributes. Datatypes here are not
# tissue-specific and are stored in a common directory

# fix dir variables
# convert gencode v26 GTF to bed
gencode_bed () {
    gtf2bed <  gencode.v26.GRCh38.genes.gtf | awk '$8 == "gene"' > gencode_v26_genes_only_with_GTEx_targets.bed
}

# fix dir variables
# get gene symbol lookup table for gencode IDs 
genesymbol_lookup () {
    cut -f4,10 gencode_v26_genes_only_with_GTEx_targets.bed | sed 's/;/\t/g' | cut -f1,5 | sed -e 's/ gene_name //g' -e 's/\"//g' > gencode_to_genesymbol_lookup_table.txt
}

# Encode SCREEN enhancer cCREs downloaded from FENRIR. Enhancers are lifted over
# to hg38 from hg19 and an index for each enhancer is kept for easier
# identification
function enhancer_lift () {
    awk -v OFS=FS='\t' '{print $2, $3, $4, "enhancer_"$1}' $1/$2 | tail -n +2 > enhancer_regions_unlifted.txt
    # awk -v OFS=FS='\t' '{print $2":"$3"-"$4, "enhancer_"$1}' $1/$2 | tail -n +2 > enhancer_indexes_unlifted.txt

    $3/liftOver \
        enhancerregions.txt \
        $3/hg19ToHg38.over.chain.gz \
        $4/enhancer_regions_lifted.txt \
        enhancers_unlifted.txt

    awk -v OFS=FS='\t' '{print $1":"$2"-"$3, $4}' enhancer_regions_lifted.txt > $4/enhancer_indexes.txt
}

# poly-(A) target sites for hg38 are downloaded from PolyASite, homo sapiens
# v2.0, release 21/04/2020
function poly_a () {
    awk -v OFS='\t' '{print "chr"$1,$2,$3,"polya_"$4_$10}' $1/$2 \
        > $3/polyasites_filtered_hg38.bed
}

# RNA Binding protein sites were downloaded from POSTAR 3. We first merged
# adjacent sites to create RBP binding site clusters. We keep sites present in
# at least 20% of the sampes (8)
#   $1 - 
#   $2 - 
#   $3 - 
function rbp_sites () {
    awk '$6 != "RBP_occupancy"' $1/$2 | \
        bedtools merge \
        -d 50 \
        -i - \
        -c 6,6,8,8 \
        -o distinct,count_distinct,count_distinct,distinct \
        | awk '($6 >= 8)' \
        | cut -f1,2,3,4 \
        | sed 's/,/_/g' \
        | awk -v OFS='\t' '{print $1, $2, $3, "rnab_"$4}' \
        > $3/rbpbindingsites_parsed_hg38.bed
}

# transcription start sites for hg38 are downloaded from refTSS v3.3, release
# 18/08/2021
# Arguments:
#   $1 - 
#   $2 - 
function tss () {
    awk -v OFS='\t' '{print $1,$2,$3,"tss_"$4}' $1 \
        > $2/tss_parsed_hg38.bed
}

# micro RNA (miRNA) target sites for hg19 are downloaded from TargetScanHuman
# 8.0, release 09/2021 target sites are lifted over to hg38
# Arguments:
#   $1 - 
#   $2 - 
function target_scan () {
    local shared_dir='/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data'
    $1/liftOver \
        $2/$3 \
        $1/hg19ToHg38.over.chain.gz \
        miRNAtargets_lifted_hg38.bed \
        mirRNA_unlifted

    awk -v OFS='\t' '{print $1,$2,$3,"miRNAtarget_"$4}' miRNAtargets_lifted_hg38.bed \
        | sed 's/::/__/g' \
        > $4/mirnatargets_parsed_hg38.bed
}

# ENCODE SCREEN candidate promoters from SCREEN registry of cCREs V3
# Arguments:
#   $1 - path to promotoer file
#   $2 - directory to save parsed files
function screen_promoters () {
    awk -v OFS='\t' '{print $1,$2,$3,"promoter"}' $1 \
        > $2/promoters_parsed_hg38.bed
}

# Genomic variant hotspots from Long & Xue, Human Genomics, 2021 First, expand
# hotspot clusers to their individual mutation type. Then, file is split into
# CNVs, indels, and SNPs.
# Arguments:
#   $1 - name of hotspot file
#   $1 - name of hotspot file
#   $2 - path to liftOver and liftover chain
#   $3 - directory to save parsed files
function var_hotspots () {
    tail -n +2 $1/$2 \
        | awk '$4 == "Cluster"' \
        | sed 's/(/\t/g' \
        | cut -f1,2,3,6 \
        | sed -e 's/; /,/g' -e 's/)//g' \
        | bedtools expand -c 4 \
        | cat - <(awk -v OFS='\t' '$4 == "GV hotspot"' $1/$2 | cut -f1,2,3,5) \
        | sort -k 1,1 -k2,2n > hotspots_expanded_hg18.bed

    $3/liftOver \
        hotspots_expanded_hg18.bed \
        $3/hg19ToHg38.over.chain.gz \
        hotspots_lifted_hg38.bed \
        hotspots_unlifted

    for variant in CNV SID SNP; do
        local varlower=$(echo $variant | tr '[:upper:]' '[:lower:]') # lowercasing
        awk -v variant=$variant '$4 ~ variant' hotspots_lifted_hg38.bed > $4/${varlower}_hg38.bed
        if [[ $variant == SID ]]; then
            mv $4/${varlower}_hg38.bed $4/indels_hg38.bed
        fi
    done
}

# simple function to clean up intermediate files
function cleanup () {
    for file in hotspots_expanded_hg18.bed hotspots_lifted_hg38.bed hotspots_unlifted miRNAtargets_lifted_hg38.bed mirRNA_unlifted enhancer_regions_unlifted.txt; do
        rm $file
    done
}

# Main function to run all preparsing
# Arguments:
#   $1 - 
#   $1 - 
#   $2 - 
#   $3 - 
function main() {
    enhancer_lift \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse \
        Enhancer_regions.txt \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data' \
        /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction

    poly_a \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        atlas.clusters.2.0.GRCh38.96.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    rbp_sites \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        human.txt \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    tss \
        refTSS_v3.3_human_coordinate.hg38.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    target_scan \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data' \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        Predicted_Target_Locations.default_predictions.hg19.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    screen_promoters \
        GRCh38-PLS.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    var_hotspots \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        '40246_2021_318_MOESM3_ESM.txt' \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data' \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    cleanup
}

# run main function! 
main \
    var1 \
    var2 \
    var3 \
    var4 