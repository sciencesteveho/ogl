#!/bin/bash
# Scripts to filter bedfiles of common attributes. Datatypes here are not
# tissue-specific and are stored in a common directory, other than the
# tissue-specific mirDIP files, which are pre-parsed to individual tissues.

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

# function to overlap SCREEN regulatory regions with EpiMap regulatory regions.
# Done for both enhancers and promoters.
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
#   $4 - 
#   $5 - 
function _overlap_regulatory_regions () {
    _liftover_19_to_38 \
        $1 \
        $2 \
        $3

    bedtools intersect \
        -a $2/$4 \
        -b $2/$3._lifted_hg38.bed \
        -wa \
        -wb \
        | sort -k1,1 -k2,2n \
        | cut -f1,2,3 \
        | uniq \
        | awk -v OFS='\t' '{print $1,$2,$3,"enhancer"}' \
        > $2/epimap_screen_${5}_overlap.bed

    rm ${3}.unlifted
}

# function to overlap SCREEN regulatory regions with EpiMap dyadic regions
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
#   $4 - 
#   $5 - 
function _overlap_dyadic_elements () {
    _liftover_19_to_38 \
        $1 \
        $2 \
        $3
    
    bedtools intersect \
        -a $2/$3._lifted_hg38.bed \
        -b $2/$4 $2/$5 \
        -wa \
        | sort -k1,1 -k2,2n \
        | cut -f1,2,3 \
        | uniq \
        | awk -v OFS='\t' '{print $1,$2,$3,"dyadic"}' \
        > $2/epimap_screen_${6}_overlap.bed

    rm ${3}.unlifted
}

# enhancers
_overlap_regulatory_regions \
    /ocean/projects/bio210019p/stevesho/resources \
    /ocean/projects/bio210019p/stevesho/data/bedfile_preparse/regulatory_elements \
    ENH_masterlist_locations \
    GRCh38-ELS.bed \
    enhancer

# promoters
_overlap_regulatory_regions \
    /ocean/projects/bio210019p/stevesho/resources \
    /ocean/projects/bio210019p/stevesho/data/bedfile_preparse/regulatory_elements \
    PROM_masterlist_locations \
    GRCh38-PLS.bed \
    promoter

# dyadic
_overlap_dyadic_elements \
    /ocean/projects/bio210019p/stevesho/resources \
    /ocean/projects/bio210019p/stevesho/data/bedfile_preparse/regulatory_elements \
    DYADIC_masterlist_locations \
    GRCh38-ELS.bed \
    GRCh38-PLS.bed \
    dyadic

# Convert gencode v26 GTF to bed, remove micro RNA genes and only keep canonical
# "gene" entries. Additionally, make a lookup table top convert from gencode to
# genesymbol.
# wget https://storage.googleapis.com/gtex_analysis_v8/reference/gencode.v26.GRCh38.genes.gtf 
# Arguments:
function _gencode_bed () {
    gtf2bed <  $1/$2 | awk '$8 == "gene"' | grep -v miR > $3/local/gencode_v26_genes_only_with_GTEx_targets.bed
    cut -f4,10 $3/local/gencode_v26_genes_only_with_GTEx_targets.bed | sed 's/;/\t/g' | cut -f1,5 | sed -e 's/ gene_name //g' -e 's/\"//g' > $3/interaction/gencode_to_genesymbol_lookup_table.txt
}

# save alternative names for gencode entries. This is for parsing node features
# downstream of graph construction
# Arguments:
function _gencode_nodetype_featsaver () {
    cat <(awk -v OFS='\t' '{print $1, $2, $3, $4}' $1/$2) \
        <(awk -v OFS='\t' '{print $1, $2, $3, $4"_protein"}' $1/$2) \
        <(awk -v OFS='\t' '{print $1, $2, $3, $4"_tf"}' $1/$2) \
        > $1/gencode_v26_node_attr.bed 
}

# Parse mirDIP database v5.2 in tissue-specific files of active miRNAs. Requires
# "mirDIP_ZA_TISSUE_MIRS.txt" from "All data / TSV" and "final_data.gff",
# "All data / GFF". miRNA targets are downloaded from miRTarBase v9.0 and
# filtered for only interactions in homo sapiens. Interactions with a blank
# "support type" were removed.
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
function _active_mirna () {
    cd $1

    if [[ ! -d tissues ]]; then
        mkdir tissues
    fi

    awk -v OFS="\t" '
        NR == FNR {a[$1][$2] = $3; next}
        {print $3,$4,$5,$1,$2, a[$1][$2]}
    ' \
        <(sed -e 's/"//g' -e 's/,/\t/g'  $1/$2 | cut -f1,5,6) \
        <(sed 's/\r$//' $3 | tail -n +2 | awk -v OFS="\t" '{print $1,$12,$4,$5,$6'}) \
        | sort -k1,1 -k2,2n \
        | awk -v OFS="\t" '($6 == 1) { print>"tissues/"$5}'
}

# DEPRECATED 
# Encode SCREEN enhancer cCREs downloaded from FENRIR. Enhancers are lifted over
# to hg38 from hg19 and an index for each enhancer is kept for easier
# identification
#   $1 - 
#   $2 - 
#   $3 - 
#   $4 - 
# function _enhancer_lift () {
#     awk -v OFS=FS='\t' '{print $2, $3, $4, "enhancer_"$1}' $1/$2 | tail -n +2 > enhancer_regions_unlifted.txt
#     # awk -v OFS=FS='\t' '{print $2":"$3"-"$4, "enhancer_"$1}' $1/$2 | tail -n +2 > enhancer_indexes_unlifted.txt

#     $3/liftOver \
#         enhancerregions.txt \
#         $3/hg19ToHg38.over.chain.gz \
#         $4/enhancer_regions_lifted.txt \
#         enhancers_unlifted.txt

#     awk -v OFS=FS='\t' '{print $1":"$2"-"$3, $4}' enhancer_regions_lifted.txt > $4/enhancer_indexes.txt
# }

# poly-(A) target sites for hg38 are downloaded from PolyASite, homo sapiens
# v2.0, release 21/04/2020
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
function _poly_a () {
    awk -v OFS='\t' '{print "chr"$1,$2,$3,"polya_"$4_$10}' $1/$2 \
        > $3/polyasites_filtered_hg38.bed
}

# RNA Binding protein sites were downloaded from POSTAR 3. We first merged
# adjacent sites to create RBP binding site clusters. We keep sites present in
# at least 20% of the sampes (8)
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
function _rbp_sites () {
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
# 18/08/2021. Gene symbols are added to refTSS names
# Arguments:
#   $1 - 
#   $2 - 
function _tss () {
    cut -f1,8 $1 
    awk -v OFS='\t' '{print $1,$2,$3,"tss_"$4}' $1 \
        > $2/tss_parsed_hg38.bed
}

# micro RNA (miRNA) target sites for hg19 are downloaded from TargetScanHuman
# 8.0, release 09/2021 target sites are lifted over to hg38
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
#   $4 - 
function _target_scan () {
    _liftover_19_to_38 \
        $1 \
        $2 \
        $3

    awk -v OFS='\t' '{print $1,$2,$3,"miRNAtarget_"$4}' Predicted_Target_Locations.default_predictions.hg19._lifted_hg38.bed \
        | sed 's/::/__/g' \
        > $4/mirnatargets_parsed_hg38.bed
}


# ENCODE SCREEN candidate promoters from SCREEN registry of cCREs V3
# Arguments:
#   $1 - path to promotoer file
#   $2 - directory to save parsed files
function _screen_promoters () {
    awk -v OFS='\t' '{print $1,$2,$3,"promoter"}' $1 \
        > $2/promoters_parsed_hg38.bed
}

# ENCODE SCREEN CTCF-only cCREs from SCREEN V3
# Arguments:
#   $1 - path to ctcf file
#   $2 - cCRE file
#   $3 - directory to save parsed files
function _screen_ctcf () {
    grep CTCF-only $1/$2 \
        | awk -v OFS='\t' '{print $1,$2,$3,"ctcfccre"}' $2 \
        > $3/ctcfccre_parsed_hg38.bed
}

# Genomic variant hotspots from Long & Xue, Human Genomics, 2021. First, expand
# hotspot clusers to their individual mutation type. Then, file is split into
# CNVs, indels, and SNPs.
# Arguments:
#   $1 - name of hotspot file
#   $1 - name of hotspot file
#   $2 - path to liftOver and liftover chain
#   $3 - directory to save parsed files
function _var_hotspots () {
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

# Replication hotspots from Long & Xue, Human Genomics, 2021. Each phase is
# split into a separate file for node attributes.
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
#   $4 - 
function _replication_hotspots () {
    awk -v string='Replication' '$4 ~ string' $1/$2 \
        | awk -v OFS='\t' '{print $1, $2, $3, "rep"$6}' \
        | sed 's/^/chr/g' \
        | tr '[:upper:]' '[:lower:]' \
        > rep_formatted_hg18.bed 

    $3/liftOver \
        rep_formatted_hg18.bed  \
        $3/hg19ToHg38.over.chain.gz \
        rep_lifted_hg38.bed \
        rep_unlifted

    awk -v OFS="\t" '{print>$4}' rep_lifted_hg38.bed

    for phase in repg1b repg2 reps1 reps2 reps3 reps4;
    do
        mv ${phase} $4/${phase}_hg38.bed
    done
}

# Average recombination rate from deCODE - Halldorsson et al, Science, 2019.
# BigWig is converted to Wig then to .bed via BEDOPS.
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
#   $4 - 
function _recombination () {
    $1/bigWigToWig ${2}/${3}.bw ${2}/${3}.wig
    wig2bed < ${2}/${3}.wig > tmp
    awk -v OFS="\t" '{print $1, $2, $3, $5, $4}' tmp > ${4}/recombination_hg38.bed
}

# TF-interactions from TFMarker. The file was not properly delimited so some
# columns were cleaned up in excel first. We filter and only keep TF
# interactions present in normal cells (not cancer cells).
function _tf_marker () {
    sed 's/ /_/g' $1 |  awk '$5 == "Normal_cell"' > $2/tf_marker.txt
}

# Simple function to clean up intermediate files
function _cleanup () {
    for file in hotspots_expanded_hg18.bed hotspots_lifted_hg38.bed hotspots_unlifted miRNAtargets_lifted_hg38.bed mirRNA_unlifted enhancer_regions_unlifted.txt rep_formatted_hg18.bed rep_lifted_hg38.bed rep_unlifted recombAvg.wig; do
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
    _gencode_bed \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse \
        gencode.v26.GRCh38.genes.gtf \
        /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data

    _gencode_nodetype_featsaver \
        /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local \
        gencode_v26_genes_only_with_GTEx_targets.bed

    _active_mirna \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse/mirDIP \
        mirDIP_ZA_TISSUE_MIRS.txt \
        final_data.gff
    
    _enhancer_lift \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse \
        Enhancer_regions.txt \
        /ocean/projects/bio210019p/stevesho/resources \
        /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction

    _poly_a \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        atlas.clusters.2.0.GRCh38.96.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _rbp_sites \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        human.txt \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _tss \
        refTSS_v3.3_human_coordinate.hg38.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _target_scan \
        /ocean/projects/bio210019p/stevesho/resources \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        Predicted_Target_Locations.default_predictions.hg19.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _screen_promoters \
        GRCh38-PLS.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _screen_ctcf \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse \
        GRCh38-CTCF.bed \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _var_hotspots \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        '40246_2021_318_MOESM3_ESM.txt' \
        /ocean/projects/bio210019p/stevesho/resources \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _replication_hotspots \
        '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse' \
        '40246_2021_318_MOESM3_ESM.txt' \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/' \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local' 

    _recombination \
        /ocean/projects/bio210019p/stevesho/resources \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse \
        recombAvg \
        '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local'

    _tf_marker \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse/tf_markers_col_filtered.txt \
        /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction

    _cleanup
}

# run main function! 
main \
    var1 \
    var2 \
    var3 \
    var4 