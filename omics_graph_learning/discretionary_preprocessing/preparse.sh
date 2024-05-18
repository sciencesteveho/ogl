#!/bin/bash
# Scripts to filter bedfiles of common attributes. Datatypes here are not
# tissue-specific and are stored in a common directory, other than the
# tissue-specific mirDIP files, which are pre-parsed to individual tissues.


# =============================================================================
# Setting up variables to track time
# =============================================================================
SECONDS=0

function convertsecs() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d\n" $hours $minutes $seconds
}


# =============================================================================
# Code was used to sum all overlaps together, only keep unique entries based on chr/start/end
# =============================================================================
cat *overlap* | sort  -k1,1 -k2,2n -k3,3n -u > concatenated_overlapped_elements.bed


# =============================================================================
# Utility function to perform reference liftover
# To download liftover, use `wget http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver`
# To download the liftover chain, use `wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz`
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
# =============================================================================
function _liftover_19_to_38 () {
    local liftover_dir="$1"
    local output_dir="$2"
    local file_name="$3"

    ${liftover_dir}/liftOver \
        ${output_dir}/${file_name}.bed \
        ${liftover_dir}/hg19ToHg38.over.chain.gz \
        ${output_dir}/${file_name}._lifted_hg38.bed \
        ${output_dir}/${file_name}.unlifted

    # cleanup
    rm ${output_dir}/${file_name}.unlifted
}


# =============================================================================
# Function to overlap SCREEN regulatory regions with EpiMap regulatory regions.
# Done for both enhancers and promoters.
# =============================================================================
function _overlap_regulatory_regions () {
    local working_dir=$1  # path/to/your working directory
    local bedfiles_dir=$2  # path/to/where your bedfiles are stored
    local epimap_masterlist_file=$3  # name of epimap masterlist file
    local encode_regulatory_element_file=$4  # name of encode regulatory element file
    local regulatory_element_naming=$5  # naming convention for the regulatory element

    _liftover_19_to_38 \
        ${working_dir} \
        ${bedfiles_dir} \
        ${epimap_masterlist_file}

    bedtools intersect \
        -a "${bedfiles_dir}/${encode_regulatory_element_file}" \
        -b "${bedfiles_di}r/${epimap_masterlist_file}_lifted_hg38.bed" \
        -wa \
        -wb \
        | sort -k1,1 -k2,2n \
        | cut -f1,2,3,6 \
        | cut -f1 -d',' \
        | uniq \
        | awk -v OFS='\t' '{print $1,$2,$3,$4}' \
        > "${bedfiles_dir}/${regulatory_element_naming}s_epimap_screen_overlap.bed"

    if [ -f "${bedfiles_dir}/${epimap_masterlist_file}.unlifted" ]; then
        rm "${bedfiles_dir}/${epimap_masterlist_file}.unlifted"
    fi
}


# =============================================================================
# Function to overlap SCREEN regulatory regions with EpiMap dyadic regions
# Arguments:
#   $1 - path/to/your working directory
#   $2 - path/to/where your bedfiles are stored
#   $3 - name of epimap masterlist file
#   $4 - name of encode regulatory element file
#   $5 - naming convention for the regulatory element
# =============================================================================
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
        > $2/${6}_epimap_screen_overlap.bed

    rm ${3}.unlifted
}

function _make_ref_for_regulatory_elements () {
    cat \
        $1/dyadic_epimap_screen_overlap.bed \
        $1/enhancers_epimap_screen_overlap.bed \
        $1/promoters_epimap_screen_overlap.bed \
        | awk -vOFS='\t' '{print $1,$2,$3,$1"_"$2"_"$4}' \
        | sort -k1,1 -k2,2n \
        > $3/regulatory_elements_node_attr.bed
    
    tail -n +2 $2/SE_package_hg38.bed \
        | awk -vOFS='\t' '{print $3,$4,$5,$3"_"$4"_superenhancer"}' \
        > $3/se_node_attr.bed
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

# make refs for adding feats to base nodes
_make_ref_for_regulatory_elements \
    /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local \
    /ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/universalgenome \
    /ocean/projects/bio210019p/stevesho/resources


# =============================================================================
# Convert gencode v26 GTF to bed, remove micro RNA genes and only keep canonical
# "gene" entries. Additionally, make a lookup table to convert from gencode to
# genesymbol.
# wget https://storage.googleapis.com/gtex_analysis_v8/reference/gencode.v26.GRCh38.genes.gtf 
# Arguments:
# =============================================================================
function _gencode_bed () {
    gtf2bed <  $1/$2 | awk '$8 == "gene"' | grep -v miR > $3/local/gencode_v26_genes_only_with_GTEx_targets.bed
    cut -f4,10 $3/local/gencode_v26_genes_only_with_GTEx_targets.bed | sed 's/;/\t/g' | cut -f1,5 | sed -e 's/ gene_name //g' -e 's/\"//g' > $3/interaction/gencode_to_genesymbol_lookup_table.txt
}


# =============================================================================
# save alternative names for gencode entries. This is for parsing node features
# downstream of graph construction
# Arguments:
function _gencode_nodetype_featsaver () {
    cat <(awk -v OFS='\t' '{print $1, $2, $3, $4}' $1/$2) \
        <(awk -v OFS='\t' '{print $1, $2, $3, $4"_protein"}' $1/$2) \
        <(awk -v OFS='\t' '{print $1, $2, $3, $4"_tf"}' $1/$2) \
        > $1/gencode_v26_node_attr.bed 
}


# =============================================================================
# Parse mirDIP database v5.2 in tissue-specific files of active miRNAs. Requires
# "mirDIP_ZA_TISSUE_MIRS.txt" from "All data / TSV" and "final_data.gff",
# "All data / GFF". miRNA targets are downloaded from miRTarBase v9.0 and
# filtered for only interactions in homo sapiens. Interactions with a blank
# "support type" were removed.
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
# =============================================================================
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


# =============================================================================
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
# =============================================================================


# =============================================================================
# poly-(A) target sites for hg38 are downloaded from PolyASite, homo sapiens
# v2.0, release 21/04/2020
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
# =============================================================================
function _poly_a () {
    awk -v OFS='\t' '{print "chr"$1,$2,$3,"polya_"$4_$10}' $1/$2 \
        > $3/polyasites_filtered_hg38.bed
}


# =============================================================================
# RNA Binding protein sites were downloaded from POSTAR 3. We first merged
# adjacent sites to create RBP binding site clusters. We keep sites present in
# at least 20% of the samples (8)
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
# The second function creates a list of RBPs and intersects their binding sites
# with gene bodies. It then removes any RBP --> Gene interactions that are
# singletons.
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
# =============================================================================
function _rbp_site_clusters () {
    awk '$6 != "RBP_occupancy"' $1/$2 \
        | bedtools merge \
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

function _rbp_binding_sites () {
    local file=$1
    local gencode=$2

    awk '$6 != "RBP_occupancy"' ${file} \
        | cut -f1,2,3,6,8 \
        | bedtools intersect \
        -a - \
        -b ${gencode} \
        -wa \
        -wb \ 
        | cut -f4,5,9 \
        | sort -u \
        |  awk 'BEGIN{FS=OFS="\t"} {array[$1 OFS $3] = array[$1 OFS $3] ? array[$1 OFS $3] "," $2 : $2} END{for (i in array) print i, array[i]}' \
        | grep "," \
        > rbp_gene_binding_sites.bed

        lookup=/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/shared_data/interaction/gencode_to_genesymbol_lookup_table.txt
        rows=/ocean/projects/bio210019p/stevesho/data/data_preparse/rbp_gene_binding_sites.bed

        awk 'BEGIN { FS=OFS="\t" } NR==FNR { lookup[$2]=$1; next } $1 in lookup { $1=lookup[$1] } 1' ${lookup} ${rows} \
            | grep -e '^ENSG' \
            > rbp_gene_binding_sites_genesymbol.bed
}


# =============================================================================
# transcription start sites for hg38 are downloaded from refTSS v3.3, release
# 18/08/2021. First, we create an associate array to store the tss --> gene
# relationships and then we use them to parse a tss file with each gene symbol
# annotated. Lastly, we convert the gene symbols to gencode IDs and only keep
# genes w/ gencode IDs.
# Arguments:
#   $1 - 
#   $2 - 
# =============================================================================
function _tss () {
    # create associative array
    declare -A tss_genes
    while read -r line; do
        key=$(echo "$line" | cut -f1 -d' ')
        value=$(echo "$line" | cut -f8 -d' ')
        tss_genes["$key"]="$value"
    done < "$1"

    # convert associative array to string
    tss_string=""
    for key in "${!tss_genes[@]}"; do
        tss_string+="$key=${tss_genes[$key]} "
    done

    # parse tss file and add array values if present
    awk -v OFS='\t' -v tss_genes="$tss_string" '
        BEGIN {
            split(tss_genes, genes, " ")
            for (i in genes) {
                split(genes[i], gene_pair, "=")
                gene = gene_pair[1]
                value = gene_pair[2]
                tss_array[gene] = value
            }
        }
        {
            key = "tss_" $4
            if (key in tss_array) {
                print $1, $2, $3, key, tss_array[$4]
            } else {
                print $1, $2, $3, key, "NA"
            }
        }
    ' "$1" > "$2/tss_parsed_hg38.bed"
}

_tss \
    '/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/reftss/refTSS_v4.1_human_hg38_annotation.txt' \
    /ocean/projects/bio210019p/stevesho/data/bedfile_preparse/reftss


# =============================================================================
# micro RNA (miRNA) target sites for hg19 are downloaded from TargetScanHuman
# 8.0, release 09/2021 target sites are lifted over to hg38
# Arguments:
#   $1 - 
#   $2 - 
#   $3 - 
#   $4 - 
# =============================================================================
# function _target_scan () {
#     _liftover_19_to_38 \
#         $1 \
#         $2 \
#         $3

#     awk -v OFS='\t' '{print $1,$2,$3,"miRNAtarget_"$4}' Predicted_Target_Locations.default_predictions.hg19._lifted_hg38.bed \
#         | sed 's/::/__/g' \
#         > $4/mirnatargets_parsed_hg38.bed
# }


# =============================================================================
# MicroRNA targets
# We download the entire miRNA catalogue from miRTarBase (version 9.0) and
# filter the miRNAs to only keep homo sapiens relevant miRNAs with functional
# evidence (remove Non-functional MTI).
# Arguments:
#   $1 - 
# =============================================================================
function _mirtarbase_targets () {
    local file=$1

    {
        echo -e "miRNA\ttarget_gene"
        cut -f2,4,6,8 -d',' "${file}" \
            | sed 's/,/\t/g' \
            | awk -vFS='\t' '$3 == "Homo sapiens" && $4 ~ /Functional/' \
            | cut -f1,2 \
            | sort -u
    } > mirtargets_filtered.txt
}

_mirtarbase_targets miRTarBase_MTI.csv


# =============================================================================
# MicroRNA coordinates
# We download coordinates of human miRNAs from miRBase Release 22.1:
# https://www.mirbase.org/download/hsa.gff3
# For the miRBase catalog, we remove unceccessary information and only keep the coordinates and name (casefolded). Because there are some repeat entrys (primary transcript vs gene body), we collapse any redundant mirnas by keeping the larger coordinates. Of not, 66 miRNAs had multiple annotations on disparate chromosomes. We removed these 66 miRNA from our analysis.
# For each ENCODE miRNA dataset, gencode entries are adjusted for only their base name then converted to miRNA aliases via gprofiler. 
# https://biit.cs.ut.ee/gprofiler/convert
# Arguments:
#   $1 - 
# =============================================================================
function _mirbase_mirnas () {
    local file=$1

    grep -v "^#" ${file} \
        | cut -f1,4,5,9 \
        | awk 'BEGIN{FS=OFS="\t"} {split($4, a, ";"); for (i in a) if (a[i] ~ /^Name=/) {split(a[i], b, "="); $4 = tolower(b[2])}} 1' \
        | sort -k4,4 -k1,1 -k2,2n \
        | bedtools groupby -g 4 -c 1,2,3 -o distinct,min,max \
        | awk 'BEGIN{OFS="\t"} {print $2, $3, $4, $1}' \
        | sort -k1,1 -k2,2n \
        | grep -v "," \
        > mirbase_coordinates_hg38.bed
}

_mirbase_mirnas hsa.gff3

function _biomart_mirna_coordinates () {
    local file=$1

    awk 'BEGIN{FS=OFS="\t"} {print "chr"$6, $3, $4, $2, $7}' ${file} \
        | grep -v "PATCH" \
        | awk '$1 != "chrMT" && $1 != "chrY" && $1 != "chrX"' \
        | sort -k1,1 -k2,2n \
        > ensembl_mirna_coordinates_hg38.bed
}

_biomart_mirna_coordinates mart_export.txt

# =============================================================================
# ENCODE SCREEN candidate promoters from SCREEN registry of cCREs V3
# Arguments:
#   $1 - path to promotoer file
#   $2 - directory to save parsed files
# =============================================================================
function _screen_promoters () {
    awk -v OFS='\t' '{print $1,$2,$3,"promoter"}' $1 \
        > $2/promoters_parsed_hg38.bed
}


# =============================================================================
# ENCODE SCREEN CTCF-only cCREs from SCREEN V3
# Arguments:
#   $1 - path to ctcf file
#   $2 - cCRE file
#   $3 - directory to save parsed files
# =============================================================================
function _screen_ctcf () {
    grep CTCF-only $1/$2 \
        | awk -v OFS='\t' '{print $1,$2,$3,"ctcfccre"}' $2 \
        > $3/ctcfccre_parsed_hg38.bed
}


# =============================================================================
# Genomic variant hotspots from Long & Xue, Human Genomics, 2021. First, expand
# hotspot clusers to their individual mutation type. Then, file is split into
# CNVs, indels, and SNPs.
# Arguments:
#   $1 - name of hotspot file
#   $1 - name of hotspot file
#   $2 - path to liftOver and liftover chain
#   $3 - directory to save parsed files
# =============================================================================
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


# =============================================================================
# Replication hotspots from Long & Xue, Human Genomics, 2021. Each phase is
# split into a separate file for node attributes.
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
#   $4 - 
# =============================================================================
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


# =============================================================================
# Average recombination rate from deCODE - Halldorsson et al, Science, 2019.
# BigWig is converted to Wig then to .bed via BEDOPS.
# Arguments:
#   $1 -
#   $2 - 
#   $3 - 
#   $4 - 
# =============================================================================
function _recombination () {
    $1/bigWigToWig ${2}/${3}.bw ${2}/${3}.wig
    wig2bed < ${2}/${3}.wig > tmp
    awk -v OFS="\t" '{print $1, $2, $3, $5, $4}' tmp > ${4}/recombination_hg38.bed
}


# =============================================================================
# TF-interactions from TFMarker. The file was not properly delimited so some
# columns were cleaned up in excel first. We filter and only keep TF
# interactions present in normal cells (not cancer cells).
# =============================================================================
function _tf_marker () {
    sed 's/ /_/g' $1 | awk '$5 == "Normal_cell"' > $2/tf_marker.txt
}


# =============================================================================
# Concatenate tf binding locations to single file and annotate each binding
# location with the tf name. Tf binding locations are from Meuleman et al.,
# Nature, 2020. Each site corresponds to a DHS with an overlapping DNase
# footprint and contains known binding motifs.
# =============================================================================
function _tf_binding_dnase () {
    # awk -v OFS="\t" '{print $0,FILENAME}' $1/* > $2/tf_binding_sites.bed
    awk -v OFS="\t" '{n=split(FILENAME, array,"/"); print $0,array[n]}' $1/* \
        | cut -f1,2,3,7 \
        | sed 's/.bed//g' \
        | sort -k1,1 -k2,2n \
        > $2/tfbinding_footprints.bed
}


# =============================================================================
# Simple function to clean up intermediate files
# =============================================================================
function _cleanup () {
    for file in hotspots_expanded_hg18.bed hotspots_lifted_hg38.bed hotspots_unlifted miRNAtargets_lifted_hg38.bed mirRNA_unlifted enhancer_regions_unlifted.txt rep_formatted_hg18.bed rep_lifted_hg38.bed rep_unlifted recombAvg.wig; do
        rm $file
    done
}


# =============================================================================
# Main function to run all preparsing
# Arguments:
#   $1 - 
#   $1 - 
#   $2 - 
#   $3 - 
# =============================================================================
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

    _rbp_site_clusters \
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

    _tf_binding_dnase \
        /ocean/projects/bio210019p/stevesho/data/bedfile_preparse/tf_binding/tfs \
        /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction

    _cleanup
}

# run main function! 
main \
    var1 \
    var2 \
    var3 \
    var4 

echo "Finished in $(convertsecs "${SECONDS}")"