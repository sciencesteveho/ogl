#!/bin/bash
#
# Code to download and parse reference base data for the project. This script
# will assume that your environment has working BEDOPS, BEDTools, and xlsx2csv
# installed. Initial data will be downloaded to the unprocessed directory and
# parsed data placement is hardcoded based on the provided root directory. To
# run the script:
#
# >>> bash reference_data.sh --root_directory /path/to/project/root [--cleanup]
# 
# The script will also prepare the necessary directory structure for the rest of
# OGL. Provided is a schematic of the directory layout:
#
# root_directory
# ├── unprocessed
# ├── raw_tissue_data
# │   └── chromatin_loops
# │       └── processed_loops
# ├── shared_data
#     ├── local
#     ├── regulatory_elements
#     ├── references
#     ├── interaction
#     └── targets
#         ├── expression
#         ├── matrices
#         └── tpm


# =============================================================================
# Set up command line variables
# Arguments:
#   --root_directory: project root directory
#   --cleanup: boolean flag to remove intermediate files
# =============================================================================
# Initialize the variables
root_directory=""
cleanup=false

# Parse the command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --root_directory)
            root_directory="$2"
            shift 2  # shift twice to pass the value
            ;;
        --cleanup)
            cleanup=true
            shift    # shift once since it's a boolean flag
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done


# =============================================================================
# Setting up variables to track time and progress
# =============================================================================
SECONDS=0

function convertsecs() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d\n" $hours $minutes $seconds
}


# Function to echo script progress to stdout
log_progress() {
    echo -e "[$(date +%Y-%m-%dT%H:%M:%S%z)] "
}


# =============================================================================
# Utility function to download raw files if they do not exist
# =============================================================================
function _download_raw_file () {
    local raw_file=$1  # absolute path to raw file
    local download_url=$2  # URL to download raw file

    if [ ! -f ${raw_file} ]; then
        echo "Downloading ${raw_file}..."
        wget -O ${raw_file} ${download_url}
    else
        echo "${raw_file} exists."
    fi
}


# =============================================================================
# Prepare directory folders and subfolders as necessary
# =============================================================================
function _prepare_directory_structure () {
    local root_dir=$1  # project root directory
    local directories=(
        "unprocessed"
        "raw_tissue_data/chromatin_loops/processed_loops"
        "shared_data/interaction"
        "shared_data/local"
        "shared_data/reference"
        "shared_data/regulatory_elements"
        "shared_data/targets/expression"
        "shared_data/targets/tpm"
        "shared_data/targets/matrices"
    )

    for directory in "${directories[@]}"; do
        mkdir -p ${root_dir}/${directory}
    done
}


# =============================================================================
# Convert gencode v26 GTF to bed, remove micro RNA genes and only keep canonical
# "gene" entries. Additionally, keep a lookup table for the converion of from
# gencode ENSG IDs to genesymbol.
# =============================================================================
function _filter_gencode_annotations () {
    local raw_gencode_file=$1  # absolute path to gencode file
    local filtered_gencode_file=$2  # absolute path to gencode bed file

    _download_raw_file \
        ${raw_gencode_file} \
        "https://storage.googleapis.com/gtex_analysis_v8/reference/gencode.v26.GRCh38.genes.gtf"

    gtf2bed < ${raw_gencode_file} \
        | awk '$8 == "gene"' \
        | grep -v miR \
        > ${filtered_gencode_file}
}

function _prepare_gencode_lookup_table () {
    local filtered_gencode_file=$1  # absolute path to filtered gencode bed file
    local lookup_table=$2  # absolute path to lookup table

    cut -f4,10 ${filtered_gencode_file} \
        | sed 's/;/\t/g' \
        | cut -f1,5 \
        | sed -e 's/ gene_name //g' -e 's/\"//g' \
        > ${lookup_table}
}


# =============================================================================
# Transcription start sites for hg38 are downloaded from refTSS v4.1, release
# 01/31/2024. First, we create an associate array to store the tss --> gene
# relationships and then we use them to parse a tss file with each gene symbol
# annotated. Lastly, we convert the gene symbols to gencode IDs and only keep
# genes w/ gencode IDs.
# =============================================================================
function _tss () {
    local raw_tss_file=$1  # absolute path to raw tss file
    local decompressed_tss_file=$2  # absolute path to decompressed tss file
    local parsed_tss_file=$3

    _download_raw_file \
        ${raw_tss_file} \
        https://reftss.riken.jp/datafiles/4.1/human/refTSS_v4.1_human_hg38_annotation.txt.gz

    gunzip -c ${raw_tss_file} > ${decompressed_tss_file}

    # create associative array
    declare -A tss_genes
    while read -r line; do
        key=$(echo "$line" | cut -f1 -d' ')
        value=$(echo "$line" | cut -f8 -d' ')
        tss_genes["$key"]="$value"
    done < "$decompressed_tss_file"

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
    ' ${decompressed_tss_file} > ${parsed_tss_file}
}


# =============================================================================
# MicroRNA targets
# We download the entire miRNA catalogue from miRTarBase (version 9.0) and
# filter the miRNAs to only keep homo sapiens relevant miRNAs with functional
# evidence (remove Non-functional MTI).
# =============================================================================
function _mirtarbase_targets () {
    local raw_mirtarbase_file=$1  # absolute path to mirtarbase xlsx
    local raw_mirtarbase_csv=$2  # absolute path to mirtarbase csv
    local parsed_mirtarbase=$3  # absolute path to final parsed file

    _download_raw_file \
        ${raw_mirtarbase_file} \
        https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2022/cache/download/9.0/miRTarBase_MTI.xlsx

    # convert to csv
    xlsx2csv ${raw_mirtarbase_file} ${raw_mirtarbase_csv}

    {
        echo -e "miRNA\ttarget_gene"
        cut -f2,4,6,8 -d',' ${raw_mirtarbase_csv} \
            | sed 's/,/\t/g' \
            | awk -vFS='\t' '$3 == "Homo sapiens" && $4 ~ /Functional/' \
            | cut -f1,2 \
            | sort -u
    } > ${parsed_mirtarbase}
}


# =============================================================================
# MicroRNA coordinates
# We download coordinates of human miRNAs from miRBase Release 22.1 For the miRBase catalog, we remove unceccessary information and only keep the coordinates and name (casefolded). Because there are some repeat entrys (primary transcript vs gene body), we collapse any redundant mirnas by keeping the larger coordinates. Of not, 66 miRNAs had multiple annotations on disparate chromosomes. We removed these 66 miRNA from our analysis.
# To convert between miRNA IDs and ENSG identifiers, we create a lookup table with coordinates from biomart.
# =============================================================================
function _mirbase_mirnas () {
    local raw_mirbase_file=$1  # absolute path to raw mirbase file
    local parsed_mirbase_file=$2  # absolute path to parsed mirbase file

    _download_raw_file \ 
        ${raw_mirbase_file} \
        https://www.mirbase.org/download/hsa.gff3

    grep -v "^#" ${raw_mirbase_file} \
        | cut -f1,4,5,9 \
        | awk 'BEGIN{FS=OFS="\t"} {split($4, a, ";"); for (i in a) if (a[i] ~ /^Name=/) {split(a[i], b, "="); $4 = tolower(b[2])}} 1' \
        | sort -k4,4 -k1,1 -k2,2n \
        | bedtools groupby -g 4 -c 1,2,3 -o distinct,min,max \
        | awk 'BEGIN{OFS="\t"} {print $2, $3, $4, $1}' \
        | sort -k1,1 -k2,2n \
        | grep -v "," \
        > ${parsed_mirbase_file}
}


function _biomart_mirna_coordinates () {
    local biomart_file=$1  # absolute path to biomart file
    local mirna_coordinates=$2  # absolute path to mirna_coordinate file

    _download_raw_file \
        ${biomart_file} \
       'https://www.ensembl.org/biomart/martservice?query=<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" ><Dataset name = "hsapiens_gene_ensembl" interface = "default" ><Attribute name = "ensembl_gene_id" /><Attribute name = "ensembl_gene_id_version" /><Attribute name = "chromosome_name" /><Attribute name = "start_position" /><Attribute name = "end_position" /><Attribute name = "external_gene_name" /><Attribute name = "mirbase_accession" /><Attribute name = "mirbase_id" /></Dataset></Query>'

    awk '{ if (length($8) > 0) print $0 }' ${biomart_file} \
        | awk 'BEGIN{FS=OFS="\t"} {print "chr"$3, $4, $5, $2, $8}' \
        | grep -v "PATCH" \
        | awk '$1 != "chrMT" && $1 != "chrY" && $1 != "chrX"' \
        | sort -k1,1 -k2,2n \
        > ${mirna_coordinates}
}


# =============================================================================
# RNA Binding protein sites were downloaded from POSTAR 3. We take the list of
# RBPs and their binding sites and intersect their binding sites with gene
# bodies. To get RBP --> Gene interactions. We then keep RBP --> Gene
# interactions that occur in multiple samples, removing singletons. 
#
# To determine RBP binding site clusters, we calculate the node feature (overlap
# with rna binding site clusters), we merge binding sites within 10bp, given
# that RBP binding sites are often small yet clustered (see Li et al., Genome
# Biology, 2017). To keep higher confidence sites we apply two filters: (1) we
# only keep sites that target multiple target genes and (2) we ensure that the
# site exists in at least 3 different samples, because filtering at 2 samples
# will still keep singletons (e.g. HEK293T,HEK_293_FRT could effectively be
# considered the same sample).
# =============================================================================
function _rbp_binding_sites () {
    local raw_postar3_gunzipped=$1  # absolute path to raw postar3 file
    local raw_postar3_txt=$2  # absolute path to gunzipped postar3 file
    local parsed_binding_sites=$3  # absolute path to parsed binding sites
    local gencode=$4  # absolute path to gencode file
    local gencode_lookup=$5  # absolute path to gencode lookup table
    local rbp_network=$6  # absolute path to rbp network in edge list format

    _download_raw_file \
        ${raw_postar3_file} \
        https://cloud.tsinghua.edu.cn/seafhttp/files/ec09ccf7-6b81-410b-8c0f-ff229bf26021/human.txt.gz

    gunzip -c ${raw_postar3_gunzipped} > ${raw_postar3_txt}

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
        > ${parsed_binding_sites}

        awk 'BEGIN { FS=OFS="\t" } NR==FNR { lookup[$2]=$1; next } $1 in lookup { $1=lookup[$1] } 1' ${gencode_lookup} ${parsed_binding_sites} \
            | grep -e '^ENSG' \
            > ${rbp_network}
}


function _rbp_site_clusters () {
    local raw_postar3_txt=$1  # absolute path to gunzipped postar3 file
    local parsed_binding_sites=$2  # absolute path to parsed binding sites

    awk '$6 != "RBP_occupancy"' ${raw_postar3_txt} \
        | bedtools merge \
        -d 10 \
        -i - \
        -c 6,6,8,8 \
        -o distinct,count_distinct,count_distinct,distinct \
        | awk '($5 > 1) && ($6 > 3)' \
        | cut -f1,2,3,4 \
        | sed 's/,/_/g' \
        | awk -v OFS='\t' '{print $1, $2, $3, "rnab_"$4}' \
        > ${parsed_binding_sites}
}


# =============================================================================
# Make node_attribute references for gencode, superenhancer, and rbp data. These
# references exist as a lookup table for node attributes downdstream during
# graph construction, which allows us to go from an edge list to a list of
# basenodes and their locations.
# The first part of the function adds attrs for gencode nodes but adds duplicate
# entries to differentiate the protein or tf as opposed to the gene body.
# =============================================================================
function _node_featsaver () {
    local gencode_file=$1  # absolute path to gencode file
    local genocode_attr=$2  # absolute path to gencode node attribute file
    local sedb_list=$3  # absolute path to superenhancer database file
    local sedb_attr=$4  # absolute path to superenhancer node attribute file

    cat <(awk -v OFS='\t' '{print $1, $2, $3, $4}' ${gencode_file}) \
        <(awk -v OFS='\t' '{print $1, $2, $3, $4"_tf"}' ${gencode_file}) \
        # <(awk -v OFS='\t' '{print $1, $2, $3, $4"_protein"}' $1/$2) \
        > ${genocode_attr}

    _download_raw_file \
        ${sedb_list} \
        https://bio.liclab.net/sedb/download/new/package/SE_package_hg38.bed

    tail -n +2 ${sedb_list} \
        | awk -vOFS='\t' '{print $3,$4,$5,$3"_"$4"_superenhancer"}' \
        > ${sedb_attr}
}


# =============================================================================
# Run main_func function! 
# =============================================================================
function main () {
    log_progress "Setting up input vars"
    local root_dir=$1  # project root directory

    # Set up directory paths
    unprocessed_dir="$root_dir/unprocessed"
    interaction_dir="$root_dir/shared_data/interaction"
    local_dir="$root_dir/shared_data/local"
    reference_dir="$root_dir/shared_data/reference"
    regulatory_elements_dir="$root_dir/shared_data/regulatory_elements"
    expression_dir="$root_dir/shared_data/targets/expression"
    tpm_dir="$root_dir/shared_data/targets/tpm"
    matrices_dir="$root_dir/shared_data/targets/matrices"

    log_progress "Preparing directory structure"
    _prepare_directory_structure \
        "$root_dir"

    log_progress "Download files that do not require processing"
    declare -A files_to_download=(
        ["$reference_dir/hg38.chrom.sizes"]="https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes"
        ["$reference_dir/liftOver"]="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
        ["$reference_dir/hg19ToHg38.over.chain.gz"]="https://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz"
        ["$unprocessed_dir/hg38.fa.gz"]="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
        ["$unprocessed_dir/hg38-blacklist.v2.bed.gz"]="https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz"
        ["$unprocessed_dir/collapsed_motifs_overlapping_consensus_footprints_hg38.bed.gz"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/consensus.index/collapsed_motifs_overlapping_consensus_footprints_hg38.bed.gz"
    )

    # download each file
    for target_path in "${!files_to_download[@]}"; do
        _download_raw_file "$target_path" "${files_to_download[$target_path]}"
    done

    # set execution permission for liftOver tool
    chmod +x "$reference_dir/liftOver"

    log_progress "Decompressing zip files"
    gunzip -c "$unprocessed_dir/hg38.fa.gz" > "$reference_dir/hg38.fa"
    gunzip -c "$unprocessed_dir/hg38-blacklist.v2.bed.gz" > "$reference_dir/hg38-blacklist.v2.bed"
    gunzip -c "$unprocessed_dir/collapsed_motifs_overlapping_consensus_footprints_hg38.bed.gz" > "$reference_dir/collapsed_motifs_overlapping_consensus_footprints_hg38.bed"

    log_progress "Prepare gencode related files"
    _filter_gencode_annotations \
        "$unprocessed_dir/gencode.v26.GRCh38.genes.gtf" \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed"

    _prepare_gencode_lookup_table \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$reference_dir/gencode_to_genesymbol_lookup_table.txt"

    # make a symlink and place gencode file in local
    ln -s \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$local_dir/gencode_v26_genes_only_with_GTEx_targets.bed"

    log_progress "Prepare TSS files"
    _tss \
        "$unprocessed_dir/refTSS_v4.1_human_hg38_annotation.txt.gz" \
        "$unprocessed_dir/refTSS_v4.1_human_hg38_annotation.txt" \
        "$local_dir/tss_parsed_hg38.bed"

    log_progress "Prepare miRTarBase files"
    _mirtarbase_targets \
        "$unprocessed_dir/miRTarBase_MTI.xlsx" \
        "$unprocessed_dir/miRTarBase_MTI.csv" \
        "$reference_dir/mirtargets_filtered.txt"

    log_progress "Prepare miRBase miRNA files"
    _mirbase_mirnas \
        "$unprocessed_dir/hsa.gff3" \
        "$reference_dir/mirbase_mirnas.bed"

    _biomart_mirna_coordinates \
        "$unprocessed_dir/mart_export.txt" \
        "$reference_dir/ensembl_mirna_coordinates_hg38.bed"

    log_progress "Prepare RBP network"
    _rbp_binding_sites \
        "$unprocessed_dir/human.txt.gz" \
        "$unprocessed_dir/human.txt" \
        "$unprocessed_dir/rbp_gene_binding_sites.bed" \
         \
        "$reference_dir/gencode_to_genesymbol_lookup_table.txt" \
        "$reference_dir/rbp_gene_network.txt"

    _rbp_site_clusters \
        "$unprocessed_dir/human.txt" \
        "$local_dir/rbpsiteclusters_parsed_hg38.bed"

    log_progress "Prepare node attribute files"
    _node_featsaver \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$reference_dir/gencode_v26_node_attr.bed" \
        "$unprocessed_dir/SE_package_hg38.bed" \
        "$reference_dir/se_node_attr.bed"
}


# check if the root_directory is not set
if [[ -z "$root_directory" ]]; then
    echo "Error: --root_directory not set."
    echo "Usage: $0 --root_directory PATH [--cleanup]"
    exit 1
else
    echo "Root directory is set to $root_directory"
    main ${root_directory}
fi

# run optional cleanup
if [[ "$cleanup" == true ]]; then
    echo "Cleanup boolean is set to true. Removing intermediate files..."
    rm -r "$root_directory/unprocessed"
fi

echo "Finished preparing reference data. Total time: $(convertsecs SECONDS)"