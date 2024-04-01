#!/bin/bash
#
# Script to process loops from DeepAnchor into bedpe format

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
# Function to convert .bigInteract files from DeepAnchor over to .bedpe files.
# Additionally liftovers the coordinates from hg19 to hg38
# =============================================================================
function _format_deepanchor_loops () {
    # Declare variables as local
    local processing_dir="$1"  # working directory
    local biginteract_dir="$2"  # directory with .bigInteract loop calls
    local prefix="$3"  # prefix for .bigInteract file
    local genome_file="$4"  # genome / chr size file
    local liftover_dir="$5"  # resource directory
    local output_dir="$6"  # final directory to place lifted and formatted calls
    local output_prefix="$7"  # tissue naming for final file
    
    # make directory for processing
    if [ ! -d "$processing_dir" ]; then
        mkdir "$processing_dir"
    fi
    
    "$biginteract_dir/bigBedToBed" \
        "$biginteract_dir/${prefix}.bigInteract" \
        "$processing_dir/${prefix}.bedpe" 

    # Split bedpe and extend regions to approximate 5kb resolution, and to
    # determine a cutoff for anchor overlap to create edges to regulatory
    # elements. The first anchor is extended upstream, and the second anchor is
    # extended downstream.
    awk -v OFS='\t' '{print $9,$10,$11,NR}' "$processing_dir/${prefix}.bedpe" \
        | bedtools slop \
        -i stdin \
        -g "$genome_file" \
        -r 4981 \
        -l 0 \
        > "$processing_dir/${prefix}.bedpe_1"
    
    awk -v OFS='\t' '{print $14,$15,$16,NR}' "$processing_dir/${prefix}.bedpe" \
        | bedtools slop \
        -i stdin \
        -g "$genome_file" \
        -r 0 \
        -l 4981 \
        > "$processing_dir/${prefix}.bedpe_2"

    # liftover hg19 to hg38
    for file in bedpe_1 bedpe_2;
    do
        local outfile="${processing_dir}/${prefix}.${file}.hg38"
        local unmappedfile="${processing_dir}/${prefix}.${file}.unmapped"
        "$liftover_dir/liftOver" \
            "$processing_dir/${prefix}.${file}" \
            "$liftover_dir/hg19ToHg38.over.chain.gz" \
            "$outfile" \
            "$unmappedfile"
        
        sort -k4,4 -o "$outfile" "$outfile"
    done

    # join bedpe by matching anchor number
    join \
        -j 4 \
        -o 1.1,1.2,1.3,2.1,2.2,2.3 \
        "${processing_dir}/${prefix}.bedpe_1.hg38" \
        "${processing_dir}/${prefix}.bedpe_2.hg38" \
        | sed 's/ /\t/g' \
        | sort -k8,8n \
        > "${output_dir}/${output_prefix}_deepanchor.bedpe.hg38"
}


# =============================================================================
# Process deepanchor loops
# =============================================================================
deepanchor_processing_main () {
    local deepanchor_dir=$1
    local final_dir=$2

    local -A loop_files=(
        ["ENCFF000RWO"]="k562"
        ["ENCFF000YCK"]="imr90"
        ["ENCFF001HHX"]="gm12878"
        ["GSM749715"]="hepg2"
        ["ENCFF000ONR"]="h1-esc"
        ["ENCFF001HPM"]="hmec"
        ["GSM749707"]="nhek"
        ["ENCFF000CFW"]="hippocampus"
        ["ENCFF000RYN"]="lung"
        ["ENCLB993ADU"]="pancreas"
        ["ENCLB645ZVC"]="skeletal_muscle"
        ["ENCLB020WHN"]="small_intestine"
        ["ENCFF002EXB"]="liver"
        ["ENCLB432SKR"]="aorta"
        ["ENCLB120LGG"]="skin"
        ["ENCLB303UFA"]="left_ventricle"
        ["ENCFF001HPM"]="mammary"
        ["ENCFF000SEC"]="spleen"
        ["ENCLB026GMS"]="ovary"
        ["ENCLB733WTO"]="adrenal"
    )

    local -A combine_files=(
        ["hippocampus"]="Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops"
        ["left_ventricle"]="Leung_2015.VentricleLeft.hg38.peakachu-merged.loops"
        ["liver"]="Leung_2015.Liver.hg38.peakachu-merged.loops"
        ["lung"]="Schmitt_2016.Lung.hg38.peakachu-merged.loops"
        ["mammary"]="Rao_2014.HMEC.hg38.peakachu-merged.loops"
        ["pancreas"]="Schmitt_2016.Pancreas.hg38.peakachu-merged.loops"
        ["skeletal_muscle"]="Schmitt_2016.Psoas.hg38.peakachu-merged.loops"
        ["skin"]="Rao_2014.NHEK.hg38.peakachu-merged.loops"
        ["small_intestine"]="Schmitt_2016.Bowel_Small.hg38.peakachu-merged.loops"
        ["aorta"]="Leung_2015.Aorta.hg38.peakachu-merged.loops"
    )

    for file in "${!loop_files[@]}"; do
        _format_deepanchor_loops \
            "${deepanchor_dir}" \
            "/ocean/projects/bio210019p/stevesho/resources/deepanchor/Xuhang01-LoopAnchor-b531d97/loopanchor/data/loops/" \
            "$file" \
            "/ocean/projects/bio210019p/stevesho/resources/hg19.chrom.sizes.txt" \
            "/ocean/projects/bio210019p/stevesho/resources" \
            "${final_dir}" \
            "${loop_files[$file]}"
    done
}