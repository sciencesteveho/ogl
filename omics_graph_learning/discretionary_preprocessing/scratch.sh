for cell in adrenal aorta hippocampus lung ovary pancreas skeletal_muscle small_intestine spleen; do
    case "$cell" in
        gm12878|h1-esc|hepg2|hmec|imr90|k562|nhek|left_ventricle)
            resolution=5000
            ;;
        adrenal|aorta|hippocampus|liver|lung|ovary|pancreas|skeletal_muscle|small_intestine|spleen)
            resolution=40000
            ;;
        *)
            echo "Invalid cell type: $cell"
            continue
            ;;
    esac
    sbatch balance_coarse_grain.sh ${cell} 0.0
done

for cell in adrenal aorta hippocampus lung ovary pancreas skeletal_muscle small_intestine spleen;
do
    cat ${cell}/*_0.0.bedpe* \
    | sort -k7,7n \
    > topk/${cell}_contacts.bed
done

for cell in adrenal aorta hippocampus lung ovary pancreas skeletal_muscle small_intestine spleen;
do
    for num in 60000 75000 100000; do
        head -n ${num} topk/${cell}_contacts.bed > topk/${cell}_contacts_${num}.bed
    done
done
