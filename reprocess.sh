# command to grab job ids
# squeue | tail -n +2 | awk '{print $1}' | tr "\n" ":"
# echo | cut -f4 -d' ' | tr "\n" ":" 

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
rm -r ${tis}
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch prepare_bedfiles.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch --dependency=afterok:13634809:13634810:13634811:13634812:13634813:13634814:13634815 local_context_parser.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch --dependency=afterok:13634864:13634865:13634866:13634867:13634868:13634869:13634870 graph_constructor.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch --dependency=afterok:13634871:13634872:13634873:13634874:13634875:13634876:13634877 graph_stats.sh ${tis}
done

for i in {0..33..1}; do
sbatch --dependency=afterok:13635256:13635257:13635258:13635259:13635260:13635261:13635262 make_scaler.sh $i
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
mkdir ${tis}/parsing/graphs_scaled
sbatch --dependency=afterok:13635268:13635269:13635270:13635271:13635272:13635273:13635274:13635275:13635276:13635277:13635278:13635279:13635280:13635281:13635282:13635283:13635284:13635285:13635286:13635287:13635288:13635289:13635290:13635291:13635292:13635293:13635294:13635295:13635296:13635297:13635298:13635299:13635300:13635301 scale_node_feats.sh ${tis}
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
mv ${tis}/gene_regions_tpm_filtered.bed gene_regions_tpm_filtered.bed_2
done