for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
rm -r ${tis}
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch prepare_bedfiles.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done


for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch --dependency=afterok:13084164:13084165:13084166:13084167:13084168:13084169:13084170 local_context_parser.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done


for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch --dependency=afterok:13084176:13084177:13084178:13084179:13084180:13084181:13084182 graph_constructor.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done

for i in {0..33..1}; do
sbatch make_scaler.sh $i
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
mkdir ${tis}/parsing/graphs_scaled
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch scale_node_feats.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done

