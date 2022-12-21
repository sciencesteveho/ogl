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
sbatch local_context_parser.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch --dependency=afterok:13635368:13635367:13635366:13635365:13635364:13635363:13635362 graph_constructor.sh genomic_graph_mutagenesis/configs/${tis}.yaml
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
sbatch --dependency=afterok:13635369:13635370:13635371:13635372:13635373:13635374:13635375 graph_stats.sh ${tis}
done

for i in {0..33..1}; do
sbatch --dependency=afterok:13635378:13635379:13635380:13635381:13635382:13635383:13635384 make_scaler.sh $i
done

for tis in hippocampus left_ventricle mammary pancreas skeletal_muscle liver lung; do
mkdir ${tis}/parsing/graphs_scaled
sbatch --dependency=afterok:13635385:13635386:13635387:13635388:13635389:13635390:13635391:13635392:13635393:13635394:13635395:13635396:13635397:13635398:13635399:13635400:13635401:13635402:13635403:13635404:13635405:13635406:13635407:13635408:13635409:13635410:13635411:13635412:13635413:13635414:13635415:13635416:13635417:13635418 scale_node_feats.sh ${tis}
done

rm -r tfrecords 
mkdir tfrecords
for folder in test train validation; do
mkdir tfrecords/${folder}
done

for folder in test train validation; do
sbatch tf_record_parser.sh $folder
done