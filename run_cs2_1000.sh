### move all files to same dir

cd /ocean/projects/bio210019p/stevesho/data/preprocess/tfrecords
mkdir test train validation

for i in {1..24}; do
    cd test_${i}
    for file in *.tfrecords*; do 
        mv $file ${file}_test_${i}
        mv ${file}_test_${i} ../test
    done
    cd ..
done

### move for validation
for i in {1..24}; do
    cd validation_${i}
    for file in *.tfrecords*; do 
        mv $file ${file}_validation_${i}
        mv ${file}_validation_${i} ../validation
    done
    cd ..
done

### move for train
for i in {2..24}; do
    cd train_${i}
    for file in *.tfrecords*; do 
        mv $file ${file}_train_${i}
        mv ${file}_train_${i} ../train
    done
    cd ..
done


### export variable 
export CEREBRAS_DIR=/ocean/neocortex/cerebras/

### set group to proper charging account
newgrp bio220004p

PROJECT='/ocean/projects/bio210019p/stevesho'

### copy modelzoo
# rsync -PLoptr $CEREBRAS_DIR/modelzoo $PROJECT/

### start interactive session
# interact --account bio220004p --partition RM -n 16

### run cerebras container
cd $PROJECT/modelzoo/graphs/tf

### make custom dirs
# mkdir custom_configs custom_output_dir

srun --pty --cpus-per-task=28 --account=bio220004p --partition RM-shared --kill-on-bad-exit singularity shell --cleanenv --bind $CEREBRAS_DIR/data,$PROJECT $CEREBRAS_DIR/cbcore_latest.sif


### NOTES
'''
DataProcessor is modified. A simple change in filepath to the TFRecords.
Validation and compilation on adjusted YAML
I parsed the TFREcords in parallel into separate folders so that they wouldnt have writers writing to the same file as the same time, then moved each file into the same folder post parsing.
'''

### test validation
python run.py --mode train --validate_only --params custom_configs/params_GCN_1000.yaml --model_dir custom_output_dir_2

'''
Singularity> python run.py --mode train --validate_only --params custom_configs/params_GCN_1000.yaml --model_dir custom_output_dir_2
INFO:tensorflow:TF_CONFIG environment variable: {}
INFO:root:Running None on CS-2
INFO:root:---------- Suggestions to improve input_fn performance ----------
WARNING:root:[input_fn] - interleave(): in ParallelInterleaveDatasetV3, `cycle_length` is not being set to CS_AUTOTUNE. Currently, it is set to 4. If determinism is not required, Using CS_AUTOTUNE is likely to improve performance unless you are deliberately using a fine-tuned value.e.g. dataset = dataset.interleave(map_func, cycle_length=cerebras.tf.tools.analyze_input_fn.CS_AUTOTUNE)
INFO:root:[input_fn] - batch(): batch_size set to 32
WARNING:root:Map is called prior to Batch. Consider reversing the order and performing the map function in a batched fashion to increase the performance of the input function
WARNING:root:[input_fn] - flat_map(): use map() instead of flat_map() to improve performance and parallelize reads. If you are not calling `flat_map` directly, check if you are using: from_generator, TextLineDataset, TFRecordDataset, or FixedLenthRecordDataset. If so, set `num_parallel_reads` to > 1 or cerebras.tf.tools.analyze_input_fn.CS_AUTOTUNE, and map() will be used automatically
INFO:root:[input_fn] - TFRecordDataset: buffer_size set to 320
INFO:root:----------------- End of input_fn suggestions -----------------
WARNING:tensorflow:From /cbcore/python/python-x86_64/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:1666: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
INFO:cerebras.stack.tools.caching_stack:Using lair flow into stack
=============== Cerebras Compilation Completed ===============
'''

### test compilation
python run.py --mode train --compile_only --params custom_configs/params_GCN_1000.yaml --model_dir custom_output_dir_2

'''
Singularity> python run.py --mode train --compile_only --params custom_configs/params_GCN_1000.yaml --model_dir custom_output_dir_2
INFO:tensorflow:TF_CONFIG environment variable: {}
INFO:root:Running None on CS-2
INFO:root:---------- Suggestions to improve input_fn performance ----------
WARNING:root:[input_fn] - interleave(): in ParallelInterleaveDatasetV3, `cycle_length` is not being set to CS_AUTOTUNE. Currently, it is set to 4. If determinism is not required, Using CS_AUTOTUNE is likely to improve performance unless you are deliberately using a fine-tuned value.e.g. dataset = dataset.interleave(map_func, cycle_length=cerebras.tf.tools.analyze_input_fn.CS_AUTOTUNE)
INFO:root:[input_fn] - batch(): batch_size set to 32
WARNING:root:Map is called prior to Batch. Consider reversing the order and performing the map function in a batched fashion to increase the performance of the input function
WARNING:root:[input_fn] - flat_map(): use map() instead of flat_map() to improve performance and parallelize reads. If you are not calling `flat_map` directly, check if you are using: from_generator, TextLineDataset, TFRecordDataset, or FixedLenthRecordDataset. If so, set `num_parallel_reads` to > 1 or cerebras.tf.tools.analyze_input_fn.CS_AUTOTUNE, and map() will be used automatically
INFO:root:[input_fn] - TFRecordDataset: buffer_size set to 320
INFO:root:----------------- End of input_fn suggestions -----------------
WARNING:tensorflow:From /cbcore/python/python-x86_64/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:1666: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
INFO:cerebras.stack.tools.caching_stack:Using lair flow into stack
=============== Cerebras Compilation Completed ===============
'''

### train model
### log into neocortex first
