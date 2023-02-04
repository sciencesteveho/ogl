### move all files to same dir

cd /ocean/projects/bio210019p/stevesho/data/preprocess/tfrecords_5000
mkdir test train validation

for i in {1..12}; do
    cd test_${i}
    for file in *.tfrecords*; do 
        mv $file ${file}_test_${i}
        mv ${file}_test_${i} ../test
    done
    cd ..
done

### move for validation
for i in {1..12}; do
    cd validation_${i}
    for file in *.tfrecords*; do 
        mv $file ${file}_validation_${i}
        mv ${file}_validation_${i} ../validation
    done
    cd ..
done

### move for train
for i in {1..12}; do
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
# mkdir custom_configs custom_output_dir

srun --pty --cpus-per-task=28 --account=bio220004p --partition=RM --kill-on-bad-exit singularity shell --cleanenv --bind $CEREBRAS_DIR/data,$PROJECT $CEREBRAS_DIR/cbcore_latest.sif

### NOTES
'''
DataProcessor is modified. A simple change in filepath to the TFRecords.
Validation and compilation on adjusted YAML
I parsed the TFREcords in parallel into separate folders so that they wouldnt have writers writing to the same file as the same time, then moved each file into the same folder post parsing.
'''

### test validation
python run.py --mode train --validate_only --params custom_configs/params_GCN_5000.yaml --model_dir custom_output_dir

'''
Singularity> python run.py --mode train --validate_only --params custom_configs/params_GCN_5000.yaml --model_dir custom_output_dir
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
Singularity>
'''

### test compilation
python run.py --mode train --compile_only --params custom_configs/params_GCN_5000.yaml --model_dir custom_output_dir

'''
Singularity> python run.py --mode train --compile_only --params custom_configs/params_GCN_5000.yaml --model_dir custom_output_dir
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
RAN COMMAND: stack_main -c custom_output_dir/cs_fd088e25a32db04e55a38e3f6a7f02cca4f4094707a218c9239d577d9bcb6363/config.json -e custom_output_dir/cs_fd088e25a32db04e55a38e3f6a7f02cca4f4094707a218c9239d577d9bcb6363/error.json -v custom_output_dir/cs_fd088e25a32db04e55a38e3f6a7f02cca4f4094707a218c9239d577d9bcb6363/viz/plan_viz.json -w custom_output_dir/cs_fd088e25a32db04e55a38e3f6a7f02cca4f4094707a218c9239d577d9bcb6363/ -d summary --skip /matching/add_gen_fu --skip /placement/debug_kernels --skip /placement/stc_gen_fu --skip /placement/rebalance_delays --start-stage matching --stop-stage /plangen/plangen custom_output_dir/cs_fd088e25a32db04e55a38e3f6a7f02cca4f4094707a218c9239d577d9bcb6363/lair_file.json custom_output_dir/cs_fd088e25a32db04e55a38e3f6a7f02cca4f4094707a218c9239d577d9bcb6363/plan.json
Traceback (most recent call last):
  File "run.py", line 270, in <module>
    main()
  File "run.py", line 264, in main
    predict_input_fn=predict_input_fn,
  File "run.py", line 197, in run
    input_fn, validate_only=runconfig_params["validate_only"], mode=mode
  File "../../../modelzoo/common/tf/estimator/cs_estimator.py", line 72, in compile
    super().compile(input_fn, validate_only, mode)
  File "/cbcore/py_root/cerebras/tf/cs_estimator.py", line 1697, in compile
    role=self.role,
  File "/cbcore/py_root/cerebras/tf/cerebras_components.py", line 1855, in cached_stack_compile
    stack.run()
  File "/cbcore/py_root/cerebras/stack/tools/caching_stack.py", line 318, in run
    stack.run()
  File "/cbcore/py_root/cerebras/stack/tools/stack.py", line 2025, in run
    self._run_capture_output(stage, cmd, outlog, append=True)
  File "/cbcore/py_root/cerebras/stack/tools/stack.py", line 1672, in _run_capture_output
    source=proto.source,
cerebras.common.errors.CerebrasSPLC006Exception: [Cerebras Internal Error (subtype "SPLC006", source "/placement/place")]
           SPLC006: No layer size message
           SPLC006: reshape_inner.unpack.tx0 has no unfiltered splits; visited 95 splits, 95 filtered out by memory upper bound, 0 filtered out by area budget or fabric dimension, 0 filtered out by buffer awareness, 0 filtered out by custom filter, 0 filtered out by unsupported
           SPLC006: Please contact Cerebras Systems Support.
Please contact Cerebras support team.
'''

### train model
