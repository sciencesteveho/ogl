### export variable 
export CEREBRAS_DIR=/ocean/neocortex/cerebras/

### set group to proper charging account
newgrp bio220004p

### copy modelzoo
rsync -PLoptr $CEREBRAS_DIR/modelzoo/graphs $PROJECT

### start interactive session
interact --account bio220004p --partition RM -n 16

### run cerebras container
srun --pty --cpus-per-task=28 --kill-on-bad-exit singularity shell --cleanenv --bind /local1/cerebras/data,/local2/cerebras/data,/local3/cerebras/data,/local4/cerebras/data,$PROJECT /local1/cerebras/cbcore_latest.sif

### test compilation
python run.py --mode train --compile_only --params custom_configs/params_GCN_5000.yaml --model_dir custom_output_dir