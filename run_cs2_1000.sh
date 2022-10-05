### move all files to same dir

cd /ocean/projects/bio210019p/stevesho/data/preprocess/tfrecords
mkdir test train validation

for i in {1..36}; do
    cd test_${i}
    for file in *.tfrecords*; do 
        mv $file ${file}_test_${i}
        mv ${file}_test_${i} ../test
    done
    cd ..
done

### move for validation
for i in {1..36}; do
    cd validation_${i}
    for file in *.tfrecords*; do 
        mv $file ${file}_validation_${i}
        mv ${file}_validation_${i} ../validation
    done
    cd ..
done

### move for train
for i in {1..36}; do
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
python run.py --mode train --validate_only --params custom_configs/params_GCN.yaml --model_dir custom_output_dir_2

'''

'''

### test compilation
python run.py --mode train --compile_only --params custom_configs/params_GCN.yaml --model_dir custom_output_dir_2

'''

'''

### train model
### log into neocortex first
