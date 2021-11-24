export DATA_PATH=/data/yangj/keyphrase/

# preprocess openkp or kp20k
python /data/yangj/keyphrase/bert-kpe/preprocess/preprocess.py --dataset_class kp20k --source_dataset_dir $DATA_PATH/dataset --output_path $DATA_PATH/prepro_dataset