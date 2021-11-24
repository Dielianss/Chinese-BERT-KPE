export DATA_PATH=/data/yangj/keyphrase

export dataset_class="chinese" # openkp , kp20k
export max_train_steps=20810 #  20810 (openkp) , 73430 (kp20k)

export model_class="bert2joint" # bert2span, bert2tag, bert2chunk, bert2rank, bert2joint
export pretrain_model="chinese_L-12_H-768_A-12" # chinese_L-12_H-768_A-12

## --------------------------------------------------------------------------------
## DataParallel (Multi-GPUs)

CUDA_VISIBLE_DEVICES=0 python ./train_chinese.py --run_mode train \
--local_rank -1 \
--max_train_epoch 3 \
--model_class $model_class \
--dataset_class $dataset_class \
--pretrain_model_type $pretrain_model \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--per_gpu_test_batch_size 1 \
--preprocess_folder $DATA_PATH/prepro_dataset \
--pretrain_model_path /path/to/the/pretrained_bert_model \
--cached_features_dir $DATA_PATH/cached_features/$dataset_class \
--display_iter 1000 \
--save_checkpoint \