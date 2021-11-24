export DATA_PATH=/data/yangj/keyphrase
export dataset_class="chinese" # openkp , kp20k
export model_class="bert2joint" # bert2span, bert2tag, bert2chunk, bert2rank, bert2joint
export pretrain_model="chinese_L-12_H-768_A-12" # chinese_L-12_H-768_A-12


CUDA_VISIBLE_DEVICES=0 python test.py --run_mode test \
--model_class bert2joint \
--pretrain_model_type roberta-base \
--dataset_class $dataset_class \
--per_gpu_test_batch_size 32 \
--preprocess_folder $DATA_PATH/prepro_dataset \
--pretrain_model_path $DATA_PATH/pretrain_model \
--cached_features_dir $DATA_PATH/cached_features \
--eval_checkpoint /home/sunsi/checkpoints/bert2joint/bert2joint.openkp.roberta.checkpoint \