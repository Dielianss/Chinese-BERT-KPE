import os


# ----------------------------------------------------------------------- Working Directory Configuration
dataset_class = ""                      # dataset name
raw_train_data_path = ""                # the path to your raw training data
raw_test_data_path = ""                 # the path to your raw testing data
base_folder = ""                        # the path to the base working folder of this implementation, where all the results will be saved
general_pretrained_model_path = ""      # the path where pretrained bert model saved
keyphrase_in_test_data = True           # whether test data provide keyphrases

cached_dataset_folder = "cached_features"  # general cached_dataset_folder
prepro_dataset_folder = "prepro_dataset"   # general prepro_dataset_folder
result_dataset_folder = "results"          # general result_save_folder

general_preprocess_folder = os.path.join(base_folder, prepro_dataset_folder)
general_cached_features_folder = os.path.join(base_folder, cached_dataset_folder, dataset_class)


# ----------------------------------------------------------------------- Model Configuration
run_mode = "train"
model_class = "bert2joint"
pretrained_model_type = "chinese_L-12_H-768_A-12"
load_checkpoint = False
save_checkpoint = True

no_cuda = False
local_rank = -1
data_workers = 0
seed = 42

max_token = 510
max_train_epochs = 2
max_train_steps = 0
per_gpu_train_batch_size = 4
per_gpu_test_batch_size = 1
gradient_accumulation_steps = 4

learning_rate = 5e-5
weight_decay = 0.01
warmup_proportion = 0.1
adam_epsilon = 1e-8
max_grad_norm = 1.0

tag_pooling = "min"
eval_checkpoint = ""
max_phrase_words = 5

use_viso = False
display_iter = 200
server_ip = ""
server_port = ""
fp16 = False
fp16_opt_level = "01"

PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
BOS_WORD = '[CLS]'
EOS_WORD = '[SEP]'
DIGIT_WORD = 'DIGIT'

Idx2Tag = ['O', 'B', 'I', 'E', 'U']
Tag2Idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'U': 4}

PAD = 0
UNK = 100
BOS = 101
EOS = 102
DIGIT = 1

# number of outputting most possible candidate keyphrases
Decode_Candidate_Number = 20


class IdxTag_Converter(object):
    ''' idx2tag : a tag list like ['O','B','I','E','U']
        tag2idx : {'O': 0, 'B': 1, ..., 'U':4}
    '''

    def __init__(self, idx2tag):
        self.idx2tag = idx2tag
        tag2idx = {}
        for idx, tag in enumerate(idx2tag):
            tag2idx[tag] = idx
        self.tag2idx = tag2idx

    def convert_idx2tag(self, index_list):
        tag_list = [self.idx2tag[index] for index in index_list]
        return tag_list

    def convert_tag2idx(self, tag_list):
        index_list = [self.tag2idx[tag] for tag in tag_list]
        return index_list

# 'O' : non-keyphrase
# 'B' : begin word of the keyphrase
# 'I' : middle word of the keyphrase
# 'E' : end word of the keyphrase
# 'U' : single word keyphrase