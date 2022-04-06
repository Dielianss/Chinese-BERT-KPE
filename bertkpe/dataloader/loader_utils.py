import os
import json
import codecs
import logging
import unicodedata
from tqdm import tqdm
from ..constant import UNK_WORD, BOS_WORD, EOS_WORD
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset
stemmer = PorterStemmer()

from .bert2span_dataloader import (bert2span_preprocessor, bert2span_converter)
from .bert2tag_dataloader import (bert2tag_preprocessor, bert2tag_converter)
from .bert2chunk_dataloader import (bert2chunk_preprocessor, bert2chunk_converter)

from .bert2rank_dataloader import (bert2rank_preprocessor, bert2rank_converter)
from .bert2joint_dataloader import (bert2joint_preprocessor_chinese, bert2joint_converter)


example_preprocessor = {'bert2span': bert2span_preprocessor,
                        'bert2tag': bert2tag_preprocessor,
                        'bert2chunk': bert2chunk_preprocessor,
                        'bert2rank': bert2rank_preprocessor,
                        'bert2joint': bert2joint_preprocessor_chinese}

feature_converter = {'bert2span': bert2span_converter,
                     'bert2tag': bert2tag_converter,
                     'bert2chunk': bert2chunk_converter,
                     'bert2rank': bert2rank_converter,
                     'bert2joint': bert2joint_converter}


logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# load & save source dataset
def load_dataset(file_path):
    """ Load file.jsonl ."""
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(fi):
            jsonl = json.loads(line)
            data_list.append(jsonl)

    logger.info('successfully load %d data' % len(data_list))
    return data_list


def save_dataset(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as fo:
        for data in tqdm(data_list):
            fo.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
        fo.close()

    logger.info("Success save %d data to %s" % (len(data_list), filename))
    
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# build dataset
class build_dataset(Dataset):
    ''' build datasets for train & eval '''
    def __init__(self, args, tokenizer, mode):
        pretrained_model = 'bert' if 'roberta' not in args.pretrained_model_type else 'roberta'
        # --------------------------------------------------------------------------------------------
        # try to reload cached features
        cached_example_path = os.path.join(args.general_cached_features_folder,
                                           "cached.%s.%s.%s.%s.json"
                                           % (args.model_class, pretrained_model, args.dataset_class, mode)
                                           )
        if os.path.exists(cached_example_path):
            logger.info("loading test data at {}".format(cached_example_path))
            cached_examples = reload_cached_features(cached_example_path)
        # --------------------------------------------------------------------------------------------
        # restart preprocessing features
        else:
            logger.info("cached test data does not exist at {}".format(os.path.join(args.general_cached_features_folder,
                                                                                    args.model_class)))
            logger.info("start creating...")
            logger.info("reading raw test data at: {}".format(os.path.join(args.preprocess_folder,
                                                                           "%s.%s.json" % (args.dataset_class, mode))))

            examples = load_dataset(os.path.join(args.preprocess_folder, "%s.%s.json" % (args.dataset_class, mode)))

            # bert2joint 的 preprocessor 已替换为中文的
            preprocessor = example_preprocessor[args.model_class]
            cached_examples = preprocessor(**{'examples': examples, 'tokenizer': tokenizer,
                                              'max_token': args.max_token, 'pretrain_model': pretrained_model,
                                              'mode': mode, 'max_phrase_words': args.max_phrase_words
                                              })

            if args.local_rank in [-1, 0]:
                save_cached_features(**{'cached_examples': cached_examples,
                                        'cached_features_dir': args.general_cached_features_folder,
                                        'model_class': args.model_class,
                                        'dataset_class': args.dataset_class,
                                        'pretrain_model': pretrained_model,
                                        'mode': mode})

        # --------------------------------------------------------------------------------------------
        self.mode = mode
        self.tokenizer = tokenizer
        self.examples = cached_examples
        self.model_class = args.model_class
        self.max_phrase_words = args.max_phrase_words

    def __len__(self):
        return len(self.examples)

    # 在类中实现 __getitem__ 方法时, 可以对类的实例 以切片的形式 访问, 例如 instance[i]
    def __getitem__(self, index):
        return feature_converter[self.model_class](index, self.examples[index],
                                                   self.tokenizer, self.mode, self.max_phrase_words)
    

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# pre-trained model tokenize
def tokenize_for_bert(doc_words, tokenizer):
    # valid_mask: 对 doc_words 中的 所有 token 执行如下编码:
    # token 中的 首个 sub_token记为 1, 其后所有 sub_tokens 记为 0, 1 的 个数即为 token的总数
    valid_mask = []
    # 使用 tokenizer 对 token(若干字组成) 再次分词, 将所有 sub_token(单字) 存入 all_sub_tokens, 例如 ["快乐", "中国"] --> ["快", "乐", "中", "国"]
    all_sub_tokens = []
    # 记录 每一个 sub_tokens 属于 doc_words 中 第几个 token
    tok_to_orig_index = []
    # 记录 预处理时的分词结果 (tokens) 中的 每个 token 中的 首个 sub_token 在 sub_tokens 中的 位置
    orig_to_tok_index = []

    for (i, token) in enumerate(doc_words):
        orig_to_tok_index.append(len(all_sub_tokens))   # len(all_sub_token) 为先前 sub_tokens 的总数
                                                        # ===> 0 ~ len -1 为 先前所有 sub_tokens 的序号
                                                        # len 可作为 当前 token 的首 sub_token 在 所有 sub_tokens 中的 序号
        sub_tokens = tokenizer.tokenize(token)
        if len(sub_tokens) < 1:
            sub_tokens = [UNK_WORD]
        for num, sub_token in enumerate(sub_tokens):
            tok_to_orig_index.append(i)         # 该 sub_token 属于 doc_words 中的 第 i 个 token
            all_sub_tokens.append(sub_token)    # 保存 每个 sub_token
            if num == 0:
                valid_mask.append(1)            # token 中的 首个 sub_token 记为 1
            else:
                valid_mask.append(0)            # token 中 非首个 sub_token 记为 0

    return {'sub_tokens': all_sub_tokens,
            'valid_mask': valid_mask,
            'tok_to_orig_index': tok_to_orig_index,
            'orig_to_tok_index': orig_to_tok_index}


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# load & save cached features
def reload_cached_features(filename: str):
    data_list = []

    with open(filename, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            jsonl = json.loads(line)
            data_list.append(jsonl)
        return data_list


def save_cached_features(cached_examples, cached_features_dir, 
                        model_class, dataset_class, pretrain_model, mode):
    logger.info("start saving:  %s (%s) for %s (%s) cached features ..." 
                %(model_class, pretrain_model, dataset_class, mode))
    if not os.path.exists(cached_features_dir):
        os.mkdir(cached_features_dir)
        
    save_filename = os.path.join(cached_features_dir, "cached.%s.%s.%s.%s.json" 
                            %(model_class, pretrain_model, dataset_class, mode))
    save_dataset(data_list=cached_examples, filename=save_filename)


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# fucntions for converting labels
def flat_rank_pos(start_end_pos):
    flatten_postions = [pos for poses in start_end_pos for pos in poses]
    sorted_positions = sorted(flatten_postions, key=lambda x: x[0])
    return sorted_positions


def strict_filter_overlap(positions):
    '''delete overlap keyphrase positions. '''
    previous_e = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if s <= previous_e:
            continue
        filter_positions.append(positions[i])
        previous_e = e
    return filter_positions


def loose_filter_overlap(positions):
    '''delete overlap keyphrase positions. '''
    previous_s = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if previous_s == s:
            continue
        elif previous_s < s:
            filter_positions.append(positions[i])
            previous_s = s
        else:
            logger.info('Error! previous start large than new start')
    return filter_positions


def limit_phrase_length(positions, max_phrase_words):
    filter_positions = [pos for pos in positions if (pos[1]-pos[0]+1) <= max_phrase_words]
    return filter_positions


# Delete Over Scope keyphrase position (token_len > 510) and phrase_length > 5
def limit_scope_length(start_end_pos, valid_length, max_phrase_words):
    """filter out positions over scope & phase_length > 5"""
    filter_positions = []
    for positions in start_end_pos:
        _filter_position = [pos for pos in positions \
                            if pos[1] < valid_length and (pos[1]-pos[0]+1) <= max_phrase_words]
        if len(_filter_position) > 0:
            filter_positions.append(_filter_position)
    return filter_positions


def stemming(phrase):
    norm_chars = unicodedata.normalize('NFD', phrase)
    stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
    return norm_chars, stem_chars


def whether_stem_existing(gram, phrase2index, tot_phrase_list):
    """If :
       unicoding(gram) and stemming(gram) not in phrase2index, 
       Return : not_exist_flag
       Else :
       Return : index already in phrase2index.
    """
    norm_gram, stem_gram = stemming(gram)
    if norm_gram in phrase2index:
        index = phrase2index[norm_gram]
        phrase2index[stem_gram] = index
        return index

    elif stem_gram in phrase2index:
        index = phrase2index[stem_gram]
        phrase2index[norm_gram] = index
        return index

    else:
        index = len(tot_phrase_list)
        phrase2index[norm_gram] = index
        phrase2index[stem_gram] = index
        tot_phrase_list.append(gram)
        return index
    
    
def whether_existing(gram, phrase2index, tot_phrase_list):
    """If :
       gram not in phrase2index, 
       Return : not_exist_flag
       Else :
       Return : index already in phrase2index.
    """
    if gram in phrase2index:
        index = phrase2index[gram]
        return index
    else:
        index = len(tot_phrase_list)
        phrase2index[gram] = index
        tot_phrase_list.append(gram)
        return index