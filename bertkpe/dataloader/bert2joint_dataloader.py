import os
import sys
import json
import torch
import logging
import traceback
from tqdm import tqdm
from . import loader_utils
from ..constant import BOS_WORD, EOS_WORD

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# preprocess label
# ------------------------------------------------------------------------------------------
def convert_to_label(filter_positions, tot_mention_list, differ_phrase_num):
    """First check keyphrase mentions index is same ; 
       Then set keyprhase ngrams = +1  and other phrase candidates = -1 .
    """
    ngram_label = [-1 for _ in range(differ_phrase_num)]
    chunk_label_list = [[0] * len(tot_mention_list[i]) for i in range(len(tot_mention_list))]
    
    for i, positions in enumerate(filter_positions):
        for s, e in positions:
            chunk_label_list[e-s][s] = 1
            key_index = tot_mention_list[e-s][s]
            ngram_label[key_index] = 1
            
    # flat chunk label
    chunk_label = [_chunk for chunks in chunk_label_list for _chunk in chunks]
    
    # keep have more than one positive and one negtive
    if (1 in ngram_label) and (-1 in ngram_label) and (1 in chunk_label) and (0 in chunk_label):
        return ngram_label, chunk_label
    else:
        return None, None

# -------------------------------------------------------------------------------------------
#
def get_ngram_features(doc_words, max_gram, stem_flag=False):
    """
    :param doc_words: Bert分词器输出的 sub_tokens, 取前 510
    :param max_gram: max_phrase_words, 关键词最大词数, 预设值, 默认为 5
    :param stem_flag:
    :return:
    """
    # 记录 所有 tot_phrase_list 中的每一词组在 此 list 中的 index
    phrase2index = {}  # use to shuffle same phrases
    # 穷举所有可能出现的(连续)词组dict
    tot_phrase_list = []  # use to final evaluation
    tot_mention_list = []  # use to train pooling the same

    gram_num = 0
    for n in range(1, max_gram + 1):
        valid_length = (len(doc_words) - n + 1)
        if valid_length < 1:
            break

        _ngram_list = []
        _mention_list = []
        for i in range(valid_length):
            gram_num += 1
            n_gram = " ".join(doc_words[i:i + n]).lower()
            if stem_flag:
                index = loader_utils.whether_stem_existing(n_gram, phrase2index, tot_phrase_list)
            else:
                index = loader_utils.whether_existing(n_gram, phrase2index, tot_phrase_list)

            _mention_list.append(index)
            _ngram_list.append(n_gram)

        tot_mention_list.append(_mention_list)

    assert len(tot_phrase_list) > 0

    assert (len(tot_phrase_list) - 1) == max(tot_mention_list[-1])
    assert sum([len(_mention_list) for _mention_list in tot_mention_list]) == gram_num
    return {"tot_phrase_list": tot_phrase_list, "tot_mention_list": tot_mention_list}


def get_ngram_features_chinese(doc_words, max_gram, stem_flag=False):
    """
    :param doc_words: Bert分词器输出的 sub_tokens, 取前 510
    :param max_gram: max_phrase_words, 关键词最大词数, 预设值, 默认为 5
    :param stem_flag:
    :return:
    """
   
    # 记录 所有 tot_phrase_list 中的每一词组在 此 list 中的 index
    phrase2index = {}  # use to shuffle same phrases
    # 在指定允许的词组长度(即词组中的词语)下, 穷举所有可能出现的(连续)词组dict
    tot_phrase_list = []  # use to final evaluation
    # 穷举所有可能的词组长度
    tot_mention_list = []  # use to train pooling the same

    gram_num = 0

    for n in range(1, max_gram+1):
        valid_length = (len(doc_words) - n + 1)
        if valid_length < 1:
            break

        _ngram_list = []
        _mention_list = []
        for i in range(valid_length):
            gram_num += 1
            # 中文的 子词之间不需要 加空格!
            n_gram = "".join(doc_words[i:i+n]).lower()
            if stem_flag:
                index = loader_utils.whether_stem_existing(n_gram, phrase2index, tot_phrase_list)
            else:
                index = loader_utils.whether_existing(n_gram, phrase2index, tot_phrase_list)
            # 为所有穷举的词组编号
            _mention_list.append(index)
            # 穷举所有可能的(连续)词组
            _ngram_list.append(n_gram)

        tot_mention_list.append(_mention_list)

    assert len(tot_phrase_list) > 0
        
    assert (len(tot_phrase_list) - 1) == max(tot_mention_list[-1])
    assert sum([len(_mention_list) for _mention_list in tot_mention_list]) == gram_num
    return {"tot_phrase_list": tot_phrase_list, "tot_mention_list": tot_mention_list}


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def get_ngram_chunk_features(doc_words, max_gram, keyphrases, stem_flag=False):
    
    keyphrases_list = [" ".join(kp).lower() for kp in keyphrases]
    
    chunk_label = []
    
    phrase2index = {} # use to shuffle same phrases
    tot_phrase_list = [] # use to final evaluation
    tot_mention_list = [] # use to train pooling the same
    
    gram_num = 0
    for n in range(1, max_gram+1):
        valid_length = (len(doc_words) - n + 1)
        
        if valid_length < 1:
            break

        _ngram_list = []
        _mention_list = []
        for i in range(valid_length):

            gram_num += 1
            
            n_gram = " ".join(doc_words[i:i+n]).lower()
            
            if stem_flag:
                index = loader_utils.whether_stem_existing(n_gram, phrase2index, tot_phrase_list)
            else:
                index = loader_utils.whether_existing(n_gram, phrase2index, tot_phrase_list)
                
            # -----------------------------------------------------------------
            # chunk label
            if n_gram in keyphrases_list:
                chunk_label.append(1)
            else:
                chunk_label.append(0)
            # -----------------------------------------------------------------
                
            _mention_list.append(index)
            _ngram_list.append(n_gram)

        tot_mention_list.append(_mention_list)

    assert len(tot_phrase_list) > 0
        
    assert (len(tot_phrase_list) - 1) == max(tot_mention_list[-1])
    assert sum([len(_mention_list) for _mention_list in tot_mention_list]) == gram_num == len(chunk_label)
    return {"tot_phrase_list": tot_phrase_list, "tot_mention_list": tot_mention_list, "chunk_label": chunk_label}


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def get_ngram_info_label(doc_words, max_phrase_words, stem_flag, start_end_pos=None):
    """
    :param doc_words: Bert分词器所得的所有 sub_tokens, 截取前 510个
    :param max_phrase_words: 预设值, 默认为 5(词)
    :param stem_flag:
    :param start_end_pos: 关键词在 预处理分词(词粒度) 中 的首次出现时的起始位置
    :return:
    """
    returns = {"overlen_flag": False, "ngram_label": None, "chunk_label":None}
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    feature = get_ngram_features(doc_words=doc_words, max_gram=max_phrase_words, stem_flag=stem_flag)
    returns["tot_phrase_list"] = feature["tot_phrase_list"]
    returns["tot_mention_list"] = feature["tot_mention_list"]

    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    if start_end_pos is not None:
        filter_positions = loader_utils.limit_scope_length(start_end_pos, len(doc_words), max_phrase_words)
        
        # check over_length
        if len(filter_positions) != len(start_end_pos):
            returns["overlen_flag"] = True
            
        if len(filter_positions) > 0:
            returns["ngram_label"], returns["chunk_label"] = convert_to_label(**{"filter_positions": filter_positions, 
                                                                                 "tot_mention_list": feature["tot_mention_list"], 
                                                                                 "differ_phrase_num": len(feature["tot_phrase_list"])})
        else:
            returns["ngram_label"] = None
            returns["chunk_label"] = None
            
    return returns


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def get_ngram_info_label_chinese(doc_words, max_phrase_words, stem_flag, start_end_pos=None):
    """
    :param doc_words: Bert分词器所得的所有 sub_tokens, 截取前 510个
    :param max_phrase_words: 预设值, 默认为 5(词)
    :param stem_flag:
    :param start_end_pos: 关键词在 预处理分词(词粒度) 中 的首次出现时的起始位置
    :return:
    """
    returns = {"overlen_flag": False, "ngram_label": None, "chunk_label": None}
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    feature = get_ngram_features_chinese(doc_words=doc_words, max_gram=max_phrase_words, stem_flag=stem_flag)
    # 在所有可能的词组长度下, 文档对应的潜在关键词种类
    returns["tot_phrase_list"] = feature["tot_phrase_list"]
    # 在所有可能的词组长度下, 每一长度下, 所有潜在的关键词的编码
    returns["tot_mention_list"] = feature["tot_mention_list"]

    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    if start_end_pos is not None:
        filter_positions = loader_utils.limit_scope_length(start_end_pos, len(doc_words), max_phrase_words)

        # check over_length
        if len(filter_positions) != len(start_end_pos):
            returns["overlen_flag"] = True

        if len(filter_positions) > 0:
            returns["ngram_label"], returns["chunk_label"] = convert_to_label(**{"filter_positions": filter_positions,
                                                                                 "tot_mention_list": feature[
                                                                                     "tot_mention_list"],
                                                                                 "differ_phrase_num": len(
                                                                                     feature["tot_phrase_list"])})
        else:
            returns["ngram_label"] = None
            returns["chunk_label"] = None

    return returns


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def bert2joint_preprocessor(examples, tokenizer, max_token, pretrain_model, mode, max_phrase_words, stem_flag=False):
    logger.info('start preparing (%s) features for bert2joint (%s) ...' % (mode, pretrain_model))
    
    overlen_num = 0
    new_examples = []    
    for idx, ex in enumerate(tqdm(examples)):
        # tokenize
        tokenize_output = loader_utils.tokenize_for_bert(doc_words=ex['doc_words'], tokenizer=tokenizer)
        
        if len(tokenize_output['tokens']) < max_token:
            max_word = max_token
        else:
            max_word = tokenize_output['tok_to_orig_index'][max_token-1] + 1

        new_ex = {}
        new_ex['url'] = ex['url']
        new_ex['tokens'] = tokenize_output['tokens'][:max_token]
        new_ex['valid_mask'] = tokenize_output['valid_mask'][:max_token]
        new_ex['doc_words'] = ex['doc_words'][:max_word]
        assert len(new_ex['tokens']) == len(new_ex['valid_mask'])
        assert sum(new_ex['valid_mask']) == len(new_ex['doc_words'])

        # ---------------------------------------------------------------------------
        parameter = {"doc_words": new_ex['doc_words'], 
                     "max_phrase_words": max_phrase_words, "stem_flag": stem_flag}
        # ---------------------------------------------------------------------------
        if mode == 'train':
            parameter["keyphrases"] = ex['keyphrases']
            parameter["start_end_pos"] = ex['start_end_pos']
        # ---------------------------------------------------------------------------
        # obtain gram info and label
        info_or_label = get_ngram_info_label(**parameter)
        
        new_ex["phrase_list"] = info_or_label["tot_phrase_list"]
        new_ex["mention_lists"] = info_or_label["tot_mention_list"]
        
        if info_or_label["overlen_flag"]:
            overlen_num += 1
        # ---------------------------------------------------------------------------
        if mode == 'train':
            if not info_or_label["ngram_label"]:
                continue
            new_ex["keyphrases"] = ex["keyphrases"]
            new_ex["ngram_label"] = info_or_label["ngram_label"]
            new_ex["chunk_label"] = info_or_label["chunk_label"]
        # ---------------------------------------------------------------------------
        new_examples.append(new_ex)
        
    logger.info('Delete Overlen Keyphrase (length > 5): %d (overlen / total = %.2f' 
                % (overlen_num, float(overlen_num / len(examples) * 100)) + '%)')
    return new_examples


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def bert2joint_preprocessor_chinese(examples, tokenizer, max_token, pretrain_model, mode, max_phrase_words, stem_flag=False):
    """
    对 中文语料[训练集] 进行预处理
    :param examples: 原始语料
    :param tokenizer: Bert分词器
    :param max_token: 最大 token 数
    :param pretrain_model:
    :param mode:
    :param max_phrase_words: 关键词组中允许的最多词数
    :param stem_flag: 词干化(对中文不适用)
    :return:
    """
    logger.info('start preparing (%s) features for bert2joint (%s) ...' % (mode, pretrain_model))

    overlen_num = 0
    new_examples = []
    for idx, ex in enumerate(tqdm(examples)):
        # tokenize
        tokenize_output = loader_utils.tokenize_for_bert(doc_words=ex['text_tokens'], tokenizer=tokenizer)
        # max_token
        if len(tokenize_output['tokens']) < max_token:
            max_word = max_token
        else:
            max_word = tokenize_output['tok_to_orig_index'][max_token - 1] + 1

        new_ex = {}
        # tokenize_output['tokens']: 使用 Bert分词器对预处理时的分词结果(词粒度) 继续分词(子词粒度), 而得所有的 sub_tokens
        new_ex['tokens'] = tokenize_output['tokens'][:max_token]
        # valid_mask, 对于所有 token 对应的 sub_tokens, 如果 sub_token为token的第一个子词,记为1, 反之,记为 0;
        # 因此 valid_mask 中的 1 的个数即为 词数
        new_ex['valid_mask'] = tokenize_output['valid_mask'][:max_token]
        new_ex['doc_words'] = ex['text_tokens'][:max_word]

        assert len(new_ex['tokens']) == len(new_ex['valid_mask'])
        assert sum(new_ex['valid_mask']) == len(new_ex['doc_words'])

        # ---------------------------------------------------------------------------
        parameter = {"doc_words": new_ex['doc_words'],
                     "max_phrase_words": max_phrase_words, "stem_flag": stem_flag}
        # ---------------------------------------------------------------------------
        if mode == 'train':
            # parameter["keyphrases"] = ex['keyphrases']
            parameter["start_end_pos"] = ex["keyword_loc"]
        # ---------------------------------------------------------------------------
        # obtain gram info and label
        info_or_label = get_ngram_info_label_chinese(**parameter)

        new_ex["phrase_list"] = info_or_label["tot_phrase_list"]
        new_ex["mention_lists"] = info_or_label["tot_mention_list"]

        if info_or_label["overlen_flag"]:
            overlen_num += 1
        # ---------------------------------------------------------------------------
        if mode == 'train':
            if not info_or_label["ngram_label"]:
                continue
            new_ex["keyphrases"] = ex["keyphrases"]
            new_ex["ngram_label"] = info_or_label["ngram_label"]
            new_ex["chunk_label"] = info_or_label["chunk_label"]
        # ---------------------------------------------------------------------------
        new_examples.append(new_ex)

    logger.info('Delete Overlen Keyphrase (length > 5): %d (overlen / total = %.2f'
                % (overlen_num, float(overlen_num / len(examples) * 100)) + '%)')

    return new_examples


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def bert2joint_preprocessor_test(example, tokenizer, max_token, mode, max_phrase_words, stem_flag=False):
    """
    对 中文语料[训练集] 进行预处理
    :param example:
    :param tokenizer:
    :param max_token:
    :param mode:
    :param max_phrase_words:
    :param stem_flag:
    :return:
    """
    overlen_num = 0
    # tokenize
    tokenize_output = loader_utils.tokenize_for_bert(doc_words=example['text_tokens'], tokenizer=tokenizer)
    # max_token
    if len(tokenize_output['tokens']) < max_token:
        max_word = max_token
    else:
        max_word = tokenize_output['tok_to_orig_index'][max_token - 1] + 1

    new_example = {}
    # tokenize_output['tokens']: 使用 Bert分词器对预处理时的分词结果(词粒度) 继续分词(子词粒度), 而得所有的 sub_tokens
    new_example['tokens'] = tokenize_output['tokens'][:max_token]
    # valid_mask, 对于所有 token 对应的 sub_tokens, 如果 sub_token为token的第一个子词,记为1, 反之,记为 0;
    # 因此 valid_mask 中的 1 的个数即为 词数
    new_example['valid_mask'] = tokenize_output['valid_mask'][:max_token]
    new_example['doc_words'] = example['text_tokens'][:max_word]

    assert len(new_example['tokens']) == len(new_example['valid_mask'])
    assert sum(new_example['valid_mask']) == len(new_example['doc_words'])

    # ---------------------------------------------------------------------------
    parameter = {"doc_words": new_example['doc_words'],
                 "max_phrase_words": max_phrase_words, "stem_flag": stem_flag}
    # ---------------------------------------------------------------------------
    if mode == 'train':
        # parameter["keyphrases"] = ex['keyphrases']
        parameter["start_end_pos"] = example["keyword_loc"]
    # ---------------------------------------------------------------------------
    # obtain gram info and label
    info_or_label = get_ngram_info_label_chinese(**parameter)

    new_example["phrase_list"] = info_or_label["tot_phrase_list"]
    new_example["mention_lists"] = info_or_label["tot_mention_list"]

    if info_or_label["overlen_flag"]:
        overlen_num += 1
    # ---------------------------------------------------------------------------
    if mode == 'train':
        if not info_or_label["ngram_label"]:
            return {}
        new_example["keyphrases"] = example["keyphrases"]
        new_example["ngram_label"] = info_or_label["ngram_label"]
        new_example["chunk_label"] = info_or_label["chunk_label"]

    return new_example


# ********************************************************************************************************
# ********************************************************************************************************
def bert2joint_converter(index, ex, tokenizer, mode, max_phrase_words):
    '''
    对数据集进行迭代时 (for 循环), 每次迭代取到的数据 会首先调用此方法 (__get_item__)处理每次迭代的数据
    convert each batch data to tensor ; add [CLS] [SEP] tokens ;
    return:
        index: 当前样本在数据集中的序号,
        src_tensor: 子词 --> id,
        valid_mask: [0] + valid_mask + [0],
        mention_lists: ,
        orig_doc_len: ,
        max_phrase_words,
        tot_phrase_len
    '''
    src_tokens = [BOS_WORD] + ex['tokens'] + [EOS_WORD]
    valid_ids = [0] + ex['valid_mask'] + [0]
    # convert_tokens_to_ids: 将 tokens 转为 ids (使用词表)
    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    valid_mask = torch.LongTensor(valid_ids)
    
    mention_lists = ex['mention_lists']
    orig_doc_len = sum(valid_ids)

    if mode == 'train':
        label = torch.LongTensor(ex['ngram_label'])
        chunk_label = torch.LongTensor(ex['chunk_label'])
        return index, src_tensor, valid_mask, mention_lists, orig_doc_len, max_phrase_words, label, chunk_label
    
    else:
        tot_phrase_len = len(ex["phrase_list"])
        return index, src_tensor, valid_mask, mention_lists, orig_doc_len, max_phrase_words, tot_phrase_len


# 在 预训练语料 (cached) 之上会继续进行一定的处理 才 真正喂入模型 进行训练,
# 对应 torch.dataloader 中的参数 collate_fn
# 具体顺序是: for循环  ---> __get_item__ (即 bert2joint_converter) ---> batchify_bert2joint_features_for_train
def batchify_bert2joint_features_for_train(batch):
    ''' train dataloader & eval dataloader .
    batch 内 样本的格式:
        - index : 当前样本在整个数据集中的 index
        - src_tensor: torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))  子词分词结果 (经过截断)
        - valid_mask: torch.LongTensor(valid_ids)  子词的位置编码, "1"表示该子词在原分词中是首个子词, 反之, 用 "0" 表示
        - mention_lists:  mention_lists 所有潜在关键词的 list
        - orig_doc_len:  sum(valid_ids) 文章中的 词数 (src_tokens 中包含的 分词个数)
        - label: torch.LongTensor(ngram_label)
        - chunk_label: torch.LongTensor(chunk_label)
    '''
    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    mention_mask = [ex[3] for ex in batch]
    doc_word_lens = [ex[4] for ex in batch] # 当前 Batch 中的每一样本的词数
    max_phrase_words = [ex[5] for ex in batch][0]
    
    # label
    label_list = [ex[6] for ex in batch] # different ngrams numbers
    chunk_list = [ex[7] for ex in batch] # whether is a chunk phrase

    bert_output_dim = 768
    # max_word_len: 当前 batch 中最长文章的字数
    max_word_len = max([word_len for word_len in doc_word_lens]) # word-level
    
    # ---------------------------------------------------------------
    # [1] [2] src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    # segment_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    
    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
        
    # ---------------------------------------------------------------
    # [3] valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)
        
    # ---------------------------------------------------------------
    # [4] active mention mask : for n-gram (original)
    
    max_ngram_length = sum([max_word_len-n for n in range(max_phrase_words)])
    chunk_mask = torch.LongTensor(len(docs), max_ngram_length).fill_(-1)

    for batch_i, word_len in enumerate(doc_word_lens):
        # 第 batch_i 篇文章需要 padding的长度
        pad_len = max_word_len - word_len

        batch_mask = []
        for n in range(max_phrase_words):
            ngram_len = word_len - n

            if ngram_len > 0:
                assert len(mention_mask[batch_i][n]) == ngram_len
                gram_list = mention_mask[batch_i][n] + [-1 for _ in range(pad_len)] # -1 for padding
            else:
                gram_list = [-1 for _ in range(max_word_len-n)]

            batch_mask.extend(gram_list)
        chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))
        
    # ---------------------------------------------------------------
    # [4] active mask : for n-gram
    max_diff_gram_num = (1 + max([max(_mention_mask[-1]) for _mention_mask in mention_mask]))
    active_mask = torch.BoolTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1) # Pytorch Version = 1.1.3
#     active_mask = torch.ByteTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1) # Pytorch Version = 1.1.0
    for gram_ids in range(max_diff_gram_num):
        tmp = torch.where(chunk_mask==gram_ids, 
                          torch.LongTensor(len(docs), max_ngram_length).fill_(0), 
                          torch.LongTensor(len(docs), max_ngram_length).fill_(1)) # shape = (batch_size, max_ngram_length) # 1 for pad
        for batch_id in range(len(docs)):
            active_mask[batch_id][gram_ids].copy_(tmp[batch_id])

    # -------------------------------------------------------------------
    # [5] label : for n-gram
    max_diff_grams_num = max([label.size(0) for label in label_list])
    ngram_label = torch.LongTensor(len(label_list), max_diff_grams_num).zero_()
    for batch_i, label in enumerate(label_list):
        ngram_label[batch_i, :label.size(0)].copy_(label)

    # -------------------------------------------------------------------
    # [6] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim) 
    
    # -------------------------------------------------------------------
    # [7] Chunk Lable : 
    max_chunks_num = max([chunks.size(0) for chunks in chunk_list])        
    chunk_label = torch.LongTensor(len(chunk_list), max_chunks_num).fill_(-1)
    for batch_i, chunks in enumerate(chunk_list):
        chunk_label[batch_i, :chunks.size(0)].copy_(chunks)

    return input_ids, input_mask, valid_ids, active_mask, valid_output, ngram_label, chunk_label, chunk_mask, ids


def batchify_bert2joint_features_for_test(batch):
    ''' test dataloader for Dev & Public_Valid.'''
    
    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    mention_mask = [ex[3] for ex in batch]
    doc_word_lens = [ex[4] for ex in batch]
    max_phrase_words = [ex[5] for ex in batch][0]
    
    phrase_list_lens = [ex[6] for ex in batch]
    
    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens]) # word-level
    
    # ---------------------------------------------------------------
    # [1] [2] src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    # segment_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    
    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
        
    # ---------------------------------------------------------------
    # [3] valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # [4] active mention mask : for n-gram (original)
    
    max_ngram_length = sum([max_word_len-n for n in range(max_phrase_words)])
    chunk_mask = torch.LongTensor(len(docs), max_ngram_length).fill_(-1)
    
    for batch_i, word_len in enumerate(doc_word_lens):
        pad_len = max_word_len - word_len
        
        batch_mask = []
        for n in range(max_phrase_words):
            ngram_len = word_len - n
            if ngram_len > 0:
                assert len(mention_mask[batch_i][n]) == ngram_len
                gram_list = mention_mask[batch_i][n] + [-1 for _ in range(pad_len)] # -1 for padding
            else:
                gram_list = [-1 for _ in range(max_word_len-n)]
            batch_mask.extend(gram_list)
        chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))
        
    # ---------------------------------------------------------------
    # [4] active mask : for n-gram
    max_diff_gram_num = (1 + max([max(_mention_mask[-1]) for _mention_mask in mention_mask]))
    active_mask = torch.BoolTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1)
#     active_mask = torch.ByteTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1) # Pytorch Version = 1.1.0
    for gram_ids in range(max_diff_gram_num):
        tmp = torch.where(chunk_mask==gram_ids, 
                          torch.LongTensor(len(docs), max_ngram_length).fill_(0), 
                          torch.LongTensor(len(docs), max_ngram_length).fill_(1)) # shape = (batch_size, max_ngram_length) # 1 for pad
        for batch_id in range(len(docs)):
            active_mask[batch_id][gram_ids].copy_(tmp[batch_id])

    # -------------------------------------------------------------------
    # [5] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)

    return input_ids, input_mask, valid_ids, active_mask, valid_output, phrase_list_lens, ids
