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
    """
    First check keyphrase mentions index is same ;
    Then set keyprhase ngrams = +1  and other phrase candidates = -1 .
    :param filter_positions: 经过滤后的 关键词起始位置
    :param tot_mention_list: 每一滑动窗口下, 经过编码后的文章
    :param differ_phrase_num: 在所有滑动窗口下, 出现的所有词组集合长度
    :return:
        ngram_label: 在 tot_phrase_list 中, 若 某一候选关键词 被标记为 关键词, 记为 1, 反之, 记为 -1
        chunk_label: 遍历 tot_mention_list, 若 某一滑动窗口下, 某一候选关键词 被标记为 关键词, 记为 1, 反之, 记为 0
    """
    ngram_label = [-1 for _ in range(differ_phrase_num)]
    chunk_label_list = [[0] * len(tot_mention_list[i]) for i in range(len(tot_mention_list))]

    for i, positions in enumerate(filter_positions):
        try:
            for s, e in positions:
                chunk_label_list[e-s-1][s] = 1 # 关键词所属的 滑动窗口, 对应的词组 ===> 1
                key_index = tot_mention_list[e-s-1][s]
                ngram_label[key_index] = 1 # 关键词对应的 词组 ===> 1
        except:
            print("position: ", positions, filter_positions)
            print("chunk_label_list", chunk_label_list)
            input()

    # flat chunk label
    chunk_label = [_chunk for chunks in chunk_label_list for _chunk in chunks]
    
    # keep have more than one positive and one negtive
    if (1 in ngram_label) and (-1 in ngram_label) and (1 in chunk_label) and (0 in chunk_label):
        return ngram_label, chunk_label
    else:
        return None, None


def get_ngram_features(doc_words, max_gram, stem_flag=False):
    tot_phrase_list = []
    phrase2index = {}

    tot_mention_list = []

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


def get_ngram_features_chinese(doc_words, max_gram):
    """
    对所有候选关键词, 根据其出现的顺序, 作为其编码（若第一个出现的词是“中国”，则“中国”的编码为 “0”）, 并据此编码文章.
    :param doc_words: 分词结果 (并非 Bert)
    :param max_gram: max_phrase_words, 关键词最大词数, 预设值, 默认为 5
    :param stem_flag: 是否词干化, 对中文无意义

    :return: a dict:
        {
            "tot_phrase_list": tot_phrase_list,     # 记录所有滑动窗口下, 出现的词组集合, 按其出现的顺序记录, 存于 list
            "tot_mention_list": tot_mention_list    # 每一滑动窗口下, 按照映射 phrase2index 编码词组
        }
    """
   
    # 记录所有滑动窗口下 出现的 词组 的 集合, 按其出现的顺序记录. 列表中 每一元素 为词组本身, 元素的 下标作为 其 index
    tot_phrase_list = []

    # 所有滑动窗口下出现的 词组 为键, 其出现的顺序（index）为值, 例如: {"中国": 0, "人民": 1, "银行": 2}
    # 事实上, phrase2index 中的键即为 tot_phrase_list 的每个元素, 值即为为元素的下标
    phrase2index = {}

    # 每一滑动窗口下, 按照映射 phrase2index 编码所有出现的词组, 例如 [["编码滑动窗口为 1"], ["编码滑动窗口为 2"], ...]
    tot_mention_list = []

    # 所有滑动窗口下, 出现的词组个数之和, 即 tot_mention_list 中每一元素的长度之和
    gram_num = 0

    for n in range(1, max_gram+1):
        valid_length = (len(doc_words) - n + 1)
        if valid_length < 1:
            break

        _ngram_list = []
        _mention_list = []
        for i in range(valid_length):
            gram_num += 1
            # 取指定滑动窗口(n)下的 gram, 另外中文的词之间不需要加空格!(对比 get_ngram_features)
            n_gram = "".join(doc_words[i:i+n]).lower()
            index = loader_utils.whether_existing(n_gram, phrase2index, tot_phrase_list)
            # 在当前滑动窗口下编码文章, index 为 当前滑动窗口下的 "词"(即词组) 的 编码.
            _mention_list.append(index)
            _ngram_list.append(n_gram)

        tot_mention_list.append(_mention_list)

    assert len(tot_phrase_list) > 0
    assert (len(tot_phrase_list) - 1) == max(tot_mention_list[-1])
    assert sum([len(_mention_list) for _mention_list in tot_mention_list]) == gram_num
    return {"tot_phrase_list": tot_phrase_list, "tot_mention_list": tot_mention_list, 'phrase2index': phrase2index}


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


def get_ngram_info_label(doc_words, max_phrase_words, stem_flag, start_end_pos=None):
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
def get_ngram_info_label_chinese(doc_words, max_phrase_words, start_end_pos=None):
    """
           根据 关键词 标记 对应的 词组

    :param doc_words: 经截断后的分词
    :param max_phrase_words: 关键词最大词数, 默认为 5
    :param stem_flag:
    :param start_end_pos: 关键词在 分词中 首次出现时的起始位置
    :return: a dict:
        {
         "overlen_flag": False,
         "ngram_label": [] ===> 若 tot_phrase_list（所有候选关键词组成的集合） 中的 某一词组 被标记为关键词, 记为 1; 反之, 记为 -1
         "chunk_label": [] ===> 遍历 tot_mention_list, 对于每一窗口下的 某一词组, 若被标记为关键词, 记为 1, 反之, 记为 0, 最后将结果拉平
        }
    """
    returns = {"overlen_flag": False, "ngram_label": None, "chunk_label": None}
    # ----------------------------------------------------------------------------------------
    feature = get_ngram_features_chinese(doc_words=doc_words, max_gram=max_phrase_words)

    returns["tot_phrase_list"] = feature["tot_phrase_list"]     # 在所有滑动窗口下, 出现的所有候选关键词的集合, 存储于 list 中
    returns["tot_mention_list"] = feature["tot_mention_list"]   # 每一滑动窗口下，所有候选关键词被编码后的结果

    # ----------------------------------------------------------------------------------------
    if start_end_pos is not None:
        # 将 视野以外的 关键词丢掉, "视野以外"指的是 在文中第 512 个字符之后 才 首次出现的关键词
        filter_positions = loader_utils.limit_scope_length(start_end_pos, len(doc_words), max_phrase_words)

        # check over_length
        if len(filter_positions) != len(start_end_pos):
            returns["overlen_flag"] = True

        if len(filter_positions) > 0:
            #  "ngram_label": [] ===> 若 tot_phrase_list 中的 某一词组 被标记为关键词, 记为 1; 反之, 记为 -1
            # "chunk_label": [] ===> 遍历 tot_mention_list, 对于每一滑窗下的 某一词组, 若被标记为关键词, 记为 1, 反之, 记为 0, 最后将结果拉平
            returns["ngram_label"], returns["chunk_label"] = convert_to_label(**{"filter_positions": filter_positions,
                                                                                 "tot_mention_list": feature[
                                                                                     "tot_mention_list"],
                                                                                 "differ_phrase_num": len(
                                                                                     feature["tot_phrase_list"])})
        else:
            returns["ngram_label"] = None
            returns["chunk_label"] = None

    return returns


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
def bert2joint_preprocessor_chinese(examples, tokenizer, max_token, pretrain_model, mode, max_phrase_words, keyphrase_in_test_data=True):
    logger.info('start preparing (%s) features for bert2joint (%s) ...' % (mode, pretrain_model))

    overlen_num = 0
    new_examples = []

    for idx, ex in enumerate(tqdm(examples)):
        # tokenize
        tokenize_output = loader_utils.tokenize_for_bert(doc_words=ex['text_tokens'], tokenizer=tokenizer)
        # max_word 用于 截断 text_tokens
        if len(tokenize_output['sub_tokens']) < max_token:
            max_word = max_token  # len(tokens) <= len(sub_words), 当 后者 < max_token时, 前者也必 < max_token
        else:
            max_word = tokenize_output['tok_to_orig_index'][max_token - 1] + 1

        new_ex = {}
        new_ex['doc_id'] = ex['doc_id']
        # tokenize_output['tokens']: 使用 Bert分词器对预处理时的分词结果(词粒度) 继续分词(子词粒度), 而得所有的 sub_tokens
        new_ex['sub_tokens'] = tokenize_output['sub_tokens'][:max_token]
        # valid_mask, 对于所有 token 对应的 sub_tokens, 如果 sub_token为 token 的第一个子词,记为1, 反之,记为 0;
        # 因此 valid_mask 中的 1 的个数即为 词数
        new_ex['valid_mask'] = tokenize_output['valid_mask'][:max_token]
        # 使用 max_word 截断 text_tokens
        new_ex['doc_words'] = ex['text_tokens'][:max_word]

        assert len(new_ex['sub_tokens']) == len(new_ex['valid_mask'])
        assert sum(new_ex['valid_mask']) == len(new_ex['doc_words'])

        # ---------------------------------------------------------------------------
        parameter = {"doc_words": new_ex['doc_words'],
                     "max_phrase_words": max_phrase_words
                     }

        # ---------------------------------------------------------------------------
        if ex.get("start_end_pos"):
            parameter["keyphrases"] = ex['keyphrases']
            parameter["start_end_pos"] = ex['start_end_pos']

        # ---------------------------------------------------------------------------
        # 基于 截断后的分词 (new_ex['doc_words']) 生成 候选关键词集合 与 各滑动窗口下的文章编码结果
        info_or_label = get_ngram_info_label_chinese(**parameter)

        new_ex["phrase_list"] = info_or_label["tot_phrase_list"]        # 所有候选关键词组成的集合
        new_ex["mention_lists"] = info_or_label["tot_mention_list"]     # 每一滑动窗口下，对所有词组进行编码后的结果

        if info_or_label["overlen_flag"]:
            overlen_num += 1
        # ---------------------------------------------------------------------------
        if mode == 'train':
            # 如果 所有关键词 在 所有滑动窗口 中均为出现, 则所有关键词均为离线关键词, 此文章 不可用
            if not info_or_label["ngram_label"]:
                continue

            new_ex["keyphrases"] = ex["keyphrases"]
            new_ex["ngram_label"] = info_or_label["ngram_label"]
            new_ex["chunk_label"] = info_or_label["chunk_label"]

        elif keyphrase_in_test_data:
            keyphrases = ex["keyphrases"]
            if keyphrases:
                new_ex["keyphrases"] = keyphrases
        # ---------------------------------------------------------------------------
        new_examples.append(new_ex)

    logger.info('Delete Overlen Keyphrase (length > 5): %d (overlen / total = %.2f'
                % (overlen_num, float(overlen_num / len(examples) * 100)) + '%)')

    return new_examples


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def bert2joint_preprocessor_single_chinese(example, tokenizer, max_sub_token, mode, max_phrase_words):
    """
    :param example: 预处理完的语料
    :param tokenizer: Bert 分词器
    :param max_token: 子词(单字)最大长度(针对 Bert)
    :param mode:
    :param max_phrase_words: 关键词最大长度(关键词分词数)
    :param stem_flag: 是否词干化, 中文不涉及
    :return:
    """
    overlen_num = 0
    # tokenize
    tokenize_output = loader_utils.tokenize_for_bert(doc_words=example['text_tokens'], tokenizer=tokenizer)
    # tokenize_output['sub_tokens']: 经 Bert 分词器分词后(子词化), 所有的 sub_tokens
    if len(tokenize_output['sub_tokens']) < max_sub_token:
        max_word = max_sub_token
    else:
        # 第 max_token 个 sub_token 属于 第几个 token
        max_word = tokenize_output['tok_to_orig_index'][max_sub_token - 1] + 1

    new_example = {'doc_id': example['doc_id'],
                   'sub_tokens': tokenize_output['sub_tokens'][:max_sub_token],  # 使用 tokenizer 子词化 tokens (词粒度 -> 字粒度) 而得的 sub_tokens
                   'valid_mask': tokenize_output['valid_mask'][:max_sub_token],  # 对所有 token 编码其 sub_tokens, 如果 sub_token为 token 的第一个子词,记为 1, 反之,记为 0;
                   'doc_words': example['text_tokens'][:max_word]  # 经截断后的 tokens(保证截断后 sub_tokens <= max_token)
                   }

    assert len(new_example['sub_tokens']) == len(new_example['valid_mask'])
    assert sum(new_example['valid_mask']) == len(new_example['doc_words'])

    # ---------------------------------------------------------------------------
    parameter = {"doc_words": new_example['doc_words'],
                 "max_phrase_words": max_phrase_words
                 }
    # ---------------------------------------------------------------------------
    if mode == 'train':
        parameter["start_end_pos"] = example["keyphrase_loc"]
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
        mention_lists: # 每一滑动窗口下, 对所有词组编码后的结果,
        orig_doc_len:  # 经过截断后, 文章的长度,
        max_phrase_words: # 候选关键词支持的最大分词数,
        tot_phrase_len: # 所有候选关键词(对应的编码)组成的集合
    '''
    src_tokens = [BOS_WORD] + ex['sub_tokens'] + [EOS_WORD]
    valid_ids = [0] + ex['valid_mask'] + [0]
    # convert_tokens_to_ids: 将 tokens 转为 ids (使用词表)
    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    valid_mask = torch.LongTensor(valid_ids)
    
    mention_lists = ex['mention_lists']
    orig_doc_len = sum(valid_ids)
    tot_phrase_len = len(ex["phrase_list"])

    if mode == 'train':
        label = torch.LongTensor(ex['ngram_label'])
        chunk_label = torch.LongTensor(ex['chunk_label'])
        return index, src_tensor, valid_mask, mention_lists, orig_doc_len, max_phrase_words, label, chunk_label, tot_phrase_len
    
    else:
        return index, src_tensor, valid_mask, mention_lists, orig_doc_len, max_phrase_words, tot_phrase_len


# 在模型进行训练/推断时, 首先对预处理语料 (cached_....) 会进一步做处理, 然后才 真正喂入模型,
# 具体执行顺序是:
#   for循环  ---> __get_item__ (即 bert2joint_converter) ---> dataloader.collate_fn(batchify_bert2joint_features_for_train/test)
def batchify_bert2joint_features_for_train(batch):
    '''
    对应 torch.dataloader 中的参数 collate_fn.
    batch 内 样本的格式:
        - index: 当前样本在整个数据集中的 index
        - src_tensor: Bert分词器的分词结果. torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
        - valid_mask: 子词的位置编码, "1"表示该子词在原分词中是首个子词, 反之, 用 "0" 表示. torch.LongTensor(valid_ids)
        - mention_mask: 即 ex["mention_lists"], 每一滑动窗口下, 文章编码的结果 [词的编码方式为其在文章中出现的顺序]
        - orig_doc_len: sum(valid_ids) 文章中的 词数 (src_tokens 中包含的 分词个数)
        - max_phrase_words: 候选关键词中包含的最大分词数
        - label: 若 tot_phrase_list（所有候选关键词组成的集合） 中的 某一词组 被标记为关键词, 记为 1; 反之, 记为 -1
        - chunk_label: 遍历 tot_mention_list, 对于每一窗口下的 某一词组, 若被标记为关键词, 记为 1, 反之, 记为 0, 最后将结果拉平
        - phrase_lens: len(tot_phrase_list)

    :returns:
        - input_ids: src_tensor 经过 0-padding 后的结果
        - input_mask: attention_mask 经过 0-padding 后的结果
        - valid_ids: vaild_mask 经过 0-padding 后的结果
        - active_mask: batch中的所有词(的index)作为词典, 记录每个词在每篇文章中出现的位置, 出现的位置记为 1, 其余记为 0
        - valid_output: 全零矩阵, shape=[len(docs), max_word_len, bert_output_dim], 用于 训练/推断时 保存所有分词的 首个子词 的隐状态
        - ngram_label: 对 label 进行 0-padding 后的结果 后的结果
        - chunk_label: 对 chunk_label 进行 -1-padding 后的结果
        - chunk_mask:  对 mention_mask 进行 -1-padding 后的结果, 再经过 flatten
        - ids: 文章 在整个 dataset中的 index
    '''
    ids = [ex[0] for ex in batch]                   # 文章 在整个 dataset中的 index
    docs = [ex[1] for ex in batch]                  # 文章经 bert 编码后的结果, 前后加 两个字符 512
    valid_mask = [ex[2] for ex in batch]            # 位词中首个字符的位置编码, 前后加 两个字符 512
    mention_mask = [ex[3] for ex in batch]          # 每一滑动窗口下, 所有的候选关键词被编码后的结果
    doc_word_lens = [ex[4] for ex in batch]         # 当前 Batch 中的每一样本的分词数
    max_phrase_words = [ex[5] for ex in batch][0]   # 当前配置的 关键词支持的最大分词数

    # label
    label_list = [ex[6] for ex in batch]            # 上一环节中的 ex["ngram_label"]
    chunk_list = [ex[7] for ex in batch]            # 上一环节中的 ex["chunk_label"], 生成于 convert_to_label
    phrase_list_lens = [ex[8] for ex in batch]      # 文章对应的候选关键词集合的元素个数

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
                gram_list = mention_mask[batch_i][n] + [-1 for _ in range(pad_len)] # -1 for padding, , gram_list 长度 = orig_doc_len - n
            else:
                gram_list = [-1 for _ in range(max_word_len-n)]

            batch_mask.extend(gram_list)
        chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))
        
    # ---------------------------------------------------------------
    # [4] active mask : for n-gram
    max_diff_gram_num = (1 + max([max(_mention_mask[-1]) for _mention_mask in mention_mask]))
    assert max_diff_gram_num == max(phrase_list_lens)

    # 该对于当前词典中的每一字符, 标记在文章中出现的位置, 即对于长度为 max_diff_ngram_length 的 全1 list, 将对应位置的字符标记为 0
    active_mask = torch.BoolTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1)
    for gram_ids in range(max_diff_gram_num):
        tmp = torch.where(chunk_mask == gram_ids,
                          torch.LongTensor(len(docs), max_ngram_length).fill_(0), 
                          torch.LongTensor(len(docs), max_ngram_length).fill_(1))
        for batch_id in range(len(docs)):
            active_mask[batch_id][gram_ids].copy_(tmp[batch_id])

    # -------------------------------------------------------------------
    # [5] label : for n-gram
    max_diff_grams_num = max(phrase_list_lens)  # 当前 batch 个 文章的 词典
    # max_diff_grams_num = max([label.size(0) for label in label_list])

    ngram_label = torch.LongTensor(len(label_list), max_diff_grams_num).zero_()
    for batch_i, label in enumerate(label_list):
        ngram_label[batch_i, :label.size(0)].copy_(label)

    # -------------------------------------------------------------------
    # [6] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim) 
    
    # -------------------------------------------------------------------
    # [7] Chunk label: 对 chunk_list padding, 这里的 chunk_list 已经 flat
    # chunk_list 生成于 convert_to_label
    max_chunks_num = max([chunks.size(0) for chunks in chunk_list])        
    chunk_label = torch.LongTensor(len(chunk_list), max_chunks_num).fill_(-1)
    for batch_i, chunks in enumerate(chunk_list):
        chunk_label[batch_i, :chunks.size(0)].copy_(chunks)

    return input_ids, input_mask, valid_ids, active_mask, valid_output, ngram_label, chunk_label, chunk_mask, ids


def batchify_bert2joint_features_for_test(batch):
    ids = [ex[0] for ex in batch]                   # 文章 在整个 dataset中的 index
    docs = [ex[1] for ex in batch]                  # 文章经 bert 编码后的结果, 前后加 两个字符 512
    valid_mask = [ex[2] for ex in batch]            # 位词中首个字符的位置编码, 前后加 两个字符 512
    mention_mask = [ex[3] for ex in batch]          # 每一滑动窗口下, 所有的候选关键词被编码后的结果
    doc_word_lens = [ex[4] for ex in batch]         # 当前 Batch 中的每一样本的分词数
    max_phrase_words = [ex[5] for ex in batch][0]   # 当前配置的 关键词支持的最大分词数
    phrase_list_lens = [ex[6] for ex in batch]      # 文章对应的候选关键词集合的元素个数

    bert_output_dim = 768

    max_word_len = max([word_len for word_len in doc_word_lens])    # 当前 batch 中文章的最大分词数

    # ---------------------------------------------------------------
    # [1] [2] src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])                 # 当前 batch篇文章中, 最大的字符数
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    # padding
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
    # [4] chunk_mask
    max_ngram_length = sum([max_word_len-n for n in range(max_phrase_words)])
    # mention_mask 尚未 flatten, 现使用 -1 padding 处理 mention_mask, 每篇文章的长度均为 max_ngram_length, 并将结果 flatten
    # 最终结果保存为 chunk_mask
    chunk_mask = torch.LongTensor(len(docs), max_ngram_length).fill_(-1)

    # 再对 chunk_mask 逆 flatten
    for batch_i, word_len in enumerate(doc_word_lens):
        pad_len = max_word_len - word_len
        
        batch_mask = []
        for n in range(max_phrase_words):
            ngram_len = word_len - n
            # 每一滑动窗口下, 文章的编码结果长度(ngram_len)将被 Padding 成 ngram_len + pad_len = max_word_len - n
            if ngram_len > 0:
                assert len(mention_mask[batch_i][n]) == ngram_len
                gram_list = mention_mask[batch_i][n] + [-1 for _ in range(pad_len)]  # -1 for padding
            else:
                gram_list = [-1 for _ in range(max_word_len-n)]
            batch_mask.extend(gram_list)

        # 最终 bacth_mask 的长度 == sum([max_word_len-n for n in range(max_phrase_words)])
        chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))

    max_diff_gram_num = max(phrase_list_lens)  # 当前 batch篇 文章 对应的 "词典"

    # activate_mask: shape = [len(docs), max_diff_gram_num, max_ngram_length]
    # 对于 当前batch 所组成的 "词典" 中 的 每个"词",
    # 标记其在 当前batch中的 所有文章中 出现的位置, 即对于长度为 max_diff_ngram_length 的 全1 list, 将对应位置的元素标记为 0
    active_mask = torch.BoolTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1)

    for gram_ids in range(max_diff_gram_num):
        tmp = torch.where(chunk_mask == gram_ids,
                          torch.LongTensor(len(docs), max_ngram_length).fill_(0), 
                          torch.LongTensor(len(docs), max_ngram_length).fill_(1))
        for batch_id in range(len(docs)):
            active_mask[batch_id][gram_ids].copy_(tmp[batch_id])

    # -------------------------------------------------------------------
    # [5] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)

    return input_ids, input_mask, valid_ids, active_mask, valid_output, phrase_list_lens, ids
