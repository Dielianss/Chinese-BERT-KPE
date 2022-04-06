import json
from bertkpe.transformers import BertTokenizer, BertModel, BertConfig
from bertkpe.networks import BertForChunkTFRanking
import torch


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


def get_ngram_features_chinese(doc_words, max_gram):
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

    for n in range(1, max_gram + 1):
        valid_length = (len(doc_words) - n + 1)
        if valid_length < 1:
            break

        _ngram_list = []
        _mention_list = []
        for i in range(valid_length):
            gram_num += 1
            # 中文的 子词之间不需要 加空格!
            n_gram = "".join(doc_words[i:i + n]).lower()
            index = whether_existing(n_gram, phrase2index, tot_phrase_list)
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
            chunk_label_list[e - s][s] = 1
            key_index = tot_mention_list[e - s][s]
            ngram_label[key_index] = 1

    # flat chunk label
    chunk_label = [_chunk for chunks in chunk_label_list for _chunk in chunks]

    # keep have more than one positive and one negtive
    if (1 in ngram_label) and (-1 in ngram_label) and (1 in chunk_label) and (0 in chunk_label):
        return ngram_label, chunk_label
    else:
        return None, None


# -------------------------------------------------------------------------------------------
def get_ngram_info_label_chinese(doc_words, max_phrase_words, start_end_pos=None):
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
    feature = get_ngram_features_chinese(doc_words=doc_words, max_gram=max_phrase_words)
    # 在所有可能的词组长度下, 文档对应的潜在关键词种类
    returns["tot_phrase_list"] = feature["tot_phrase_list"]
    # 在所有可能的词组长度下, 每一长度下, 所有潜在的关键词的编码
    returns["tot_mention_list"] = feature["tot_mention_list"]

    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    if start_end_pos is not None:
        filter_positions = limit_scope_length(start_end_pos, len(doc_words), max_phrase_words)

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


class KPEModel():
    def __init__(self, pretrained_kpe_path: str):
        self.device = torch.device("cuda", 1)
        self.kpe_model, self.tokenizer = self.load_checkpoint(pretrained_kpe_path)
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
        self.BOS_WORD = '[CLS]'
        self.EOS_WORD = '[SEP]'
        self.DIGIT_WORD = 'DIGIT'
        self.max_token_length = 510
        self.max_phrase_words = 5
        self.max_predicted_keyphrases = 5
        self.bert_output_dim = 768
        self.kpe_model.eval()

    # 0: pre-trained model tokenize: 对应 预处理语料的生成 (cached)
    def tokenize_for_bert(self, doc_words: list):
        """
        将原始语料的分词结果使用 BertTokenizer 子词化
        :param doc_words:
        :return:
        """
        # vaild_mask: 若 token不含 sub_token, 记为 1, 若 token 含有 sub_token, 则首个 sub_token记为 1, 其后 sub_tokens 记为 0
        # vaild_mask中 1的 个数即为 词的总数
        valid_mask = []
        # 将 (预处理时的) 分词结果 (tokens) 使用 Bert 拆分为 sub_tokens, 例如 ["unhappy"] --> ["un", "happy"]
        all_doc_subtokens = []
        # 记录 每一个 sub_tokens 所在的那个 token 在 预处理时的分词结果 (tokens) 中 的序号
        subtokens_to_orig_index = []
        # 记录 预处理时的分词结果 (tokens) 中的 每个 token 中的 首个 sub_token 在 sub_tokens 中的 位置
        orig_to_subtokens_index = []

        tmp_orig_to_tok_index = 0
        for (i, token) in enumerate(doc_words):
            orig_to_subtokens_index.append(len(all_doc_subtokens))
            # sorig_to_tok_index.append(tmp_orig_to_tok_index)
            sub_tokens = self.tokenizer.tokenize(token)
            if len(sub_tokens) < 1:
                sub_tokens = [self.UNK_WORD]
            for num, sub_token in enumerate(sub_tokens):
                tmp_orig_to_tok_index += 1
                subtokens_to_orig_index.append(i)
                all_doc_subtokens.append(sub_token)
                if num == 0:
                    valid_mask.append(1)
                else:
                    valid_mask.append(0)

        # max_token
        if len(all_doc_subtokens) < self.max_token_length:
            max_word = self.max_token_length
        else:
            max_word = subtokens_to_orig_index[self.max_token_length - 1] + 1

        # tokenize_output['tokens']: 使用 Bert分词器对预处理时的分词结果(词粒度) 继续分词(子词粒度), 而得所有的 sub_tokens
        truncated_subtokens = all_doc_subtokens[:self.max_token_length]
        truncated_valid_mask = valid_mask[:self.max_token_length]
        truncated_tokens = doc_words[:max_word]

        # obtain gram info and label
        info_or_label = get_ngram_info_label_chinese(truncated_tokens, self.max_phrase_words)

        return truncated_subtokens, truncated_valid_mask, truncated_tokens, \
               info_or_label["tot_phrase_list"], info_or_label["tot_mention_list"]

    # 对应 for 训练时 对每一迭代数据的处理 (__getitem__)
    def bert2joint_converter(self, truncated_subtokens, truncated_valid_mask, truncated_tokens,
                             phrase_list, mention_list):
        '''
        对数据集进行迭代是 会首先调用此方法处理每次迭代的数据
        convert each batch data to tensor ; add [CLS] [SEP] tokens ;
        '''
        src_tokens = [self.BOS_WORD] + truncated_subtokens + [self.EOS_WORD]
        valid_ids = [0] + truncated_valid_mask + [0]
        # convert_tokens_to_ids: 将 tokens 转为 ids (使用词表)
        src_tensor = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(src_tokens))
        valid_mask = torch.LongTensor(valid_ids)
        # 词数
        orig_doc_len = sum(valid_ids)
        # (所有滑动窗口下的)候选关键词总数
        phrase_list_len = len(phrase_list)
        return src_tensor, valid_mask, mention_list,\
               orig_doc_len, phrase_list_len

    def bert2joint_features_for_test(self, doc, valid_mask, mention_mask, doc_word_lens, phrase_list_len):
        ''' test dataloader for Dev & Public_Valid.'''

        # ---------------------------------------------------------------
        # [1] [2] src tokens tensor
        doc_max_length = doc.size(0)
        input_ids = torch.LongTensor(1, doc_max_length).zero_()
        input_mask = torch.LongTensor(1, doc_max_length).zero_()
        # segment_ids = torch.LongTensor(len(docs), doc_max_length).zero_()

        input_ids[:doc.size(0)].copy_(doc)
        input_mask[:doc.size(0)].fill_(1)

        # ---------------------------------------------------------------
        # [3] valid mask tensor
        valid_max_length = valid_mask.size(0)
        valid_ids = torch.LongTensor(1, valid_max_length).zero_()
        valid_ids[0, :valid_mask.size(0)].copy_(valid_mask)

        # ---------------------------------------------------------------
        # [4] active mention mask : for n-gram (original)
        max_ngram_length = sum([doc_word_lens - n for n in range(self.max_phrase_words)])
        chunk_mask = torch.LongTensor(1, max_ngram_length).fill_(-1)

        for batch_i, word_len in enumerate([doc_word_lens]):
            pad_len = doc_word_lens - word_len

            batch_mask = []
            for n in range(self.max_phrase_words):
                ngram_len = word_len - n
                if ngram_len > 0:
                    assert len(mention_mask[n]) == ngram_len
                    gram_list = mention_mask[n] + [-1 for _ in range(pad_len)]  # -1 for padding
                else:
                    gram_list = [-1 for _ in range(doc_word_lens - n)]
                batch_mask.extend(gram_list)
            chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))

        # ---------------------------------------------------------------
        # [4] active mask : for n-gram
        max_diff_gram_num = (1 + max(mention_mask[-1]))
        active_mask = torch.BoolTensor(1, max_diff_gram_num, max_ngram_length).fill_(1)
        for gram_ids in range(max_diff_gram_num):
            tmp = torch.where(chunk_mask == gram_ids,
                              torch.LongTensor(1, max_ngram_length).fill_(0),
                              torch.LongTensor(1, max_ngram_length).fill_(
                                  1))
            active_mask[0][gram_ids].copy_(tmp[0])

        # -------------------------------------------------------------------
        # [5] Empty Tensor : word-level max_len
        valid_output = torch.zeros(1, doc_word_lens, self.bert_output_dim)

        return input_ids.to(self.device), input_mask.to(self.device), valid_ids.to(self.device),\
               active_mask.to(self.device), valid_output.to(self.device)

    def load_checkpoint(self, filename):
        """
        加载 预训练模型!
        :param filename:
        :return:
        """
        saved_params = torch.load(filename, map_location=lambda storage, loc: storage)
        # 读取已训练模型的参数
        args = saved_params['args']
        state_dict = saved_params['state_dict']

        # 定义 词语的关键词属性, 共有五种
        args.num_labels = 2
        model_config = BertConfig.from_pretrained(args.cache_dir, num_labels=args.num_labels)

        # args.cache_dir: 预训练模型的保存路径
        tokenizer = BertTokenizer.from_pretrained(args.cache_dir)
        kpe_model = BertForChunkTFRanking.from_pretrained(args.cache_dir, config=model_config)
        kpe_model.load_state_dict(state_dict)

        # 指定模型运行的 GPU
        kpe_model.to(self.device)
        return kpe_model, tokenizer

    def decode_n_best_candidates(self, gram_list, score_logits):
        """
        从 所有的候选关键词 中 筛选出最有可能的 TopN 个关键词及对应的 socre
        :param gram_list: 候选关键词 list
        :param score_logits: 所有候选关键词对应的 score
        :return:
        """
        ngrams = [(gram, score.item()) for gram, score in zip(gram_list, score_logits) if len(gram) > 1]
        sorted_ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)
        topN_keyphrases = sorted_ngrams[:self.max_predicted_keyphrases]
        return topN_keyphrases

    def remove_one_token_phase(self, phrases_scores):
        """
        删去只有一个字的候选关键词
        :param phrases_scores:
        :return:
        """
        phrase_list = []
        score_list = []
        for phrase, score, in phrases_scores:
            phrase_list.append(phrase)
            score_list.append(score)
        return phrase_list, score_list

    # 根据 logits 从 phrase_list(候选关键词 list) 中取出最有可能的几个关键词
    def rank2phrase_chinese(self, phrase_list, logits):
        params = {'gram_list': phrase_list,
                  'score_logits': logits}

        n_best_phrases_scores = self.decode_n_best_candidates(**params)

        KP_list = []
        for kp, score in n_best_phrases_scores:
            KP_list.append({"keyowrd": kp, "weight": score})

        return KP_list

    @torch.no_grad()
    def predict(self, doc_words: list):
        # 使用 BertTokenizer 分词
        tokenized = self.tokenize_for_bert(doc_words)
        # 添加 Bert 首尾标识
        convertered = self.bert2joint_converter(*tokenized)
        featured = self.bert2joint_features_for_test(*convertered)
        inputs = {'input_ids': featured[0],
                  'attention_mask': featured[1],
                  'valid_ids': featured[2],
                  'active_mask': featured[3],
                  'valid_output': featured[4]
                  }
        # featured: 包含五项, input_ids, input_mask, valid_ids, active_mask, valid_output
        # kpe_model forward:  input_ids, attention_mask, valid_ids, active_mask, valid_output
        logits = self.kpe_model(**inputs)
        predicted_keyphrases = self.rank2phrase_chinese(tokenized[3], logits[0])
        return predicted_keyphrases

    @torch.no_grad()
    def batch_predict(self, docs: list):
        batch_list = []
        for doc in docs:
            doc_tokens = doc["text_tokens"]
            doc_id = doc["mepDocKey"]
            prediction = self.predict(doc_tokens)
            batch_list.append({"returnCode": 0, "mepDocKey": doc_id, "mepKeywords": prediction})

        return {"res": batch_list}
