import json
import os
import sys
import tqdm
import torch
import utils
import logging
import argparse
import traceback
from tqdm import tqdm

sys.path.append("..")

from bertkpe import tokenizer_class, Idx2Tag, Tag2Idx, Decode_Candidate_Number
from bertkpe import dataloader, generator, evaluator

torch.backends.cudnn.benchmark = True


logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Decoder Selector
# -------------------------------------------------------------------------------------------
def select_decoder(name):
    if name == 'bert2span':
        return bert2span_decoder
    elif name == 'bert2tag':
        return bert2tag_decoder
    elif name == 'bert2chunk':
        return bert2chunk_decoder
    elif name in ['bert2rank', 'bert2joint']:
        return bert2rank_decoder

    raise RuntimeError('Invalid retriever class: %s' % name)


# bert2span
def bert2span_decoder(args, data_loader, dataset, model, test_input_refactor,
                      pred_arranger, mode, stem_flag=False):
    logging.info('Start Generating Keyphrases for %s ... \n' % mode)
    test_time = utils.Timer()
    if args.dataset_class == "kp20k": stem_flag = True

    tot_examples = 0
    tot_predictions = []
    for step, batch in enumerate(tqdm(data_loader)):
        inputs, indices, lengths = test_input_refactor(batch, model.args.device)
        try:
            start_lists, end_lists = model.test_bert2span(inputs, lengths)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        # decode logits to phrase per batch
        params = {'examples': dataset.examples,
                  'start_lists': start_lists,
                  'end_lists': end_lists,
                  'indices': indices,
                  'max_phrase_words': args.max_phrase_words,
                  'return_num': Decode_Candidate_Number,
                  'stem_flag': stem_flag}

        batch_predictions = generator.span2phrase(**params)
        tot_predictions.extend(batch_predictions)

    candidate = pred_arranger(tot_predictions)
    return candidate


# bert2tag
def bert2tag_decoder(args, data_loader, dataset, model, test_input_refactor,
                     pred_arranger, mode, stem_flag=False):
    logging.info('Start Generating Keyphrases for %s ... \n' % mode)
    test_time = utils.Timer()
    if args.dataset_class == "kp20k":
        stem_flag = True

    tot_examples = 0
    tot_predictions = []
    for step, batch in enumerate(tqdm(data_loader)):
        inputs, indices, lengths = test_input_refactor(batch, model.args.device)
        try:
            logit_lists = model.test_bert2tag(inputs, lengths)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        # decode logits to phrase per batch
        params = {'examples': dataset.examples,
                  'logit_lists': logit_lists,
                  'indices': indices,
                  'max_phrase_words': args.max_phrase_words,
                  'pooling': args.tag_pooling,
                  'return_num': Decode_Candidate_Number,
                  'stem_flag': stem_flag}

        batch_predictions = generator.tag2phrase(**params)
        tot_predictions.extend(batch_predictions)

    candidate = pred_arranger(tot_predictions)
    return candidate


# Bert2Chunk
def bert2chunk_decoder(args, data_loader, dataset, model, test_input_refactor,
                       pred_arranger, mode, stem_flag=False):
    logging.info('Start Generating Keyphrases for %s ... \n' % mode)
    test_time = utils.Timer()
    if args.dataset_class == "kp20k":
        stem_flag = True

    tot_examples = 0
    tot_predictions = []
    for step, batch in enumerate(tqdm(data_loader)):
        inputs, indices, lengths = test_input_refactor(batch, model.args.device)
        try:
            logit_lists = model.test_bert2chunk(inputs, lengths, args.max_phrase_words)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        # decode logits to phrase per batch
        params = {'examples': dataset.examples,
                  'logit_lists': logit_lists,
                  'indices': indices,
                  'max_phrase_words': args.max_phrase_words,
                  'return_num': Decode_Candidate_Number,
                  'stem_flag': stem_flag}

        batch_predictions = generator.chunk2phrase(**params)
        tot_predictions.extend(batch_predictions)

    candidate = pred_arranger(tot_predictions)
    return candidate


# Bert2Rank & Bert2Joint
def bert2rank_decoder(args, data_loader, dataset, model, test_input_refactor,
                      pred_arranger, mode):
    logging.info('Start Generating Keyphrases for %s ... \n' % mode)
    tot_predictions = []

    # 根据数据集语种选择对应的处理方法
    predictor = generator.rank2phrase_chinese

    for step, batch in enumerate(tqdm(data_loader)):
        # utils.test_input_refactor
        # inputs, indices, lengths 分别表示 喂入模型的数据, 该语料在数据集中的序号, phrase_list 的 长度
        inputs, indices, lengths = test_input_refactor(batch, model.args.device)

        try:
            logit_lists = model.test_bert2rank(inputs, lengths)

        except:
            logging.error(str(traceback.format_exc()))
            continue

        # decode logits to phrase per batch
        params = {'examples': dataset.examples,
                  'logit_lists': logit_lists,
                  'indices': indices,
                  'return_num': -1}
        # 根据 logits 对 候选关键词进行排序 并格式化返回结果
        batch_predictions = predictor(**params)
        tot_predictions.extend(batch_predictions)

    with open(os.path.join(args.result_save_path, "result.txt"), "w", encoding="utf-8") as output_file:
        for item in tot_predictions:
            json_dict = {}
            json_dict["doc_id"] = item[0]
            # json_dict["predicted_keyphrases"] = item[1]
            json_dict["prediction"] = item[3]
            json_dict["doc_len"] = item[4]

            output_file.write("{}\n".format(json.dumps(json_dict, ensure_ascii=False)))

    # pred_arranger 在 utils 下定义
    candidate = pred_arranger(tot_predictions)

    return candidate
