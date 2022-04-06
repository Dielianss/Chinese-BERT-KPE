# **Chinese Language Implementation of BERT for Keyphrase Extraction** (PyTorch)

This repository contains a Chinese language implementation for the code of the paper [**Capturing Global Informativeness in Open Domain Keyphrase Extraction**](https://arxiv.org/abs/2004.13639).  

Thanks the authors for their generous help!❤

Please note that in this implementation, only model BertKPE-Joint supports Chinese Language! (All the modifications are for this model). And the rest models keep unchanged.

This implementation works fine on a linux server with following specifications:

1.  OS: Ubuntu 16.04.7
2.  GPU: Nvidia Tesla V100(32GB) 
3.  Cuda: 10.0
4.  Python: 3.7.4
5.  Pytorch: 1.2.0
6.  Pretrained Bert Model for Chinese: chinese_L-12_H-768_A-12(Pytorch version)

## Dataset Requirements

Requirements regarding datasets of this implementation are quite simple. 

For training data. you just need to provide the tokenizations of articles and corresponding keyphrases, both are in Python list. plus. you should also assign each article with an id (in str or int), although it does not get involved in model training&evaluating.

For testing data, the tokenizations of articles are required, while corresponding keyphrases are optional. If keyphrases are unavailable, this implementation will end with outputting predictions over testing data, and if available, this  implementation will also calculate the performance metrices. BTW, ids for articles are also required.

Below shows a example for training data:

`{`

`"doc_id": 1`,

`"text_tokens": ["凯普生物", "：", "公司", "可", "为", "各级", "医疗机构", "提供", "PCR", "实验室", "整体解决方案"]`,

`"keyphrases": ["凯普生物", "PCR实验室整体解决方案"]`

`}`

**IMPORTANT**

1.  Each provided keyphrase must be a exact token or a combination of consecutive ones in tokenizations!  Otherwise, the article will be dropped.
2.  Raw dataset are recommended in .json format.

## How to run?

1.  Download a pretrained Bert model for Chinese (**Pytorch version**). I used "chinese_L-12_H-768_A-12".


1.  Configure necessary parameters in bertkpe/constant/Constant.py. Below ones are currently unspecified and must be done before running:

    1.  dataset_class: the name of dataset
    2.  raw_training_data_path: the path to your raw training data
    3.  raw_testing_data_path: the path to your raw testing data
    4.  base_folder: the path to the base working folder of this implementation, where all the results will be saved.
    5.  general_pretrained_model_folder: the path where pretrained bert model saved
    6.  pretrained_model_type: the name of your provided pretrained bert model, for example: "chinese_L-12_H-768_A-12"

    **Attention**

    1.  choose a correct `run_mode`, "train" or "eval"
    2.  if your test data provide keyphrases, please set `keyphrase_in_test_data=True`
    3.  model_class must be `bert2joint` when datasets are in Chinese. And this model only support Chinese temporarily.

2.  Run 0_preprocess.py to preprocess the raw files

3.  Run 1_main.py to train/evaluate the model.

This implementation is still very primitive, and I've tried my best to make it coherent, and write comments as much as possible. For you convenience, you can quickly get all my modifications by searching methods which name ending with "chinese". I'll update this implementation when I'm available. Thank you!