# **Chinese Language Implementation of BERT for Keyphrase Extraction** (PyTorch)

This repository contains a Chinese language implementation for the code of the paper [**Capturing Global Informativeness in Open Domain Keyphrase Extraction**](https://arxiv.org/abs/2004.13639).  

This implementation requires tokenizations of Chinese language. The standard format of each sample is a dict object which contains 3 keys:  

1. text: string, a particular article, for example: "我爱你, 中国!"
2. text_tokens: list of string, tokenizations of article, for example: ["我", "爱", "你", ",", "中国", "!"]
3. keyphrases: a list of string, exists only in training dataset, labeled key phrases of a particular article, for example: ["我爱你", "中国"] 

IMPORTANT: the labelled keyphrase must be a exact token or a combination of consecutive tokens! in the tokenizations!  

Due to the very limited time, this repository is really at its infancy stage, and the readibility is really pool (I've try my best to write comments and arrange the code coherently). And I tested its correctness and usefulness in my self-made dataset, I will arrange this repository to make it coherent and readable if I'm available.

For your convenience, you can search all the functions ended with "chinese" to get all the modifications I made.
