# coding=utf-8
import json
import os
from bertkpe import constant
from tqdm import tqdm
from config import create_folder_if_absence


def show_progress(t: tqdm, statistic):
    t.set_postfix(statistic)


def write_data(file_operator, data, have_keyphrase):
    """
    依据数据集是否是训练集还是测试集, 执行相应的写数据规则
    :param file_operator:
    :param data:
    :param mode:
    :return:
    """

    if have_keyphrase:
        if data.get("keyphrases") and not data.get("absent_keyphrases") and not data.get("inside_token_keyphrases"):
            file_operator.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
    else:
        file_operator.write("{}\n".format(json.dumps(data, ensure_ascii=False)))


def read_tokens(raw_data: dict, text_token_field="text_tokens"):
    """
    读取分词信息
    :param raw_data:
    :return:
    """
    text_tokens = raw_data.get(text_token_field, None)

    return text_tokens


def get_keyphrases(raw_data: dict, field_name="keyphrases"):
    """
    读取标注的关键词
    """
    keyphrase_list = raw_data.get(field_name, None)

    return keyphrase_list


def trace_keywords(text_part: list, keyphrase: str):
    """
    追踪 标注关键词 在分词中的 相对位置
    """
    end = 0
    tmp = ""
    keyphrase_len = len(keyphrase)

    for token in text_part:
        end += 1
        tmp = "".join([tmp, token])

        if tmp == keyphrase:
            return end
        elif len(tmp) >= keyphrase_len:
            return 0

    return 0


def find_keyphrase_position(text_tokens: list, keyphrase_list: list):
    """
    定位 每一个 标记的关键词 在 分词结果 中的 起止位置, 格式为 [start_index, end_index+1].
    """
    keyphrase_loc = []
    new_keyphrase_list = []

    for keyphrase in keyphrase_list:
        for i, token in enumerate(text_tokens):
            if token == keyphrase:
                keyphrase_loc.append([[i, i+1]])
                new_keyphrase_list.append([keyphrase])
                break

            elif keyphrase.startswith(token):
                end = trace_keywords(text_tokens[i:], keyphrase)
                if end:
                    keyphrase_loc.append([[i, i + end]])
                    new_keyphrase_list.append(text_tokens[i: i + end])
                    break

    return keyphrase_loc, new_keyphrase_list


def check_keyphrase_status(text: str, labeled_keyphrases: list, found_keyphrases: list, num):
    """
    检查 标注的关键词 是否合法
    """
    check_keyphrases = []
    absent_keyphrases = 0           # 标注关键词是否是离线关键词, 即原文中未出现的
    inside_token_keyphrases = 0     # 标注的关键词是否是一个分词的一部分, 正常的关键词应该是一个完整的分词, 或是 连续的几个分词的组合
    for i in found_keyphrases:
        check_keyphrases.append("".join(i))

    new_labeled_keyphrases = []
    # 1) 检查是否存在离线的关键词
    for i in labeled_keyphrases:
        if i not in text:
            absent_keyphrases += 1
        else:
            new_labeled_keyphrases.append(i)

    # 检查真实关键词是否存在非完整分词的关键词
    if not set(check_keyphrases) == set(new_labeled_keyphrases):
        diff = set(labeled_keyphrases) - set(check_keyphrases)
        inside_token_keyphrases += len(diff)

    return absent_keyphrases, inside_token_keyphrases


def preprocess(input_path, output_path, badcase_path="", have_keyphrase=True):
    """
    对原始语料进行预处理, 包括 判断标记的关键词是否合法, 找出关键词 对应的分词, 以及位置信息
    原始语料必须包含以下三个字段:
        doc_id: str 或者 int型, 每篇文章的唯一标识符, 对整个模型无用,
        text_tokens: list of strings, 文章的分词结果
        keyphrases: list of string, 文章标记的关键词 [该字段对测试集为可选]

    input_path: 原始语料读取路径,
    output_path: 预处理语料保存路径
    badcase_path: 不合格样本保存路径
    have_keyphrase: 当前数据集是否提供真实的关键词
    """
    # 统计数据
    statistic = {
        "total": 0,
        "valid": 0,
        "error": 0,
        "empty_text": 0,
        "total_kp": 0,
        "empty_kp": 0,
        "absent_kp": 0,
        "inside_token_kp": 0
    }

    badcase_lists = []
    with open(input_path, "r", encoding="utf-8") as file, open(output_path, "w", encoding="utf-8") as output_file:
        t = tqdm(file)
        for line in t:
            statistic["total"] += 1
            try:
                json_load = json.loads(line)
            except json.decoder.JSONDecodeError:
                statistic["error"] += 1
                show_progress(t, statistic)
                continue

            # 读取文章原文及其分词结果
            text_tokens = read_tokens(json_load)

            if not text_tokens:
                statistic["empty_text"] += 1
                show_progress(t, statistic)
                continue
            else:
                record = {"doc_id": json_load["doc_id"],  # 每篇文章的唯一标识符, 可随意指定, 对模型无用
                          "text_tokens": text_tokens}

                # 读取标注的关键词
                labeled_keyphrases = get_keyphrases(json_load)
                if labeled_keyphrases:
                    # 检测 标注的关键词 是否在 分词 中 出现, 并确定其起止位置
                    found_keyphrase_loc, found_keyphrases = find_keyphrase_position(text_tokens, labeled_keyphrases)
                    if found_keyphrases:
                        # 检查标注关键词的合法性
                        text = "".join(text_tokens)
                        absent_keyphrases, inside_token_keyphrases = check_keyphrase_status(text, labeled_keyphrases,
                                                                                            found_keyphrases,
                                                                                            statistic["total"])
                        statistic["total_kp"] += len(labeled_keyphrases)
                        statistic["absent_kp"] += absent_keyphrases
                        statistic["inside_token_kp"] += inside_token_keyphrases

                        if absent_keyphrases or inside_token_keyphrases:
                            record["absent_keyphrases"] = absent_keyphrases
                            record["inside_token_keyphrases"] = inside_token_keyphrases
                            record["keyphrases"] = labeled_keyphrases
                            record["found_kp"] = found_keyphrases
                            badcase_lists.append(record)
                        else:
                            statistic["valid"] += 1
                            # 关键词位置信息
                            record["keyphrase_loc"] = found_keyphrase_loc
                            record["keyphrases"] = found_keyphrases
                            write_data(output_file, record, have_keyphrase)
                else:
                    # 针对不含关键词的测试集
                    if not have_keyphrase:
                        write_data(output_file, record, have_keyphrase)
                        statistic["valid"] += 1
                    else:
                        statistic["empty_kp"] += 1

                show_progress(t, statistic)

        if not badcase_lists and badcase_path:
            with open(badcase_path, "w", encoding="utf-8") as badcase_file:
                for badcase in badcase_lists:
                    badcase_file.write("{}\n".format(json.dumps(badcase, ensure_ascii=False)))

    print("preprocess data finished!")


raw_train_data_path = constant.raw_train_data_path
raw_test_data_path = constant.raw_test_data_path
train_data_have_keyphrase = True
test_data_have_keyphrase = True


preprocess_folder = os.path.join(constant.general_preprocess_folder, constant.dataset_class)
create_folder_if_absence(preprocess_folder)

preprocessed_train_data_path = os.path.join(preprocess_folder,
                                            "{}.{}.json".format(constant.dataset_class, "train"))
badcase_train_data_save_path = os.path.join(preprocess_folder,
                                            "{}.{}.{}.json".format(constant.dataset_class, "train", "badcase"))
preprocessed_test_data_path = os.path.join(preprocess_folder,
                                            "{}.{}.json".format(constant.dataset_class, "test"))
badcase_test_data_save_path = os.path.join(preprocess_folder,
                                            "{}.{}.{}.json".format(constant.dataset_class, "test", "badcase"))

# generate preprocessed train data
preprocess(raw_train_data_path,
           preprocessed_train_data_path,
           badcase_train_data_save_path,
           train_data_have_keyphrase
           )
# generate preprocessed test data
preprocess(raw_test_data_path,
           preprocessed_test_data_path,
           badcase_test_data_save_path,
           test_data_have_keyphrase
           )