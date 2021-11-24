# !/usr/bin/env python
# coding: utf-8

import fasttext
import pandas as pd
import json


def check_E2E(input_file_path: str, threshold=0.3, use_source_name=True):
    number = 0
    unknown_real = 0
    unknown_predict_real = 0
    unknown_predict = 0
    right1 = 0
    right2 = 0
    right3 = 0
    right4 = 0

    print("当前判为 Unknown 的阈值为 {}".format(threshold))
    print("当前 use_sourcename设置为 {}".format(use_sourcename))

    def cal_metrics(true_label, predicted_label):
        nonlocal unknown_real, unknown_predict_real, unknown_predict, right1, right2, right3, right4
        layer3_label_list = []
        layer2_label_list = []
        layer1_label_list = []

        if ";" in true_label:
            layer3_label_list = true_label.split(";")
            for label in layer3_label_list:
                layer2_label_list.append(label[:4])
                layer1_label_list.append(label[:2])

        else:
            layer3_label_list.append(true_label)
            layer2_label_list.append(true_label[:4])
            layer1_label_list.append(true_label[:2])

        if "0000000000" in true_label:
            unknown_real += 1

        if predicted_label == "0000000000" and "0000000000" in true_label:
            unknown_predict_real += 1

        if predicted_label == "0000000000":
            unknown_predict += 1

        else:
            if predicted_label[:2] in layer1_label_list:
                right1 += 1

            if predicted_label[:4] in layer2_label_list:
                right2 += 1

            if predicted_label in true_label:
                right3 += 1

            if predicted_label not in true_label:
                if (predicted_label[:4] + "000000") in true_label:
                    right4 += 1

    def metric_output():
        print("预测总数: {}".format(number))
        print("一级分类正确率: 正确个数: {}, 正确率: {}%".format(right1, right1 / number * 100))
        print("二级分类正确率: 正确个数: {}, 正确率: {}%".format(right2, right2 / number * 100))
        print("三级分类正确率: 正确个数: {}, 正确率: {}%".format(right3, right3 / number * 100))
        print("未到叶子结点数目：{} , 占比：{}%".format(right4, right4 / number * 100))
        print("真实标签的unknown数目：{} , 占比：{}%".format(unknown_real, unknown_real / number * 100))
        print("预测标签的unknown数目：{} , 占比：{}%".format(unknown_predict, unknown_predict / number * 100))
        print("预测标签和真实标签都为unknown数目：{} , 占比：{}%".format(unknown_predict_real,
                                                        unknown_predict_real / number * 100))

    model = Model(maps_path, model20_path, model21_path, model30_path, model31_path, use_source_name)
    with open("/data/yangj/standard_corpus/mep_5w/check_mep_2w.json", 'w', encoding="utf-8") as output_file:
        with open(input_file_path, 'r', encoding="utf-8") as file, open(
                "/data/yangj/standard_corpus/mep_5w/mep_source_2w.json") as file2:
            for line, line2 in tqdm(zip(file, file2)):
                json_data = json.loads(line)
                record = json.loads(line2)["mep"]["data"]["docs"][0]["cleanText"]
                write = {}
                mep_post = json_data["mep_post_content"]["data"]["docs"][0]
                mep_terms = json.loads(mep_post["mep_terms"])
                mep_terms_noNature = json.loads(mep_post["mep_terms_noNature"])
                source_name = mep_post['16003']
                raw_labels = json_data['label']

                title_terms, body_terms, body_stopwords_rm = model.data_process(mep_terms_noNature, mep_terms)

                predicted_label = model.transform(title_terms, body_terms, body_stopwords_rm, source_name, threshold)[0]
                write["cleanText"] = record
                write["true_label"] = raw_labels
                write["predicted_label"] = predicted_label
                # write["check_true_label"] = json.loads(line2)['labels']

                json.dump(write, output_file, ensure_ascii=False)
                output_file.write("\n")
                if predicted_label != "":
                    number += 1
                    cal_metrics(raw_labels, predicted_label)

    metric_output()


if __name__ == "__main__":
    print("\n开始推断...\n")
    model_path = "/data/yangj/standard_corpus/basic_char_model.bin"
    test_data = "/data/yangj/standard_corpus/0_preprocessed_dataset.json"
    fasttext_model = fasttext.load_model("/data/yangj/standard_corpus/basic_char_model.bin")
    policy_output = "/data/yangj/standard_corpus/policy_output.json"
    entertainment_output = "/data/yangj/standard_corpus/entertainment_output.json"

    policy_number = 0
    entertainment_number = 0
    with open(test_data, "r", encoding="utf-8") as file, open(policy_output, "w", encoding="utf-8") as policy_file, \
            open(policy_output, "w", encoding="utf-8") as entertainment_file:
        for line in file:
            json_data = json.loads(line)
            text = json_data["text"]
            predicted_label, predicted_score = fasttext_model.predict(text, 1)

            if predicted_label[9:] == ["1600000000"] and policy_number < 500000:
                policy_file.write(line)
                policy_number += 1
            elif predicted_label[9:] == ["1900000000"] and entertainment_number < 500000:
                entertainment_file.write(line)
                entertainment_number += 1

        print("Predicting done!")
        print("Policy Number: ", policy_number)
        print("Entertainment Number: ", entertainment_number)




