import json
import os
import logging
from tqdm import tqdm
import torch.utils.data as Data
from .bert2span_dataloader import (bert2span_preprocessor, bert2span_converter)
from .bert2tag_dataloader import (bert2tag_preprocessor, bert2tag_converter)
from .bert2chunk_dataloader import (bert2chunk_preprocessor, bert2chunk_converter)

from .bert2rank_dataloader import (bert2rank_preprocessor, bert2rank_converter)
from .bert2joint_dataloader import (bert2joint_preprocessor, bert2joint_preprocessor_chinese, bert2joint_converter)


example_preprocessor = {'bert2span': bert2span_preprocessor,
                        'bert2tag': bert2tag_preprocessor,
                        'bert2chunk': bert2chunk_preprocessor,
                        'bert2rank': bert2rank_preprocessor,
                        'bert2joint': bert2joint_preprocessor}

feature_converter = {'bert2span': bert2span_converter,
                     'bert2tag': bert2tag_converter,
                     'bert2chunk': bert2chunk_converter,
                     'bert2rank': bert2rank_converter,
                     'bert2joint': bert2joint_converter}

logger = logging.getLogger()


class MyDataset(Data.Dataset):
    """
    读取, 或生成适用于训练/测试 Bert-KPE 模型的数据集文件
    目前暂时不支持 DataLoader 中 shuffle功能, 当 shuffle=True时, 实际输出的仍是顺序的.
    shuffle 功能需要读入全量数据集, 并非本类的应用场景
    """
    def __init__(self, args, tokenizer, mode):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.model_class = self.args.model_class
        self.max_phrase_words = self.args.max_phrase_words
        self.epochs = self.args.max_train_epochs

        self.pretrain_model = 'bert' if 'roberta' not in self.args.pretrain_model_type else 'roberta'
        # cached 文件, 将数据集经 BertTokenizer 预处理后得到的文件
        self.cached_file_path = os.path.join(self.args.cached_features_dir,
                                      "cached.%s.%s.%s.%s.json" %
                                      (self.args.model_class, self.pretrain_model, self.args.dataset_class, self.mode))
        # cached 文件的描述文件
        self.info_file_path = os.path.join(self.args.preprocess_folder, "%s.INFO.json" % (self.args.dataset_class))

        self.epoch = 0
        self.example_num = 0
        self.time = 0
        # 如果 cached 数据集不存在, 则会重新生成; 反之, 读取之
        if not os.path.exists(self.cached_file_path):
            self.build_cached_dataset()
        self.iterator = self.open_file()
        self.get_len()

    def open_file(self):
        return open(self.cached_file_path, "r", encoding="utf-8")

    def build_cached_dataset(self):
        logger.info("start loading source %s %s data ..." % (self.args.dataset_class, self.mode))
        # yuanwenjian
        file_path = os.path.join(self.args.preprocess_folder, "%s.%s.json" % (self.args.dataset_class, self.mode))

        if not os.path.exists(self.args.cached_features_dir):
            os.mkdir(self.args.cached_features_dir)

        # 选用 适配中文的预处理方法
        preprocessor = bert2joint_preprocessor_chinese

        logger.info('start preparing (%s) features for bert2joint (%s) ...' % (self.mode, self.pretrain_model))
        with open(file_path, "r", encoding="utf-8") as input_file, open(self.cached_file_path, "w", encoding="utf-8") as output_file:
            for line in tqdm(input_file):
                json_line = json.loads(line)
                cached_example = preprocessor(**{'example': json_line, 'tokenizer': self.tokenizer,
                                                 'max_token': self.args.max_token, 'mode': self.mode,
                                                 'max_phrase_words': self.args.max_phrase_words, 'stem_flag': False})
                if cached_example != {}:
                    self.example_num += 1
                    output_file.write("{}\n".format(json.dumps(cached_example, ensure_ascii=False)))

        # 数据集描述文件不存在
        if not os.path.exists(self.info_file_path):
            with open(self.info_file_path, "w", encoding="utf-8") as file:
                file.write(json.dumps({self.mode: self.example_num}, ensure_ascii=False))
        else:
            with open(self.info_file_path, "r", encoding="utf-8") as file:
                json_data = json.loads(file.readline())
                json_data[self.mode] = json_data.get(self.mode, self.example_num)
            with open(self.info_file_path, "w", encoding="utf-8") as file:
                file.write(json.dumps(json_data, ensure_ascii=False))

    def get_len(self):
        with open(self.info_file_path, "r", encoding="utf-8") as file:
            json_line = json.loads(file.readline())
            tmp = json_line.get(self.mode)

            if tmp is None:
                print("Found no Info for %s, rebuild it!" % self.cached_file_path)
                self.build_cached_dataset()
            self.example_num = tmp

    # 该方法为继承 Dataset 所必须重写的方法之一, 返回读取的数据集条数
    def __len__(self):
        return self.example_num

    # 该方法为继承 Dataset 所必须重写的方法
    # 在类中实现 __getitem__ 方法时, 可以对类的实例 以切片的形式 访问, 例如 instance[i]
    def __getitem__(self, num):
        line = self.iterator.__next__()
        result = json.loads(line)
        self.time += 1
        if self.time == self.example_num:
            self.iterator.close()
            self.epoch += 1
            if self.epoch < self.epochs:
                self.iterator = self.open_file()
            self.time = 0
        return feature_converter[self.model_class](num, result, self.tokenizer, self.mode, self.max_phrase_words)