import logging
import torch

from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from utils import override_args
from bertkpe import Idx2Tag, networks, generator
from bertkpe.networks.Bert2Joint import BertForChunkTFRanking


logger = logging.getLogger()


class KeyphraseSpanExtraction(object):
    """
    Idx2Tag: 判断一个词是否属于关键词组: ['O', 'B', 'I', 'E', 'U']
    O : Non Keyphrase ; B : Begin word of the keyprase ; I : Middle word of the keyphrase ; E : End word of keyprhase ; U : Uni-word keyphrase
    'O': 非关键词; 'B': 关键词组首词; 'I': 关键词组中间词; 'E': 关键词组尾词; 'U':  单一词的关键词组
    """

    def __init__(self, args, state_dict=None):
        self.args = args
        self.updates = 0

        # 定义模型, 若模型为 Bert2Joint, 则其网络定义在 network/Bert2Joint.BertForChunkTFRanking中
        network = BertForChunkTFRanking
        # 定义 词语的关键词属性, 有 5 种 和 2种 之分, 设为 5 种时, 有以下类型: 非关键词, 关键词首个分词, 中间分词, 尾分词, 单分词关键词
        args.num_labels = 2 if args.model_class != 'bert2tag' else len(Idx2Tag)

        logger.info('Config num_labels = %d' % args.num_labels)
        model_config = BertConfig.from_pretrained(args.pretrained_model_dir, num_labels=args.num_labels)

        self.network = network.from_pretrained(args.pretrained_model_dir, config=model_config)
        self.return_num = 5
        # load checkpoint
        if state_dict is not None:
            self.network.load_state_dict(state_dict)
            logger.info('loaded checkpoint state_dict')

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def init_optimizer(self, num_total_steps):
        num_warmup_steps = int(self.args.warmup_proportion * num_total_steps)
        logger.info('warmup steps : %d' % num_warmup_steps)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']

        param_optimizer = list(self.network.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_total_steps)

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # train
    def update(self, step, inputs):
        # Train mode
        self.network.train()

        # run !
        loss = self.network(**inputs)

        if self.args.n_gpu > 1:
            # mean() to average on multi-gpu parallel (not distributed) training
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)

        # if self.args.fp16:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
        # else:
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
            self.updates += 1
        return loss.item()

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # test

    # bert2rank & bert2joint
    def test_bert2rank(self, inputs, numbers):
        """
        获取每一样本的候选关键词(phrase_list) 对应的预测概率(logits)
        :param inputs:
        :param numbers: 候选关键词长度, 即 phrase_list 的长度
        :return:
        """
        self.network.eval()
        with torch.no_grad():
            logits = self.network(**inputs)  # shape = (batch_size, max_diff_gram_num)
        # assert (logits.shape[0] == len(numbers)) and (logits.shape[1] == max(numbers))
        logits = logits.data.cpu().tolist()

        logit_lists = []
        for batch_id, num in enumerate(numbers):
            logit_lists.append(logits[batch_id][:num])
        return logit_lists

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def save_checkpoint(self, filename, epoch):
        network = self.network.module if hasattr(self.network, 'module') else self.network
        params = {
            'args': self.args,
            'epoch': epoch,
            'state_dict': network.state_dict(),
        }
        try:
            torch.save(params, filename)
            logger.info('success save epoch_%d checkpoints !' % epoch)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load_checkpoint(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(filename, map_location=lambda storage, loc: storage)

        args = saved_params['args']
        epoch = saved_params['epoch']
        state_dict = saved_params['state_dict']

        if new_args:
            args = override_args(args, new_args)

        model = KeyphraseSpanExtraction(args, state_dict)
        logger.info('success loaded epoch_%d checkpoints ! From : %s' % (epoch, filename))
        return model, epoch

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def zero_grad(self):
        self.optimizer.zero_grad()
        # self.network.zero_grad()

    def set_device(self):
        self.network.to(self.args.device)

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)

    def distribute(self):
        self.distributed = True
        self.network = torch.nn.parallel.DistributedDataParallel(self.network,
                                                                 device_ids=[self.args.local_rank],
                                                                 output_device=self.args.local_rank,
                                                                 find_unused_parameters=True)


