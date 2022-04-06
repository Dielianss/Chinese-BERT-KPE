import os
import sys
import tqdm
import torch
import logging
import argparse
import traceback
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import utils
import config
from test import bert2rank_decoder
from model import KeyphraseSpanExtraction
from utils import pred_arranger_chinese, chinese_evaluate_script, train_input_refactor_bert2joint, test_input_refactor
from bertkpe import dataloader
# Select dataloader
from bertkpe.dataloader.bert2joint_dataloader import batchify_bert2joint_features_for_train, batchify_bert2joint_features_for_test
torch.backends.cudnn.benchmark = True
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------------------------
def train(args, data_loader, model, train_input_refactor, stats, writer):
    logger.info("start training %s on %s (%d epoch) || local_rank = %d..." %
                (args.model_class, args.dataset_class, stats['epoch'], args.local_rank))

    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    epoch_loss = 0
    epoch_step = 0

    epoch_iterator = tqdm(data_loader, desc="Train_Iteration", disable=args.local_rank not in [-1, 0])

    # data_loader 中定义了 __getitem__ 方法, 在迭代取数据时还会对其进一步做处理,生成 batch, 再喂入模型
    for step, batch in enumerate(epoch_iterator):
        # inputs: 将预处理完的数据整理为存入 dict
        inputs, indices = train_input_refactor(batch, model.args.device)
        try:
            loss = model.update(step, inputs)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        train_loss.update(loss)
        epoch_loss += loss
        epoch_step += 1

        if args.local_rank in [-1, 0] and step % args.display_iter == 0:
            if args.use_viso:
                writer.add_scalar('train/loss', train_loss.avg, model.updates)
                writer.add_scalar('train/lr', model.scheduler.get_lr()[0], model.updates)

            # logging.info('Local Rank = %d | train: Epoch = %d | iter = %d/%d | ' %
            #             (args.local_rank, stats['epoch'], step, len(train_data_loader)) +
            #             'loss = %.4f | lr = %f | %d updates | elapsed time = %.2f (s) \n' %
            #             (train_loss.avg, model.scheduler.get_lr()[0], model.updates, stats['timer'].time()))
            train_loss.reset()

    logging.info('Local Rank = %d | Epoch Mean Loss = %.8f ( Epoch = %d ) | Time for epoch = %.2f (s) \n' %
                 (args.local_rank, (epoch_loss / epoch_step), stats['epoch'], epoch_time.time()))


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # setting args
    parser = argparse.ArgumentParser('BertKPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_default_args(parser)
    args = parser.parse_args()
    config.init_args_config(args)

    output_filename = os.path.join(args.result_save_path, 'result.txt')

    # -------------------------------------------------------------------------------------------
    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # local_rank: 0 开启分布式训练
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda", 0)
        # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    logger.info("process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    print()
    # -------------------------------------------------------------------------------------------
    utils.set_seed(args)

    # 预处理语料选取器, 从 预处理语料 中 选取适当的字段 用于模型的 训练/推断
    train_input_refactor, test_input_refactor = train_input_refactor_bert2joint, test_input_refactor
    # Bert 分词器
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)
    # 模型推断主函数
    candidate_decoder = bert2rank_decoder
    # 从 推断结果中 提取 必要的字段 用于计算指标
    dataset_pred_arranger = pred_arranger_chinese
    # 计算脚本选择
    evaluate_script, main_metric_name = chinese_evaluate_script, "max_f1_score5"

    # -------------------------------------------------------------------------------------------
    if args.run_mode == "eval":
        model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(args.checkpoint_file, args)
        # set model device
        model.set_device()
        # -------------------------------------------------------------------------------------------
        # build dev dataloader
        dev_dataset = dataloader.build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'dev'})
        args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_data_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_bert2joint_features_for_test,
            pin_memory=args.cuda,
        )
        logger.info("Successfully load cached test features!")
        print()

        # decode candidate phrases
        dev_candidate = candidate_decoder(args, dev_data_loader, dev_dataset, model, test_input_refactor,
                                          dataset_pred_arranger, 'dev')
        stats = {'timer': utils.Timer(), 'epoch': 0, main_metric_name: 0}

        # 当测试集提供关键词时, 计算指标
        if args.keyphrase_in_test_data:
            stats = evaluate_script(args, dev_candidate, stats, mode='dev', metric_name=main_metric_name)

    else:
        # 在 训练 模式下, 使用 自己实现的 MyDataset 加载训练集, MyDataset 不会一次性将全部数据都读入内存, 对大文件更友好
        train_dataset = dataloader.MyDataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'train'})
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = torch.utils.data.sampler.RandomSampler(
            train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            # sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_bert2joint_features_for_train,
            pin_memory=args.cuda,
        )
        logger.info("Successfully load cached train features!")
        print()
        # -------------------------------------------------------------------------------------------
        dev_dataset = dataloader.build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'dev'})
        args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_data_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_bert2joint_features_for_test,
            pin_memory=args.cuda,
        )
        logger.info("Successfully load cached test features!")
        print()

        # -------------------------------------------------------------------------------------------
        # Set training total steps
        t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.max_train_epochs

        # -------------------------------------------------------------------------------------------
        # Preprare Model & Optimizer
        # -------------------------------------------------------------------------------------------
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        logger.info(" ************************** Initilize Model & Optimizer ************************** ")
        if args.load_checkpoint and os.path.isfile(args.checkpoint_file):
            model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(args.checkpoint_file, args)
            logger.info('Training model from epoch {}...'.format(checkpoint_epoch))
            start_epoch = checkpoint_epoch + 1
            train_dataset.epoch = checkpoint_epoch

        else:
            logger.info('Training model from scratch...')
            model = KeyphraseSpanExtraction(args)
            start_epoch = 1

        # initial optimizer
        model.init_optimizer(num_total_steps=t_total)
        # set model device
        model.set_device()

        if args.n_gpu > 1:
            model.parallelize()

        if args.local_rank != -1:
            model.distribute()

        if args.local_rank in [-1, 0] and args.use_viso:
            tb_writer = SummaryWriter(args.viso_folder)
        else:
            tb_writer = None

        # -------------------------------------------------------------------------------------------
        # start training
        # -------------------------------------------------------------------------------------------
        model.zero_grad()
        stats = {'timer': utils.Timer(), 'epoch': 0, main_metric_name: 0}

        for epoch in range(start_epoch, (args.max_train_epochs + 1)):
            stats['epoch'] = epoch
            # train
            train(args, train_data_loader, model, train_input_refactor, stats, tb_writer)

            # previous metric score
            prev_metric_score = stats[main_metric_name]

            # decode candidate phrases
            dev_candidate = candidate_decoder(args, dev_data_loader, dev_dataset, model, test_input_refactor,
                                              dataset_pred_arranger, 'dev')
            stats = evaluate_script(args, dev_candidate, stats, mode='dev', metric_name=main_metric_name)

            # new metric score
            new_metric_score = stats[main_metric_name]

            # save checkpoint : when new metric score > previous metric score
            if args.save_checkpoint and (new_metric_score > prev_metric_score) and (
                    args.local_rank == -1 or torch.distributed.get_rank() == 0):
                checkpoint_name = '{}.{}.{}.epoch_{}.checkpoint'.format(args.model_class, args.dataset_class,
                                                                        args.pretrained_model_type.split('-')[0], epoch)

                model.save_checkpoint(os.path.join(args.checkpoint_folder, checkpoint_name), stats['epoch'])