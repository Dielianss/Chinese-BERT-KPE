import os
import sys
import logging

import bertkpe.constant as constant
logger = logging.getLogger()


def create_folder_if_absence(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_lastest_checkpoint_file(checkpoint_path: str, mode):
    """
    检查在 当前配置的 checkpoint_file保存路径 下的 最新 checkpoint_file.
    :param args:
    :return:
    """
    tmp_epoch = -1
    checkponit_file = ""
    for name in os.listdir(checkpoint_path):
        if "epoch_" in name:
            tmp = name.split(".")[3].split("_")[1]
            if tmp.isdigit():
                tmp = int(tmp)
                if tmp > tmp_epoch:
                    tmp_epoch = tmp
                    checkponit_file = name

    if tmp_epoch == -1 and mode == "eval":
        print("Found no checkpoint file in the directory: ", checkpoint_path)
        exit()

    return os.path.join(checkpoint_path, checkponit_file)


def add_default_args(parser):
    # ---------------------------------------------------------------------------------------------
    # mode select
    modes = parser.add_argument_group('Modes')
    modes.add_argument('--run_mode', type=str, choices=['train', 'eval'], default=constant.run_mode,
                       help='Select running mode. ')
    modes.add_argument('--dataset_class', type=str,
                       choices=['openkp', 'kp20k'],
                       default=constant.dataset_class,
                       help='Select datasets.')

    modes.add_argument('--model_class', type=str, 
                       choices=['bert2span', 'bert2tag', 'bert2chunk', 'bert2rank', 'bert2joint'],
                       default=constant.model_class,
                       help='Select different model types.')
    modes.add_argument("--pretrained_model_type", type=str,
                       choices=['bert-base-cased', 'spanbert-base-cased', 'roberta-base', 'chinese_L-12_H-768_A-12'],
                       default=constant.pretrained_model_type,
                       help="Select pretrain model type.")

    # ---------------------------------------------------------------------------------------------
    # Filesystem
    files = parser.add_argument_group('Files')
    files.add_argument('--base_folder', type=str, default=constant.base_folder,
                       help='Directory of general working folder.')
    files.add_argument('--general_preprocess_folder', type=str, default=constant.general_preprocess_folder,
                       help='Directory of preprocess data.')
    files.add_argument("--general_pretrained_model_path", type=str, default=constant.general_pretrained_model_path,
                       help="Path to pre-trained BERT model.")
    files.add_argument("--general_cached_features_folder", type=str, default=constant.general_cached_features_folder,
                       help="Filepath used to reload preprocessed data features.")

    # ---------------------------------------------------------------------------------------------
    # Runtime environment
    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--no_cuda', action='store_true', default=constant.no_cuda,
                         help='Train Model on GPUs (False)')
    runtime.add_argument("--local_rank", type=int, default=constant.local_rank,
                         help="Set local_rank=0 for distributed training on multiple gpus")
    # DataLoader中读取数据集的线程数! 0: 只使用一个进程
    runtime.add_argument('--data_workers', type=int, default=constant.data_workers,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--seed', type=int, default=constant.seed,
                         help="random seed for initialization")

    # ---------------------------------------------------------------------------------------------
    # Train parameters
    train = parser.add_argument_group('Training')
    train.add_argument("--max_token", type=int, default=constant.max_token,
                       help="Max length of document WordPiece tokens + '[CLS]'+'[SEP]' <= 512.")
    train.add_argument("--max_train_epochs", type=int, default=constant.max_train_epochs,
                       help="Total number of training epochs to perform. ")
    train.add_argument("--max_train_steps", type=int, default=constant.max_train_steps,
                       help="Total number of training steps. ")
    train.add_argument("--per_gpu_train_batch_size", type=int, default=constant.per_gpu_train_batch_size,
                       help="Batch size per GPU/CPU for training.")
    train.add_argument("--per_gpu_test_batch_size", type=int, default=constant.per_gpu_test_batch_size,
                       help="Batch size per GPU/CPU for test, orignal = 128")
    train.add_argument("--gradient_accumulation_steps", type=int, default=constant.gradient_accumulation_steps,
                       help="Number of updates steps to accumulate before performing a backward/update pass.")

    # ---------------------------------------------------------------------------------------------
    # Optimizer
    optim = parser.add_argument_group('Optimizer')
    optim.add_argument("--learning_rate", default=constant.learning_rate, type=float,
                       help="The initial learning rate for Adam.")
    optim.add_argument("--weight_decay", default=constant.weight_decay, type=float,
                       help="Weight deay if we apply some.")
    optim.add_argument("--warmup_proportion", default=constant.warmup_proportion, type=float,
                       help="Linear warmup over warmup_ratio warm_step / t_total.")
    optim.add_argument("--adam_epsilon", default=constant.adam_epsilon, type=float,
                       help="Epsilon for Adam optimizer.")
    optim.add_argument("--max_grad_norm", default=constant.max_grad_norm, type=float,
                       help="Max gradient norm.")

    # ---------------------------------------------------------------------------------------------
    # Evaluation
    evaluate = parser.add_argument_group('Evaluation')
    evaluate.add_argument("--tag_pooling", default=constant.tag_pooling, type=str,
                          help="Pooling methods for Bert2Tag.")
    evaluate.add_argument("--eval_checkpoint", default=constant.eval_checkpoint, type=str,
                          help="Tha checkpoint file to be evaluated. ")
    evaluate.add_argument("--max_phrase_words", default=constant.max_phrase_words, type=int,
                          help="The max length of phrases. ")
    
    # ---------------------------------------------------------------------------------------------
    # General
    general = parser.add_argument_group('General')
    # action="store_true" 意为如果在命令行中显示调用对应的参数, 例如 "--use_viso", 但不指定值, 则值记为True, 若未调用对应参数
    # 则值为默认值
    general.add_argument('--use_viso', action='store_true', default=constant.use_viso,
                         help='Whether use tensorboadX to log loss.')
    general.add_argument('--display_iter', type=int, default=constant.display_iter,
                         help='Log state after every <display_iter> batches.')
    general.add_argument('--load_checkpoint', action='store_true', default=constant.load_checkpoint,
                         help='Path to a checkpoint for generation .')
    general.add_argument('--save_checkpoint', action='store_true', default=constant.save_checkpoint,
                         help='If true, Save model + optimizer state after each epoch.')
    general.add_argument('--server_ip', type=str, default=constant.server_ip,
                         help="For distant debugging.")
    general.add_argument('--server_port', type=str, default=constant.server_port,
                         help="For distant debugging.")
    general.add_argument('--fp16', action='store_true', default=constant.fp16,
                         help="Whether to use 16-bit float precision instead of 32-bit")
    general.add_argument('--fp16_opt_level', type=str, default=constant.fp16_opt_level,
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")


# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def init_args_config(args):
    # bert pretrained model path
    args.pretrained_model_dir = os.path.join(args.general_pretrained_model_path, args.pretrained_model_type)

    # create general preprocessed data folder and cached features folder if necessary
    create_folder_if_absence(args.general_preprocess_folder)
    create_folder_if_absence(args.general_cached_features_folder)

    # specified dataset saved folder
    args.preprocess_folder = os.path.join(args.base_folder, constant.prepro_dataset_folder, args.dataset_class)
    create_folder_if_absence(args.preprocess_folder)
    args.cached_features_folder = os.path.join(args.base_folder, constant.cached_dataset_folder, args.dataset_class)
    create_folder_if_absence(args.cached_features_folder)

    # whether test data provide keyphrases, if yes, will evaluate the performance of the model on test_data
    args.keyphrase_in_test_data = constant.keyphrase_in_test_data

    # general prediction save folder [including training&testing results]
    args.general_save_path = os.path.join(args.base_folder, constant.result_dataset_folder)
    create_folder_if_absence(args.general_save_path)

    # specified model&data_class&mode result save folder
    predicted_result_name = "%s_%s_%s_%s" % (args.run_mode, args.model_class, args.dataset_class,
                                             args.pretrained_model_type.split('-')[0])
    args.result_save_path = os.path.join(args.general_save_path, predicted_result_name)
    create_folder_if_absence(args.result_save_path)

    # checkpoint files saved folder
    args.checkpoint_folder = os.path.join(args.general_save_path,
                                          "%s_%s_%s_%s" % ("train", args.model_class, args.dataset_class,
                                                           args.pretrained_model_type.split('-')[0]),
                                          "checkpoints")
    create_folder_if_absence(args.checkpoint_folder)

    # get the latest checkpoint file
    args.checkpoint_file = check_lastest_checkpoint_file(args.checkpoint_folder, args.run_mode)

    # viso folder
    if args.use_viso:
        args.viso_folder = os.path.join(args.result_save_path, 'viso')
        create_folder_if_absence(args.viso_folder)
    
    # logging file
    args.log_file = os.path.join(args.result_save_path, 'logging.txt')
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 
    logger.addHandler(console) 
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    logger.info("preprocess_folder = {}".format(args.preprocess_folder))
    logger.info("pretrained Model Type = {}".format(args.pretrained_model_type))