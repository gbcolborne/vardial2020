""" Train or evaluate classifier. """

import sys, os, argparse, glob, pickle, random, logging, math
from io import open
from datetime import datetime
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, BertConfig
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from CharTokenizer import CharTokenizer
from BertDataset import BertDatasetForClassification, BertDatasetForMLM, BertDatasetForTesting
from Pooler import Pooler
sys.path.append("..")
from comp_utils import ALL_LANGS


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def count_params(model):
    count = 0
    for p in model.parameters():
         count += torch.prod(torch.tensor(p.size())).item()
    return count


def check_for_unk_train_data(train_paths):
    for path in train_paths:
        if os.path.split(path)[-1] == "unk.train":
            return path
    return None


def train(model, pooler, optimizer, scheduler, train_dataset, args, unk_dataset=None):
    """ Train model. 

    Args:
    - model: BertModelForMaskedLM
    - pooler: Pooler
    - optimizer
    - scheduler
    - train_dataset: BertDatasetForClassification
    - args
    - unk_dataset: optional) BertDatasetForMLM for unlabeled data

    """
    # Write config and tokenizer in output directory
    path_config = os.path.join(args.dir_output, "config.json")
    path_tokenizer = os.path.join(args.dir_output, "tokenizer.pkl")
    with open(path_tokenizer, "wb") as f:
        pickle.dump(tokenizer, f)
    

def main():
    parser = argparse.ArgumentParser()
    # Model and data are required
    parser.add_argument("--dir_pretrained_model",
                        type=str,
                        required=True,
                        help="Dir containing pre-trained model (checkpoint), which may have been fine-tuned already.")
    parser.add_argument("--dir_data",
                        type=str,
                        required=True,
                        help=("Dir containing data (files <lang>.train for training, <valid|test>.tsv for testing) "
                              "Training data files only contains sentences. "
                              "Validation data files must be in 2-column TSV format, with sentence and label. "
                              "Test data may contain one or 2 columns."))
    # Required for training
    parser.add_argument("--dir_output",
                        type=str,
                        help="Directory in which model will be written (required if --do_train)")
    
    # Execution modes
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Run training")
    parser.add_argument("--eval_during_training",
                        action="store_true",
                        help="Run evaluation on dev set during training")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Evaluate model on dev set")
    parser.add_argument("--do_pred",
                        action="store_true",
                        help="Run prediction on test set")

    # Hyperparameters
    parser.add_argument("--avgpool",
                        action="store_true",
                        help=("Use average pooling of all last hidden states, rather than just the last hidden state of CLS, to do classification. "
                              "Note that in either case, the pooled vector passes through a square linear layer and a tanh before the classification layer."))
    parser.add_argument("--sampling_distro",
                        choices=["uniform", "relfreq", "dampfreq"],
                        default="relfreq",
                        help="Distribution used for sampling training data within each group (relevant, confound, and irrelevant)")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seq_len",
                        default=128,
                        type=int,
                        help="Length of input sequences. Shorter seqs are padded, longer ones are trucated")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--equal_betas",
                        action='store_true',
                        help="Use beta1=beta2=0.9 for AdamW optimizer.")
    parser.add_argument("--correct_bias",
                        action='store_true',
                        help="Correct bias in AdamW optimizer (correct_bias=False is meant to reproduce BERT behaviour exactly.")
    parser.add_argument("--max_train_steps",
                        default=1000000,
                        type=int,
                        help="Maximum number of training steps to perform. Note: # optimization steps = # train steps / # accumulation steps.")
    parser.add_argument("--num_warmup_steps",
                        default=10000,
                        type=int,
                        help="Number of optimization steps (i.e. training steps / accumulation steps) to perform linear learning rate warmup for. ")
    parser.add_argument('--grad_accum_steps',
                        type=int,
                        default=1,
                        help="Number of training steps (i.e. batches) to accumualte before performing a backward/update pass.")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Num GPUs to use for training (0 for none, -1 for all available)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    
    # Check args
    assert args.do_train or args.do_eval or args.do_pred
    if args.eval_during_training:
        assert args.do_train
    if args.do_train:
        train_paths = glob.glob(os.path.join(args.dir_data, "*.train"))        
        assert len(train_paths) > 0
        assert args.dir_output is not None
        if os.path.exists(args.dir_output) and os.path.isdir(args.dir_output) and len(os.listdir(args.dir_output)) > 1:
            msg = "%s already exists and is not empty" % args.dir_output
            raise ValueError(msg)
        if not os.path.exists(args.dir_output):
            os.makedirs(args.dir_output)
    if args.do_eval or args.eval_during_training:
        path_dev_data = os.path.join(args.dir_data, "valid.tsv")        
        assert os.path.exists(path_dev_data)
    if args.do_pred:
        path_test_data = os.path.join(args.dir_data, "test.tsv")        
        assert os.path.exists(path_test_data)
    if args.grad_accum_steps < 1:
        raise ValueError("Invalid grad_accum_steps parameter: {}, should be >= 1".format(
                            args.grad_accum_steps))

    # What GPUs do we use?
    if args.num_gpus == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        device_ids = None
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
        args.n_gpu = args.num_gpus
        if args.n_gpu > 1:
            device_ids = list(range(args.n_gpu))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1)))
    
    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load checkpoint. This contains a pre-trained model which may or
    # may not have been fine-tuned for language identification already
    logger.info("Loading checkpoint...")        
    checkpoint_path = os.path.join(args.dir_pretrained_model, "checkpoint.tar")        
    checkpoint_data = torch.load(checkpoint_path)

    # Check if lang2id is in checkpoint data (which is required unless
    # we are training)
    if "lang2id" in checkpoint_data:
        lang2id = checkpoint_data["lang2id"]
    else:
        lang2id = None
    if not args.do_train:
        assert lang2id is not None
    
    # Load config
    logger.info("Loading config...")
    config_path = os.path.join(args.dir_pretrained_model, "config.json")
    config = BertConfig.from_json_file(config_path)        

    # Create model and load pre-trained weigths
    logger.info("Loading model...")
    model = BertForMaskedLM(config)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(args.device)
    
    # Create pooler and load pretrained weights
    if "pooler_state_dict" in checkpoint_data:
        logger.info("Loading pooler...")
    else:
        logger.info("Making pooler...")
    pooler = Pooler(model.config.hidden_size, cls_only=(not args.avgpool))
    if "pooler_state_dict" in checkpoint_data:
        pooler.load_state_dict(checkpoint_data["pooler_state_dict"])
    pooler.to(args.device)

    # Distributed or parallel?
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed training.")
        model = DDP(model)
        pooler = DDP(pooler) 
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        pooler = torch.nn.DataParallel(pooler, device_ids=device_ids)

    # Log some info on the model
    logger.info("Model config: %s" % repr(model.config))
    logger.info("Nb params: %d" % count_params(model))
    logger.info("Nb params in pooler: %d" % count_params(pooler))        

    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_path = os.path.join(args.dir_pretrained_model, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Get data
    max_seq_length = args.seq_len + 2 # We add 2 for CLS and SEP
    if args.do_train:
        # Remove unk.train if present, and create a MLM dataset for it.
        path_unk = check_for_unk_train_data(train_paths)
        if path_unk is None:
            unk_dataset = None
        else:
            train_paths.remove(path_unk)
            logger.info("Loading MLM-only data from %s..." % path_unk)            
            unk_dataset = BertDatasetForMLM([path_unk],
                                            tokenizer,
                                            seq_len=max_seq_length,
                                            unk_only=True,
                                            sampling_distro=args.sampling_distro,
                                            encoding="utf-8",
                                            seed=args.seed)

        logger.info("Loading training data from %s training files in %s..." % (len(train_paths),args.dir_data))
        train_dataset = BertDatasetForClassification(train_paths,
                                                     tokenizer,
                                                     max_seq_length,
                                                     include_mlm=True,
                                                     sampling_distro=args.sampling_distro,
                                                     encoding="utf-8",
                                                     seed=args.seed)
        if path_unk is not None:
            assert len(unk_dataset) == len(train_dataset)
        lang2id = train_dataset.lang2id
        # Check lang2id: keys should contain all langs, and nothing else
        assert all(k in lang2id for k in ALL_LANGS)
        for k in lang2id:
            if k not in ALL_LANGS:
                msg = "lang2id contains invalid key '%s'" % k
                raise RuntimeError(msg)
        # Store lang2id in checkpoint data
        checkpoint_data["lang2id"] = lang2id
    if args.do_eval or args.eval_during_training:
        logger.info("Loading validation data from %s..." % path_dev_data)                                
        dev_dataset = BertDatasetForTesting(path_dev_data,
                                            tokenizer,
                                            lang2id,
                                            max_seq_length,
                                            require_labels=True,
                                            encoding="utf-8")
    if args.do_pred:
        logger.info("Loading test data from %s..." % path_test_data)                                
        test_dataset = BertDatasetForTesting(path_test_data,
                                             tokenizer,
                                             lang2id,
                                             max_seq_length,
                                             require_labels=False,
                                             encoding="utf-8")

    # Training
    if args.do_train:
        # Prepare optimizer and scheduler
        checkpoint_data["global_step"] = 0
        checkpoint_data["max_opt_steps"] = args.max_train_steps // args.grad_accum_steps
        num_opt_steps_per_epoch = int(len(train_dataset) / args.train_batch_size / args.grad_accum_steps)
        args.num_epochs = math.ceil(checkpoint_data["max_opt_steps"] / num_opt_steps_per_epoch)
        logger.info("Preparing optimizer...")
        np_list = list(model.named_parameters())
        np_list += list(pooler.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        opt_params = [
            {'params': [p for n, p in np_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in np_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.equal_betas:
            betas = (0.9, 0.9)
        else:
            betas = (0.9, 0.999)
        optimizer = AdamW(opt_params,
                          lr=args.learning_rate,
                          betas=betas,
                          correct_bias=args.correct_bias) # To reproduce BertAdam specific behaviour, use correct_bias=False
        logger.info("Preparing learning rate scheduler...")
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.num_warmup_steps,
                                                    num_training_steps=checkpoint_data["max_opt_steps"])

        # Log some info before training
        logger.info("*** Training info: ***")
        logger.info("Max training steps: %d" % args.max_train_steps)
        logger.info("Gradient accumulation steps: %d" % args.grad_accum_steps)
        logger.info("Max optimization steps: %d" % checkpoint_data["max_opt_steps"])
        logger.info("Training dataset size: %d" % len(train_dataset))
        logger.info("Batch size: %d" % args.train_batch_size)
        logger.info("# optimization steps/epoch: %d" % num_opt_steps_per_epoch)
        logger.info("# epochs to do: %d" % args.num_epochs)
        if args.eval_during_training:
            logger.info("Validation dataset size: %d" % len(dev_dataset))

        # Prepare training log file
        time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        train_log_path = os.path.join(args.dir_output, "%s.train.log" % time_str)        
        args.train_log_path = train_log_path

        # Run training
        train(model, pooler, optimizer, scheduler, train_dataset, args, unk_dataset=unk_dataset)

    if args.do_eval:
        pass
    if args.do_pred:
        pass
        
if __name__ == "__main__":
    main()
