# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pre-train BERT.

Code based on: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_lm_finetuning.py.

"""

import os, argparse, logging, pickle, glob, math
from io import open
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from CharTokenizer import CharTokenizer
from BertDataset import BertDatasetForMLM, BertDatasetForSPCAndMLM


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


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model_or_config_file", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Directory containing pre-trained BERT model or path of configuration file (if no pre-training).")
    parser.add_argument("--dir_train_data",
                        default=None,
                        type=str,
                        required=True,
                        help="Path of a directory containing training files (names must match <lang>.train) and vocab.txt")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--mlm_only",
                        action="store_true",
                        help=("Use only masked language modeling, no sentence pair classification "
                              " (e.g. if you only have unk.train in your training directory)"))
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
    parser.add_argument("--min_freq",
                        default=1,
                        type=int,
                        help="Minimum character frequency. Characters whose frequency is under this threshold will be mapped to <UNK>")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_steps",
                        default=1000000,
                        type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--num_warmup_steps",
                        default=10000,
                        type=int,
                        help="Number of training steps to perform linear learning rate warmup for. ")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
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
    
    # Check whether bert_model_or_config_file is a file or directory
    if os.path.isdir(args.bert_model_or_config_file):
        pretrained=True
        targets = [WEIGHTS_NAME, CONFIG_NAME, "tokenizer.pkl"]
        for t in targets:
            path = os.path.join(args.bert_model_or_config_file, t)
            if not os.path.exists(path):
                msg = "File '{}' not found".format(path)
                raise ValueError(msg)
        fp = os.path.join(args.bert_model_or_config_file, CONFIG_NAME)
        config = BertConfig(fp)
    else:
        pretrained=False
        config = BertConfig.from_json_file(args.bert_model_or_config_file)
        
    # What GPUs do we use?
    if args.num_gpus == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        device_ids = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
        n_gpu = args.num_gpus
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))
    
    # Check some other args
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_paths = glob.glob(os.path.join(args.dir_train_data, "*.train"))
    assert len(train_paths) > 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
                            
    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer
    if pretrained:
        fp = os.path.join(args.bert_model_or_config_file, "tokenizer.pkl")
        with open(fp, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        path_vocab = os.path.join(args.dir_train_data, "vocab.txt")
        assert os.path.exists(path_vocab)
        tokenizer = CharTokenizer(path_vocab)
        if args.min_freq > 1:
            tokenizer.trim_vocab(args.min_freq)
        # Adapt vocab size in config
        config.vocab_size = len(tokenizer.vocab)
        logger.info("Size of vocab: {}".format(len(tokenizer.vocab)))
    
            
    # Prepare model
    if pretrained:
        model = BertForMaskedLM.from_pretrained(args.bert_model_or_config_file)
    else:
        model = BertForMaskedLM(config)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info("Model config: %s" % repr(model.config))
    logger.info("Nb params: %d" % count_params(model))

    # Check if there is unk training data. 
    path_unk = check_for_unk_train_data(train_paths)

    # Get training data
    max_seq_length = args.seq_len + 2 # We add 2 for CLS and SEP
    logger.info("Preparing dataset using data from %s" % args.dir_train_data)
    if args.mlm_only:
        # We only want to do MLM
        train_dataset_spc = None
        train_dataset_mlm = BertDatasetForMLM(train_paths,
                                              tokenizer,
                                              unk_only=False,
                                              seq_len=max_seq_length,
                                              sampling_distro=args.sampling_distro,
                                              encoding="utf-8",
                                              seed=args.seed)
    else:
        # We want do to SLC and MLM. If unk data is present, we remove
        # it from the paths provided to BertLabeledDataset.
        if path_unk is not None:
            train_paths.remove(path_unk)
        train_dataset_spc = BertDatasetForSPCAndMLM(train_paths,
                                                    tokenizer,
                                                    seq_len=max_seq_length,
                                                    sampling_distro=args.sampling_distro,
                                                    encoding="utf-8",
                                                    seed=args.seed)
        if path_unk is None:
            train_dataset_mlm = None
        else:
            # In this case we use a BertDatasetForMLM for the unk
            # data. Both datasets will be of the same size. The latter
            # is used for MLM only.
            train_dataset_mlm = BertDatasetForMLM([path_unk],
                                                  tokenizer,
                                                  unk_only=True,
                                                  seq_len=max_seq_length,
                                                  sampling_distro=args.sampling_distro,
                                                  encoding="utf-8",
                                                  seed=args.seed)
            assert len(train_dataset_spc) == len(train_dataset_mlm)

    sys.exit()
    #### TODO: Adapt samplers and dataloaders to be dual like the train dataset: mlm and spc
        
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    num_steps_per_epoch = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) 
    if args.local_rank != -1:
        num_steps_per_epoch = num_steps_per_epoch // torch.distributed.get_world_size()
    num_epochs = math.ceil(args.num_train_steps / num_steps_per_epoch)
    logger.info("Dataset size: %d" % len(train_dataset))
    logger.info("# steps/epoch (with batch size = %d, # accumulation steps = %d): %d" % (args.train_batch_size,
                                                                                     args.gradient_accumulation_steps,
                                                                                     num_steps_per_epoch))
    logger.info("# epochs (for %d steps): %d" % (args.num_train_steps, num_epochs))
        
    # Prepare training log
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    train_log_path = os.path.join(args.output_dir, "%s.train.log" % time_str)
    with open(train_log_path, "w") as f:
        f.write("Steps\tTrainLoss\n")
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      correct_bias=True) # To reproduce BertAdam specific behaviour, use correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_train_steps)
    
    # Start training
    global_step = 0
    total_tr_steps = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", args.num_train_steps)
    model.train()
    for epoch in trange(int(num_epochs), desc="Epoch"):
        # Get fresh training samples
        if epoch > 0:
            train_dataset.resample()
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_dataset)
            else:
                train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            # Call model. Note: if position_ids is None, they
            # assume input IDs are in order starting at position 0
            outputs = model(input_ids=input_ids,
                            attention_mask=input_mask,
                            token_type_ids=segment_ids,
                            lm_labels=lm_label_ids,
                            position_ids=None)
            loss = outputs[0]
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if global_step >= args.num_train_steps:
                    break
        avg_loss = tr_loss / nb_tr_examples

        # Update training log
        total_tr_steps += nb_tr_steps
        log_data = [str(total_tr_steps), "{:.5f}".format(avg_loss)]
        with open(train_log_path, "a") as f:
            f.write("\t".join(log_data)+"\n")

        # Save model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        fn = os.path.join(args.output_dir, "tokenizer.pkl")
        with open(fn, "wb") as f:
            pickle.dump(tokenizer, f)


if __name__ == "__main__":
    main()
