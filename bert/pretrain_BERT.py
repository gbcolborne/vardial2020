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
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from CharTokenizer import CharTokenizer
from BertDataset import BertDatasetForMLM, BertDatasetForSPCAndMLM
from Pooler import Pooler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

    
def count_params(model):
    count = 0
    for p in model.parameters():
         count += torch.prod(torch.tensor(p.size())).item()
    return count


def accuracy(pred_scores, labels):
    ytrue = labels.cpu().numpy()
    ypred = pred_scores.detach().cpu().numpy()    
    ypred = np.argmax(ypred, axis=1)
    assert len(ytrue) == len(ypred)
    accuracy = np.sum(ypred == ytrue)/len(ytrue)
    return accuracy


def check_for_unk_train_data(train_paths):
    for path in train_paths:
        if os.path.split(path)[-1] == "unk.train":
            return path
    return None


def get_dataloader(dataset, batch_size, local_rank):
    if local_rank == -1:
        sampler = RandomSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def weighted_avg(vals, weights):
    vals = np.asarray(vals)
    weights = np.asarray(weights)
    assert len(vals.shape) == 1
    assert vals.shape == weights.shape
    probs = weights / weights.sum()
    return np.sum(vals * probs)    


def adjust_loss(loss, args):
    # Adapt loss for distributed training or gradient accumulation
    if args.n_gpu > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    return loss


def train_spc_and_mlm(model, pooler, tokenizer, optimizer, scheduler, dataset, args, train_log_path, mlm_dataset=None):
    """Pretrain a BertModelForMaskedLM using both sentence pair
    classification and MLM.
    
    Args:
    - model: BertModelForMaskedLM
    - pooler: Pooler used for SPC
    - tokenizer: CharTokenizer
    - optimizer
    - scheduler
    - dataset: BertDatasetForSPCandMLM
    - args
    - train_log_path
    - (Optional) mlm_dataset: a BertDatasetForMLM. If provided, we add
    a batch from this to every batch of the other dataset. This is
    useful for pre-training on the unlabeled test data, which we can
    not use for sentence pair classification.

    """
    if mlm_dataset is not None:
        assert len(dataset) == len(mlm_dataset)
        
    # Write header in log
    header = "GlobalStep\tLossMLM\tAccuracyMLM\tLossSPC\tAccuracySPC"
    if mlm_dataset is not None:
        header += "\tLossExtraMLM\tAccuracyExtraMLM"
    with open(train_log_path, "w") as f:
        f.write(header + "\n")
        
    # Start training
    global_step = 0  # Number of optimization steps (less than number of model calls if we accumulate gradients)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num training steps = %d", args.num_train_steps)
    logger.info("  Num accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Num optimization steps = %d", args.num_optimization_steps)
    model.train()
    for epoch in trange(int(args.num_epochs), desc="Epoch"):
        # Get fresh training samples
        if epoch > 0:
              dataset.resample()
              if mlm_dataset is not None:
                  mlm_dataset.resample()
                  
        # Make dataloader(s)
        dataloader = get_dataloader(dataset, args.train_batch_size, args.local_rank)
        if mlm_dataset is not None:
            mlm_dataloader = get_dataloader(mlm_dataset, args.train_batch_size, args.local_rank)
            assert len(mlm_dataloader) == len(dataloader)           
            mlm_batch_enum = enumerate(mlm_dataloader)
        
        # Some stats for this epoch
        real_batch_sizes = []
        query_mlm_losses = []
        query_mlm_accs = []
        spc_losses = []
        spc_accs = []
        extra_mlm_losses = []
        extra_mlm_accs = []
        
        # Run training for one epoch
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):            
            batch = tuple(t.to(args.device) for t in batch)
            input_ids_query = batch[0]
            input_mask_query = batch[1]
            segment_ids_query = batch[2]
            input_ids_cands = batch[3]
            input_mask_cands = batch[4]
            segment_ids_cands = batch[5]
            cand_labels = batch[6]
            lm_label_ids = batch[7]
            real_batch_sizes.append(len(input_ids_query))

            # Call underlying BERT model to get encodings of query and candidates
            outputs = model.bert(input_ids=input_ids_query,
                                 attention_mask=input_mask_query,
                                 token_type_ids=segment_ids_query,
                                 position_ids=None)
            query_last_hidden_states = outputs[0] # Last hidden states, shape (batch_size, seq_len, hidden_size)

            # Do MLM on last hidden states obtained using query
            # inputs.
            mlm_pred_scores = model.cls(query_last_hidden_states)

            # Get encodings of query and candidates for SPC
            query_encodings = pooler(query_last_hidden_states)
            all_cand_encodings = []            
            for i in range(2):
                outputs = model.bert(input_ids=input_ids_cands[:,i,:],
                                     attention_mask=input_mask_cands[:,i,:],
                                     token_type_ids=segment_ids_cands[:,i,:],
                                     position_ids=None)
                last_hidden_states = outputs[0]
                encodings = pooler(last_hidden_states)
                all_cand_encodings.append(encodings.unsqueeze(1))
            cand_encodings = torch.cat(all_cand_encodings, dim=1)            

            # Score candidates using dot(query, candidate)
            spc_scores = torch.bmm(cand_encodings, query_encodings.unsqueeze(2)).squeeze(2)

            # Do MLM on mlm_dataset if provided. First, upack batch
            extra_mlm_loss = None
            if mlm_dataset is not None:
                mlm_batch_id, mlm_batch = next(mlm_batch_enum)
                # Make sure the training steps are synced
                assert mlm_batch_id == step
                mlm_batch = tuple(t.to(args.device) for t in mlm_batch)
                xinput_ids, xinput_mask, xsegment_ids, xlm_label_ids = mlm_batch
                # Make sure the batch sizes are equal
                assert len(xinput_ids) == len(input_ids_query)
                outputs = model(input_ids=xinput_ids,
                                attention_mask=xinput_mask,
                                token_type_ids=xsegment_ids,
                                lm_labels=xlm_label_ids,
                                position_ids=None)
                extra_mlm_pred_scores = outputs[1]
                
            # Compute loss, do backprop. Compute accuracies.
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mlm_pred_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
            query_mlm_losses.append(loss.item())
            spc_loss = loss_fct(spc_scores, cand_labels)
            loss = loss + spc_loss
            spc_losses.append(spc_loss.item())
            if mlm_dataset is not None:
                extra_mlm_loss = loss_fct(extra_mlm_pred_scores.view(-1, model.config.vocab_size), xlm_label_ids.view(-1))
                loss = loss + extra_mlm_loss
                extra_mlm_losses.append(extra_mlm_loss.item())

            # Backprop
            loss = adjust_loss(loss, args)
            loss.backward()

            # Compute accuracies
            query_mlm_acc = accuracy(mlm_pred_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
            query_mlm_accs.append(query_mlm_acc)
            spc_acc = accuracy(spc_scores, cand_labels)
            spc_accs.append(spc_acc)
            if mlm_dataset is not None:
                extra_mlm_acc = accuracy(extra_mlm_pred_scores.view(-1, model.config.vocab_size), xlm_label_ids.view(-1))
                extra_mlm_accs.append(extra_mlm_acc)

            # Check if we accumulate grad or do an optimization step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if global_step >= args.num_optimization_steps:
                    break

        # Compute stats for this epoch
        avg_query_mlm_loss = weighted_avg(query_mlm_losses, real_batch_sizes)
        avg_query_mlm_acc = weighted_avg(query_mlm_accs, real_batch_sizes)
        avg_spc_loss = weighted_avg(spc_losses, real_batch_sizes)
        avg_spc_acc = weighted_avg(spc_accs, real_batch_sizes)
        if mlm_dataset is not None:
            avg_extra_mlm_loss = weighted_avg(extra_mlm_losses, real_batch_sizes)
            avg_extra_mlm_acc = weighted_avg(extra_mlm_accs, real_batch_sizes)
        
        # Write stats for this epoch in log
        log_data = []
        log_data.append(str(global_step))
        log_data.append("{:.5f}".format(avg_query_mlm_loss))
        log_data.append("{:.5f}".format(avg_query_mlm_acc))
        log_data.append("{:.5f}".format(avg_spc_loss))
        log_data.append("{:.5f}".format(avg_spc_acc))
        if mlm_dataset is not None:
            log_data.append("{:.5f}".format(avg_extra_mlm_loss))
            log_data.append("{:.5f}".format(avg_extra_mlm_acc))        
        
        with open(train_log_path, "a") as f:
            f.write("\t".join(log_data)+"\n")

        # Save model at end of each epoch
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)

        # Save state dict of pooler
        fn = os.path.join(args.output_dir, "pooler.pt")
        torch.save(pooler.state_dict(), fn)

        # Save tokenizer
        fn = os.path.join(args.output_dir, "tokenizer.pkl")
        with open(fn, "wb") as f:
            pickle.dump(tokenizer, f)

            
def train_mlm(model, tokenizer, optimizer, scheduler, dataset, args, train_log_path):
    """Pretrain a BertModelForMaskedLM using MLM only.
    
    Args:
    - model: BertModelForMaskedLM
    - tokenizer: CharTokenizer
    - optimizer
    - scheduler
    - dataset: BertDatasetForMLM
    - args
    - train_log_path

    """
    # Write header in log
    header = "GlobalStep\tLossMLM\tAccuracyMLM"
    with open(train_log_path, "w") as f:
        f.write(header + "\n")
        
    # Start training
    global_step = 0  # Number of optimization steps (less than number of model calls if we accumulate gradients)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num training steps = %d", args.num_train_steps)
    logger.info("  Num accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Num optimization steps = %d", args.num_optimization_steps)
    model.train()
    for epoch in trange(int(args.num_epochs), desc="Epoch"):
        # Get fresh training samples
        if epoch > 0:
              dataset.resample()

        # Make dataloader
        dataloader = get_dataloader(dataset, args.train_batch_size, args.local_rank)

        # Some stats for this epoch
        tr_loss = 0
        nb_tr_examples = 0
        real_batch_sizes = []
        accs = []
        
        # Run training for one epoch
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):            
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            real_batch_sizes.append(len(input_ids))
            
            # Call model.
            outputs = model(input_ids=input_ids,
                            attention_mask=input_mask,
                            token_type_ids=segment_ids,
                            lm_labels=lm_label_ids,
                            position_ids=None)
            loss = outputs[0]
            pred_scores = outputs[1]
            acc = accuracy(pred_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
            accs.append(acc)

            # Backprop
            loss = adjust_loss(loss, args)
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if global_step >= args.num_optimization_steps:
                    break
        avg_loss = tr_loss / nb_tr_examples
        avg_acc = weighted_avg(accs, real_batch_sizes)
        
        # Update training log
        log_data = []
        log_data.append(str(global_step))
        log_data.append("{:.5f}".format(avg_loss))
        log_data.append("{:.5f}".format(avg_acc))
        with open(train_log_path, "a") as f:
            f.write("\t".join(log_data)+"\n")

        # Save model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        fn = os.path.join(args.output_dir, "tokenizer.pkl")
        with open(fn, "wb") as f:
            pickle.dump(tokenizer, f)


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
    parser.add_argument("--avgpool_for_spc",
                        action="store_true",
                        help=("Use average pooling of all last hidden states, rather than just the last hidden state of CLS, to do SPC. "
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
                        help="Total number of training steps to perform. Note: # optimization steps = # train steps / # accumulation steps.")
    parser.add_argument("--num_warmup_steps",
                        default=10000,
                        type=int,
                        help="Number of optimization steps (i.e. training steps / accumulation steps) to perform linear learning rate warmup for. ")
    parser.add_argument('--gradient_accumulation_steps',
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
    
    # Check some other args
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    train_paths = glob.glob(os.path.join(args.dir_train_data, "*.train"))
    assert len(train_paths) > 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
                            
    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
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
    
            
    # Prepare model (and pooler if we are doing SPC)
    if pretrained:
        model = BertForMaskedLM.from_pretrained(args.bert_model_or_config_file)
    else:
        model = BertForMaskedLM(config)
    model.to(args.device)
    if not args.mlm_only:
        pooler = Pooler(model.config.hidden_size, cls_only=(not args.avgpool_for_spc))
        pooler.to(args.device)    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed training.")
        model = DDP(model)
        if not args.mlm_only:
            pooler = DDP(pooler) 
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        pooler = torch.nn.DataParallel(pooler, device_ids=device_ids)
    logger.info("Model config: %s" % repr(model.config))
    logger.info("Nb params: %d" % count_params(model))
    if not args.mlm_only:
        logger.info("Nb params in pooler: %d" % count_params(pooler))        

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

    # Comput number of optimization steps
    args.num_optimization_steps = args.num_train_steps // args.gradient_accumulation_steps
    if train_dataset_mlm is not None:
        num_opt_steps_per_epoch = int(len(train_dataset_mlm) / args.train_batch_size / args.gradient_accumulation_steps)
    elif train_dataset_spc is not None:
        num_opt_steps_per_epoch = int(len(train_dataset_spc) / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_opt_steps_per_epoch = num_opt_steps_per_epoch // torch.distributed.get_world_size()
    args.num_epochs = math.ceil(args.num_optimization_steps / num_opt_steps_per_epoch)    
    logger.info("Dataset size: %d" % (len(train_dataset_mlm) if train_dataset_mlm else len(train_dataset_spc)))
    logger.info("# optimization steps (for %d steps, # accumulation steps = %d): %d" % (args.num_train_steps,
                                                                                        args.gradient_accumulation_steps,
                                                                                        args.num_optimization_steps))
    logger.info("# opt. steps/epoch (with batch size = %d, # accumulation steps = %d): %d" % (args.train_batch_size,
                                                                                              args.gradient_accumulation_steps,
                                                                                              num_opt_steps_per_epoch))
    logger.info("# epochs (for %d steps, %d opt. steps): %d" % (args.num_train_steps, args.num_optimization_steps, args.num_epochs))
    
    # Prepare training log
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    train_log_path = os.path.join(args.output_dir, "%s.train.log" % time_str)
    
    # Prepare optimizer
    np_list = list(model.named_parameters())
    if not args.mlm_only:
        np_list += list(pooler.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in np_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in np_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      correct_bias=True) # To reproduce BertAdam specific behaviour, use correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_optimization_steps)
    
    # Train
    if args.mlm_only:
        train_mlm(model, tokenizer, optimizer, scheduler, train_dataset_mlm, args, train_log_path)
    else:
        train_spc_and_mlm(model, pooler, tokenizer, optimizer, scheduler, train_dataset_spc, args, train_log_path, mlm_dataset=train_dataset_mlm)


if __name__ == "__main__":
    main()
