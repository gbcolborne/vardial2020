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

import os, random, argparse, logging, pickle
from io import open
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm, trange

from CharTokenizer import CharTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def line_to_data(line):
    """ Takes a line from a labeled dataset, and returns the text and label. """
    elems = line.strip().split("\t")
    assert len(elems == 2)
    return (elems[0], elems[1])


class BERTDataset(Dataset):
    
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", nb_corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.nb_corpus_lines = nb_corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.sample_counter = 0  # used to keep track of how many times we have sampled from the dataset (across all epochs)

        # load samples into memory
        if on_memory:
            self.lines = []
            doc = []
            self.nb_corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=nb_corpus_lines):
                    (text, label) = line_to_data(line)
                    if text is not None:
                        self.lines.append(line)
            self.nb_corpus_lines = len(self.lines)

        # load samples later lazily from disk
        else:
            if self.nb_corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.nb_corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=nb_corpus_lines):
                        (text, label) = line_to_data(line)
                        if text is not None:
                            self.nb_corpus_lines += 1
            self.file = open(corpus_path, "r", encoding=encoding)

            
    def __len__(self):
        return self.nb_corpus_lines 

    
    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        t = self.get_line(item)

        # tokenize
        tokens = self.tokenizer.tokenize(t)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens=tokens)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)
        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids))
        return cur_tensors

    
    def get_line(self, item):
        """
        Get one sample from corpus consisting of a single line.
        :param item: int, index of sample.
        :return: str, a sentence from corpus
        """
        t = ""
        assert item < self.nb_corpus_lines
        if self.on_memory:
            t = self.lines[item]
            return t
        else:
            line = next(self.file)
            (t,_) = line_to_data(line)
        assert t is not None
        return t


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens: string. The untokenized text of a sequence. 
            labels: (Optional) string. The language model labels of the example.
        """
        self.guid = guid
        self.tokens = tokens
        self.lm_labels = lm_labels 


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (a sentence as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens = example.tokens

    # Modify `tokens` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP] with "- 2"
    tokens = tokens[:max_seq_length-2]

    tokens, t_label = random_word(tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP
    lm_label_ids = ([-1] + t_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    out_tokens = []
    segment_ids = []
    out_tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens:
        out_tokens.append(token)
        segment_ids.append(0)
    out_tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(out_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids)
    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model_or_config_file", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Directory containing pre-trained BERT model or path of configuration file (if no pre-training).")
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps"
                        default=0,
                        type=int,
                        help="Number of training steps to perform linear learning rate warmup for. ")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Num GPUs to use for training (0 for none, -1 for all available)")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

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
        config = BertConfig(args.bert_model_or_config_file)

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
    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    # Seed RNGs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare output directory
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer
    if pretrained:
        fp = os.path.join(args.bert_model_or_config_file, "tokenizer.pkl")
        with open(fp, "rb") as f:
            tokenizer = pickle.load(f)

    # Get training data
    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_file)
        train_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length,
                                    nb_corpus_lines=None, on_memory=args.on_memory)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        training_data = [line.strip() for line in open(args.train_file).readlines()]
        tokenizer = CharTokenizer()
        # Train tokenizer
        for i range(len(train_dataset)):
            tokenizer.update_vocab(train_dataset.get_line(i))
        tokenizer.trim_vocab(config.min_freq)
        # Adapt vocab size in config
        config.vocab_size = len(tokenizer.vocab)
        print("Size of vocab: {}".format(len(tokenizer.vocab)))

            
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_train_optimization_steps)

    # Prepare training log
    output_log_file = os.path.join(args.output_dir, "training_log.txt")
    with open(output_log_file, "w") as f:
        f.write("Steps\tTrainLoss\n")

    # Start training
    global_step = 0
    total_tr_steps = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids = batch
                outputs = model(input_ids, segment_ids, input_mask, lm_label_ids)
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
            avg_loss = tr_loss / nb_tr_examples

            # Update training log
            total_tr_steps += nb_tr_steps
            log_data = [str(total_tr_steps), "{:.5f}".format(avg_loss)]
            with open(output_log_file, "a") as f:
                f.write("\t".join(log_data)+"\n")

            # Save model
            logger.info("** ** * Saving model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            model_to_save.save_pretrained(args.output_dir)
            fn = os.path.join(args.output_dir, "tokenizer.pkl")
            with open(fn, "wb") as f:
                pickle.dump(tokenizer, f)


if __name__ == "__main__":
    main()
