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

import sys, os, random, argparse, logging, pickle, glob, math
from io import open
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from iteround import saferound
from CharTokenizer import CharTokenizer
sys.path.append("..")
from comp_utils import RELEVANT_LANGS, IRRELEVANT_URALIC_LANGS, ALL_LANGS

# Sample sizes of 3 language groups
REL_SAMPLE_SIZE = 20000
CON_SAMPLE_SIZE = 20000
IRR_SAMPLE_SIZE = 20000

# Label used in BertForLM to indicate a token is not masked.
NO_MASK_LABEL = -100

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_lang_group(lang):
    assert lang in ALL_LANGS
    if lang in RELEVANT_LANGS:
        return "rel"
    elif lang in IRRELEVANT_URALIC_LANGS:
        return "con"
    else:
        return "irr"


def get_nb_langs(group):
    assert group in ["rel", "con", "irr"]
    if group == "rel":
        return len(RELEVANT_LANGS)
    elif group == "con":
        return (len(IRRELEVANT_URALIC_LANGS))
    else:
        return len(ALL_LANGS) - len(RELEVANT_LANGS) - len(IRRELEVANT_URALIC_LANGS)

    
def count_params(model):
    count = 0
    for p in model.parameters():
         count += torch.prod(torch.tensor(p.size())).item()
    return count


def line_to_data(line, is_labeled):
    """ Takes a line from a dataset (labeled or unlabeled), and returns the text and label. """
    
    if is_labeled:
        elems = line.strip().split("\t")
        assert len(elems) == 2
        text = elems[0]
        label = elems[1]
    else:
        text = line.strip()
        label = None
    return (text, label)
    

class BERTDataset(Dataset):
    
    def __init__(self, train_paths, tokenizer, seq_len, sampling_distro="uniform", encoding="utf-8"):
        assert sampling_distro in ["uniform", "relfreq", "custom"]
        self.train_paths = train_paths # Paths of training files (names must match <lang>.train)
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.seq_len = seq_len
        self.sampling_distro = sampling_distro
        self.encoding = encoding        
        self.sample_counter = 0  # total number of examples sampled by calling __getitem__ (across all epochs)
        self.total_dataset_size = 0
        self.sampled_dataset_size = REL_SAMPLE_SIZE + CON_SAMPLE_SIZE + IRR_SAMPLE_SIZE
        self.lang2path = {}
        self.lang2file = {}
        self.lang2freq = {}
        self.group2freq = {"rel":0, "con":0, "irr":0}
        self.lang2ix = {}
        self.lang2samplesize = {}
        self.sampled_dataset = None
        
        # Prepare training files to sample lazily from disk
        for path in sorted(train_paths):
            fn = os.path.split(path)[-1]
            assert fn[-6:] == ".train"
            cut = fn.rfind(".")
            lang = fn[:cut]
            assert lang in ALL_LANGS
            self.lang2path[lang] = path
            # Open file to load lazily from disk later when we start sampling
            self.lang2file[lang] = open(path, 'r', encoding=encoding)
            self.lang2freq[lang] = 0
            self.lang2ix[lang] = 0
            # Count examples
            with open(path, 'r', encoding=encoding) as f:
                print("Processing %s" % path)
                for line in f:
                    (text, label) = line_to_data(line, False)
                    if text is not None:
                        self.lang2freq[lang] += 1
                        self.total_dataset_size += 1
            # Check which of the 3 groups this lang belongs to
            group = get_lang_group(lang)
            self.group2freq[group] += self.lang2freq[lang]
        print("Total dataset size: %d" % self.total_dataset_size)
        print("Sampled dataset size: %d" % self.sampled_dataset_size)

        # Compute expected number of examples sampled from each language
        self.lang2samplesize = self.compute_expected_sample_sizes()
        print("Sum of expected sample sizes per language: %d" % (sum(self.lang2samplesize.values())))

        # Sample a training set
        self.resample()
        return
    
        
    def compute_expected_sample_sizes(self):
        # Map langs in the 3 groups to IDs
        rel_langs = sorted(RELEVANT_LANGS)
        con_langs = sorted(IRRELEVANT_URALIC_LANGS)
        irr_langs = sorted(ALL_LANGS.difference(RELEVANT_LANGS).difference(IRRELEVANT_URALIC_LANGS))
        rel_lang2id = dict((l,i) for (i,l) in enumerate(rel_langs))
        con_lang2id = dict((l,i) for (i,l) in enumerate(con_langs))
        irr_lang2id = dict((l,i) for (i,l) in enumerate(irr_langs))
        if self.sampling_distro == "uniform":
            rel_probs = np.ones(len(rel_langs), dtype=float) / len(rel_langs)
            con_probs = np.ones(len(con_langs), dtype=float) / len(con_langs)
            irr_probs = np.ones(len(irr_langs), dtype=float) / len(irr_langs)
        if self.sampling_distro == "relfreq":
            rel_counts = np.array([self.lang2freq[k] for k in rel_langs], dtype=np.float)
            rel_probs = rel_counts / rel_counts.sum()
            con_counts = np.array([self.lang2freq[k] for k in con_langs], dtype=np.float)
            con_probs = con_counts / con_counts.sum()
            irr_counts = np.array([self.lang2freq[k] for k in irr_langs], dtype=np.float)
            irr_probs = irr_counts / irr_counts.sum()
        if self.sampling_distro == "custom":
            raise NotImplementedError
        rel_dev_counts = [int(x) for x in saferound(rel_probs * REL_SAMPLE_SIZE, 0, "largest")]
        con_dev_counts = [int(x) for x in saferound(con_probs * CON_SAMPLE_SIZE, 0, "largest")]
        irr_dev_counts = [int(x) for x in saferound(irr_probs * IRR_SAMPLE_SIZE, 0, "largest")]
        lang2samplesize = {}
        for i,x in enumerate(rel_dev_counts):
            lang2samplesize[rel_langs[i]] = x
        for i,x in enumerate(con_dev_counts):
            lang2samplesize[con_langs[i]] = x
        for i,x in enumerate(irr_dev_counts):
            lang2samplesize[irr_langs[i]] = x
        return lang2samplesize

    
    def resample(self):
        """ Sample dataset by lazily loading part of the data from disk. """
        data = []
        for lang in self.lang2path.keys():
            # Check how many examples we sample for this lang
            sample_size = self.lang2samplesize[lang]
            for _ in range(sample_size):
                # Check if we have reached EOF
                if self.lang2ix[lang] >= (self.lang2freq[lang]-1):
                    self.lang2file[lang].close()
                    self.lang2file[lang] = open(self.lang2path[lang], "r", encoding=self.encoding)
                    self.lang2ix[lang] = 0
                line = next(self.lang2file[lang])
                self.lang2ix[lang] += 1
                (text,_) = line_to_data(line, False)
                assert text is not None and len(text)
                data.append(text)
        assert len(data) == self.sampled_dataset_size
        # Shuffle
        np.random.shuffle(data)
        self.sampled_dataset = data
        return None

    
    def __len__(self):
        return self.sampled_dataset_size

    
    def __getitem__(self, item):
        t = self.sampled_dataset[item]
        example_id = self.sample_counter
        self.sample_counter += 1
        tokens = self.tokenizer.tokenize(t)
        example = InputExample(guid=example_id, tokens=tokens)
        features = convert_example_to_features(example, self.seq_len, self.tokenizer)
        tensors = (torch.tensor(features.input_ids),
                   torch.tensor(features.input_mask),
                   torch.tensor(features.segment_ids),
                   torch.tensor(features.lm_label_ids))
        return tensors


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens: list of strings. The tokens.
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
            output_label.append(NO_MASK_LABEL)

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
    lm_label_ids = ([NO_MASK_LABEL] + t_label + [NO_MASK_LABEL])

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
        lm_label_ids.append(NO_MASK_LABEL)

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
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
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
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
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
    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")
    train_paths = glob.glob(os.path.join(args.dir_train_data, "*.train"))
    assert len(train_paths) > 0
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
                            
    # Seed RNGs
    random.seed(args.seed)
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
    print(model.config)
    print("Nb params: %d" % count_params(model))


    # Get training data
    max_seq_length = args.seq_len + 2 # We add 2 for CLS and SEP
    if args.do_train:        
        print("Preparing dataset using data from %s" % args.dir_train_data)
        train_dataset = BERTDataset(train_paths,
                                    tokenizer,
                                    seq_len=max_seq_length,
                                    sampling_distro="uniform",
                                    encoding="utf-8")

        num_steps_per_epoch = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) 
        if args.local_rank != -1:
            num_steps_per_epoch = num_steps_per_epoch // torch.distributed.get_world_size()
        num_epochs = math.ceil(args.num_train_steps / num_steps_per_epoch)
        print("  Dataset size: %d" % len(train_dataset))
        print("  # steps/epoch (with batch size = %d, # accumulation steps = %d): %d" % (args.train_batch_size,
                                                                                         args.gradient_accumulation_steps,
                                                                                         num_steps_per_epoch))
        print("  # epochs (for %d steps): %d" % (args.num_train_steps, num_epochs))
        
    # Prepare training log
    output_log_file = os.path.join(args.output_dir, "training_log.txt")
    with open(output_log_file, "w") as f:
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
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(num_epochs), desc="Epoch"):
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
