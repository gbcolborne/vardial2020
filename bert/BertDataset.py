""" Dataset classes for training BERT """

import sys, os, random, logging
from io import open
import numpy as np
import torch
from torch.utils.data import Dataset
from iteround import saferound
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
    

class BertDatasetUnlabeled(Dataset):
    
    def __init__(self, train_paths, tokenizer, seq_len, sampling_distro="uniform", encoding="utf-8", seed=None):
        assert sampling_distro in ["uniform", "relfreq", "dampfreq"]
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

        if seed:
            random.seed(seed)            
            np.random.seed(seed)
            torch.manual_seed(seed)
        
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
                logger.info("Processing %s" % path)
                for line in f:
                    (text, label) = line_to_data(line, False)
                    if text is not None:
                        self.lang2freq[lang] += 1
                        self.total_dataset_size += 1
            # Check which of the 3 groups this lang belongs to
            group = get_lang_group(lang)
            self.group2freq[group] += self.lang2freq[lang]
        logger.info("Total dataset size: %d" % self.total_dataset_size)
        logger.info("Sampled dataset size: %d" % self.sampled_dataset_size)

        # Compute expected number of examples sampled from each language
        self.lang2samplesize = self.compute_expected_sample_sizes()
        logger.info("Sum of expected sample sizes per language: %d" % (sum(self.lang2samplesize.values())))

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
        elif self.sampling_distro in ["relfreq", "dampfreq"]:
            rel_counts = np.array([self.lang2freq[k] for k in rel_langs], dtype=np.float)
            con_counts = np.array([self.lang2freq[k] for k in con_langs], dtype=np.float)
            irr_counts = np.array([self.lang2freq[k] for k in irr_langs], dtype=np.float)
            rel_probs = rel_counts / rel_counts.sum()
            con_probs = con_counts / con_counts.sum()
            irr_probs = irr_counts / irr_counts.sum()                            
            if self.sampling_distro == "dampfreq":
                rel_probs_damp = rel_probs ** 0.5
                rel_probs = rel_probs_damp / rel_probs_damp.sum()
                con_probs_damp = con_probs ** 0.5
                con_probs = con_probs_damp / con_probs_damp.sum()                
                irr_probs_damp = irr_probs ** 0.5
                irr_probs = irr_probs_damp / irr_probs_damp.sum()                
        rel_sample_sizes = [int(x) for x in saferound(rel_probs * REL_SAMPLE_SIZE, 0, "largest")]
        con_sample_sizes = [int(x) for x in saferound(con_probs * CON_SAMPLE_SIZE, 0, "largest")]
        irr_sample_sizes = [int(x) for x in saferound(irr_probs * IRR_SAMPLE_SIZE, 0, "largest")]
        logger.info("  # samples (relevant): %d" % sum(rel_sample_sizes))
        logger.info("    Min samples/lang (relevant): %d" % min(rel_sample_sizes))
        logger.info("    Max samples/lang (relevant): %d" % max(rel_sample_sizes))
        logger.info("  # samples (confounders): %d" % sum(con_sample_sizes))        
        logger.info("    Min samples/lang (confounders): %d" % min(con_sample_sizes))
        logger.info("    Max samples/lang (confounders): %d" % max(con_sample_sizes))
        logger.info("  # samples (irrelevant): %d" % sum(irr_sample_sizes))                
        logger.info("    Min samples/lang (irrelevant): %d" % min(irr_sample_sizes))
        logger.info("    Max samples/lang (irrelevant): %d" % max(irr_sample_sizes))
        lang2samplesize = {}
        for i,x in enumerate(rel_sample_sizes):
            lang2samplesize[rel_langs[i]] = x
        for i,x in enumerate(con_sample_sizes):
            lang2samplesize[con_langs[i]] = x
        for i,x in enumerate(irr_sample_sizes):
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
        logger.info("guid: {}".format(example.guid))
        logger.info("tokens: {}".format(tokens))
        logger.info("input_ids: {}".format(input_ids))
        logger.info("input_mask: {}".format(input_mask))
        logger.info("segment_ids: {}".format(segment_ids))
        logger.info("lm_label_ids: {}".format(lm_label_ids))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids)
    return features
    
