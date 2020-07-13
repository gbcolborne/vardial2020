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
UNK_SAMPLE_SIZE = 20000

# Label used in BertForLM to indicate a token is not masked.
NO_MASK_LABEL = -100

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def check_for_unk_train_data(train_paths):
    for path in train_paths:
        if os.path.split(path)[-1] == "unk.train":
            return path
    return None


def get_lang_group(lang):
    if lang == "unk":
        return "unk"
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


def mask_random_tokens(tokens, tokenizer):
    """
    Masking some random tokens for masked language modeling with probabilities as in the original BERT paper.

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


class BertDatasetForTraining(Dataset):

    """Abstract class for BertDataset used for training. Implements
    sampling of data from disk, as we can not load all the training
    data into memory. Subclasses must implement `__getitem__`, as well
    as `resample`.

    """

    def __init__(self, train_paths, tokenizer, seq_len, size, unk_only=False, sampling_distro="uniform", encoding="utf-8", seed=None, verbose=False):
        assert sampling_distro in ["uniform", "relfreq", "dampfreq"]
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.sampled_dataset_size = size # Expected size (will be enforced)        
        self.unk_only = unk_only
        self.seq_len = seq_len # Includes CLS and SEP tokens
        self.sampling_distro = sampling_distro
        self.encoding = encoding
        self.verbose = verbose
        self.sample_counter = 0  # total number of examples sampled by calling __getitem__ (across all epochs)
        self.total_dataset_size = 0
        self.lang_list = []
        self.lang2id = {} # Maps to indices in lang_list
        self.lang2path = {}
        self.lang2file = {}
        self.lang2freq = {}
        self.group2freq = {"rel":0, "con":0, "irr":0, "unk":0}
        self.lang2ix = {} # Maps to the current index in the training file
        self.lang2samplesize = {}
        
        if seed:
            random.seed(seed)            
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Set train_paths
        for path in sorted(train_paths):
            filename = os.path.split(path)[-1]
            assert filename[-6:] == ".train"
            cut = filename.rfind(".")
            lang = filename[:cut]
            assert lang in ALL_LANGS or lang == "unk"
            if lang == "unk" or not self.unk_only:
                self.lang2path[lang] = path                            
        assert len(self.lang2path) > 0
        if self.unk_only:
            assert len(self.lang2path) == 1
            assert 'unk' in self.lang2path
                
        # Prepare training files to sample lazily from disk
        for lang, path in sorted(self.lang2path.items(), key=lambda x:x[0], reverse=False):
            # Open file to load lazily from disk later when we start sampling
            self.lang2file[lang] = open(self.lang2path[lang], 'r', encoding=self.encoding)
            self.lang2freq[lang] = 0
            self.lang2ix[lang] = 0
            # Count examples
            with open(path, 'r', encoding=self.encoding) as f:
                logger.info("Processing %s" % path)
                for line in f:
                    (text, label) = line_to_data(line, False)
                    if text is not None:
                        self.lang2freq[lang] += 1
                        self.total_dataset_size += 1
            # Check which of the 3 groups this lang belongs to
            group = get_lang_group(lang)
            self.group2freq[group] += self.lang2freq[lang]
        self.lang_list = sorted(self.lang2freq.keys())
        self.lang2id = {x:i for i,x in enumerate(self.lang_list)}
        logger.info("Total dataset size: %d" % self.total_dataset_size)
        logger.info("Sampled dataset size (expectected): %d" % self.sampled_dataset_size)

        # Compute expected number of examples sampled from each language
        self.lang2samplesize = self.compute_expected_sample_sizes()
        logger.info("Sum of expected sample sizes per language: %d" % (sum(self.lang2samplesize.values())))
        assert sum(self.lang2samplesize.values()) == self.sampled_dataset_size
        return


    def __len__(self):
        return self.sampled_dataset_size
    

    def compute_expected_sample_sizes(self):
        if self.unk_only:
            lang2samplesize = {'unk': self.sampled_dataset_size}
            return lang2samplesize
        
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
        lang2samplesize = {}
        for i,x in enumerate(rel_sample_sizes):
            lang2samplesize[rel_langs[i]] = x
        for i,x in enumerate(con_sample_sizes):
            lang2samplesize[con_langs[i]] = x
        for i,x in enumerate(irr_sample_sizes):
            lang2samplesize[irr_langs[i]] = x
        if 'unk' in self.lang2path:            
            lang2samplesize['unk'] = UNK_SAMPLE_SIZE
        logger.info("  # samples (relevant): %d" % sum(rel_sample_sizes))
        logger.info("    Min samples/lang (relevant): %d" % min(rel_sample_sizes))
        logger.info("    Max samples/lang (relevant): %d" % max(rel_sample_sizes))
        logger.info("  # samples (confounders): %d" % sum(con_sample_sizes))        
        logger.info("    Min samples/lang (confounders): %d" % min(con_sample_sizes))
        logger.info("    Max samples/lang (confounders): %d" % max(con_sample_sizes))
        logger.info("  # samples (irrelevant): %d" % sum(irr_sample_sizes))                
        logger.info("    Min samples/lang (irrelevant): %d" % min(irr_sample_sizes))
        logger.info("    Max samples/lang (irrelevant): %d" % max(irr_sample_sizes))
        if 'unk' in self.lang2path:
            logger.info("  # samples (unknown): %d" % UNK_SAMPLE_SIZE)            
        return lang2samplesize

    
class InputExampleForMLM(object):
    """A single training/test example for masked language modeling only."""

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
        

class InputFeaturesForMLM(object):
    """A single set of features of data for masked language modeling only."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids):
        self.input_ids = input_ids # List input token IDs
        self.input_mask = input_mask # List containing input mask 
        self.segment_ids = segment_ids # List containing token type (segment) IDs
        self.lm_label_ids = lm_label_ids # List containing LM label IDs 
    
    
class BertDatasetForMLM(BertDatasetForTraining):
    
    def __init__(self, train_paths, tokenizer, seq_len, unk_only=False, sampling_distro="uniform", encoding="utf-8", seed=None, verbose=False):
        # Init parent class
        size = REL_SAMPLE_SIZE + CON_SAMPLE_SIZE + IRR_SAMPLE_SIZE
        if not unk_only and check_for_unk_train_data(train_paths) is not None:
            size += UNK_SAMPLE_SIZE
        super().__init__(train_paths, tokenizer, seq_len, size, unk_only=unk_only, sampling_distro=sampling_distro, encoding=encoding, seed=seed, verbose=verbose)
        
        # Sample a training set
        self.resample() 
        return
    
    
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

    
    def __getitem__(self, item):
        t = self.sampled_dataset[item]
        example_id = self.sample_counter
        self.sample_counter += 1
        tokens = self.tokenizer.tokenize(t)
        example = InputExampleForMLM(guid=example_id, tokens=tokens)
        features = self._convert_example_to_features(example)
        tensors = (torch.tensor(features.input_ids),
                   torch.tensor(features.input_mask),
                   torch.tensor(features.segment_ids),
                   torch.tensor(features.lm_label_ids))
        return tensors

    
    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for MLM only, with IDs, LM labels,
        input_mask, CLS and SEP tokens etc.
        
        :param example: InputExampleForMLM, containing sentence input as lists of tokens.
        :return: InputFeaturesForMLM, containing all inputs and labels of one sample as IDs (as used for model training)

        """
        tokens = example.tokens
        
        # Truncate sequence if necessary. Account for [CLS] and [SEP] by subtracting 2.
        tokens = tokens[:self.seq_len-2]
        
        # Mask tokens for MLM
        tokens, lm_label_ids = mask_random_tokens(tokens, self.tokenizer)

        # Add CLS and SEP
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        lm_label_ids = [NO_MASK_LABEL] + lm_label_ids + [NO_MASK_LABEL]
        
        # Get input token IDs (unpadded)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    
        # Zero-pad input token IDs
        input_ids += [0] * (self.seq_len - len(tokens))
    
        # Zero-pad labels
        lm_label_ids += [NO_MASK_LABEL] * (self.seq_len - len(tokens))

        # Make input mask (1 for real tokens and 0 for padding tokens)
        input_mask = [1] * len(tokens) + [0] * (self.seq_len - len(tokens))
    
        # Make segment IDs (padded)
        segment_ids = [0] * self.seq_len

        # Check data
        assert len(input_ids) == self.seq_len
        assert len(input_mask) == self.seq_len
        assert len(segment_ids) == self.seq_len
        assert len(lm_label_ids) == self.seq_len
        
        if self.verbose and example.guid < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(tokens))
            logger.info("input_ids: {}".format(input_ids))
            logger.info("input_mask: {}".format(input_mask))
            logger.info("segment_ids: {}".format(segment_ids))
            logger.info("lm_label_ids: {}".format(lm_label_ids))

        features = InputFeaturesForMLM(input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       lm_label_ids=lm_label_ids)
        return features

    
class InputExampleForSPCAndMLM(object):
    """A single training/test example for masked language modeling and sentence pair classification. """

    def __init__(self, guid, tokens_query, tokens_pos, tokens_neg, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_query: list of strings. The tokens of the query (text sample).
            tokens_pos: list of strings. The tokens of the positive candidate for this query (from same language).
            tokens_neg: list of strings. The tokens of the negative candidate for this query (from other language).
            lm_labels: (Optional) string. The language model labels of the example.
        """
        self.guid = guid
        self.tokens_query = tokens_query
        self.tokens_pos = tokens_pos
        self.tokens_neg = tokens_neg
        self.lm_labels = lm_labels 

        
class InputFeaturesForSPCAndMLM(object):
    """A single set of features of data for both masked language modeling and sentence pair classification."""

    def __init__(self, input_ids_query, input_mask_query, segment_ids_query, input_ids_cands, input_mask_cands, segment_ids_cands, pos_cand_id, lm_label_ids_query):
        self.input_ids_query = input_ids_query # List of query input IDs
        self.input_mask_query = input_mask_query # List containing input mask of query        
        self.segment_ids_query = segment_ids_query # List of query token type (segment) IDs
        self.input_ids_cands = input_ids_cands # List of lists of candidate input IDs
        self.input_mask_cands = input_mask_cands # List of lists containing input mask of candidates
        self.segment_ids_cands = segment_ids_cands # List of lists of candidate token type (segment) IDs
        self.pos_cand_id = pos_cand_id # Integer index of the positive candidate (among the candidates for this example)
        self.lm_label_ids_query = lm_label_ids_query # List containing LM label IDs of query


class BertDatasetForSPCAndMLM(BertDatasetForTraining):
    
    def __init__(self, train_paths, tokenizer, seq_len, sampling_distro="uniform", encoding="utf-8", seed=None, verbose=False):
        size = REL_SAMPLE_SIZE + CON_SAMPLE_SIZE + IRR_SAMPLE_SIZE
        
        # Init parent class
        super().__init__(train_paths, tokenizer, seq_len, size, sampling_distro=sampling_distro, encoding=encoding, seed=seed, verbose=verbose)

        # Compute sampling probabilities for negative candidates
        counts = np.array([self.lang2samplesize[k] for k in self.lang_list], dtype=float)
        probs = counts / counts.sum()
        # Dampen
        damp = probs ** 0.5
        probs = damp / damp.sum()
        self.neg_sampling_probs = probs

        # Make a buffer of negative candidates
        self.neg_buffer_size = 2*len(self)
        self.neg_buffer = self._get_neg_buffer()
        self.neg_buffer_ix = 0

        # Sample a training set
        self.resample()
        return


    def _get_neg_buffer(self):
         return np.random.choice(np.arange(0,len(self.lang_list)), self.neg_buffer_size, replace=True, p=self.neg_sampling_probs)

    
    def _sample_lang_id_for_neg_sampling(self):
        """ Sample language for negative sampling. Return ID of language. """
        sampled_id = self.neg_buffer[self.neg_buffer_ix]
        self.neg_buffer_ix += 1
        if self.neg_buffer_ix == self.neg_buffer_size:
            self.neg_buffer = self._get_neg_buffer()
            self.neg_buffer_ix = 0
        return sampled_id

    
    def resample(self):
        """ Sample dataset by lazily loading part of the data from disk. """
        lang2texts = {}
        for lang in self.lang_list:
            # Check how many examples we sample for this lang
            sample_size = self.lang2samplesize[lang]
            texts = []
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
                texts.append(text)
            lang2texts[lang] = texts
        assert sum(len(x) for x in lang2texts.values()) == self.sampled_dataset_size
        
        # Now pick a positive candidate and a negative candidate for each query
        queries = []
        pos_candidates = []
        neg_candidates = []
        for lang in self.lang_list:
            texts = lang2texts[lang]
            if len(texts) < 1:
                msg = "We must have 2 samples from each language. "
                msg += "Increase sample size or use a different sampling_distro."
                raise RuntimeError(msg)

            # Loop over queries
            for i in range(len(texts)):
                queries.append(texts[i])
                
                # Pick a positive candidate (same language)
                other_indices = list(range(0,i)) + list(range(i+1,len(texts)))
                pos_ix = random.choice(other_indices)
                pos_candidates.append(texts[pos_ix])
                
                # Sample negative candidate from another language. First
                # we sample the language, according to their relative
                # frequency in our sample of texts
                sampled_lang = None
                while sampled_lang is None:
                    sampled_id = self._sample_lang_id_for_neg_sampling()
                    if self.lang_list[sampled_id] != lang:
                        sampled_lang = self.lang_list[sampled_id]
                # Now we sample a text at random
                neg_candidate = random.choice(lang2texts[sampled_lang])
                neg_candidates.append(neg_candidate)
        assert len(queries) == len(pos_candidates)
        assert len(queries) == len(neg_candidates)
        data = list(zip(queries, pos_candidates, neg_candidates))
        
        # Shuffle
        np.random.shuffle(data)
        self.sampled_dataset = data
        return None

    
    def __getitem__(self, item):
        (q,p,n) = self.sampled_dataset[item]
        example_id = self.sample_counter
        self.sample_counter += 1
        tokens_query = self.tokenizer.tokenize(q)
        tokens_pos = self.tokenizer.tokenize(p)
        tokens_neg = self.tokenizer.tokenize(n)        
        example = InputExampleForSPCAndMLM(guid=example_id, tokens_query=tokens_query, tokens_pos=tokens_pos, tokens_neg=tokens_neg)
        features = self._convert_example_to_features(example)
        tensors = (torch.tensor(features.input_ids_query),
                   torch.tensor(features.input_mask_query),
                   torch.tensor(features.segment_ids_query),
                   torch.tensor(features.input_ids_cands),
                   torch.tensor(features.input_mask_cands),
                   torch.tensor(features.segment_ids_cands),
                   torch.tensor(features.pos_cand_id),
                   torch.tensor(features.lm_label_ids_query))
        return tensors
    

    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for both MLM and SPC, with IDs, LM labels,
        input_mask, candidate labels for sentence pair classification, CLS
        and SEP tokens, etc.

        :param example: InputExampleForSPCAndMLM, containing sentence inputs as lists of tokens.

        :return: InputFeaturesForSPCAndMLM, containing all inputs and labels of one sample as IDs (as used for model training)

        """
        tokens_query = example.tokens_query
        tokens_pos = example.tokens_pos
        tokens_neg = example.tokens_neg
    
        # Truncate sequence if necessary. Account for [CLS] and [SEP] by subtracting 2.
        tokens_query = tokens_query[:self.seq_len-2]
        tokens_pos = tokens_pos[:self.seq_len-2]
        tokens_neg = tokens_neg[:self.seq_len-2]    
    
        # Mask tokens for MLM
        tokens_query, lm_label_ids_query = mask_random_tokens(tokens_query, self.tokenizer)

        # Add CLS and SEP
        tokens_query = ["[CLS]"] + tokens_query + ["[SEP]"]
        tokens_pos = ["[CLS]"] + tokens_pos + ["[SEP]"]
        tokens_neg = ["[CLS]"] + tokens_neg + ["[SEP]"]
        lm_label_ids_query = [NO_MASK_LABEL] + lm_label_ids_query + [NO_MASK_LABEL]

        # Get input token IDs (unpadded)
        input_ids_query = self.tokenizer.convert_tokens_to_ids(tokens_query)
        input_ids_pos = self.tokenizer.convert_tokens_to_ids(tokens_pos)
        input_ids_neg = self.tokenizer.convert_tokens_to_ids(tokens_neg)
    
        # Zero-pad input token IDs
        input_ids_query += [0] * (self.seq_len - len(tokens_query))
        input_ids_pos += [0] * (self.seq_len - len(tokens_pos))
        input_ids_neg += [0] * (self.seq_len - len(tokens_neg))    
    
        # Zero-pad labels
        lm_label_ids_query += [NO_MASK_LABEL] * (self.seq_len - len(tokens_query))

        # Make input mask (1 for real tokens and 0 for padding tokens)
        input_mask_query = [1] * len(tokens_query) + [0] * (self.seq_len - len(tokens_query))
        input_mask_pos = [1] * len(tokens_pos) + [0] * (self.seq_len - len(tokens_pos))
        input_mask_neg = [1] * len(tokens_neg) + [0] * (self.seq_len - len(tokens_neg))                                                          
    
        # Make segment IDs (padded)
        segment_ids_query = [0] * self.seq_len
        segment_ids_cands = [[0] * self.seq_len, [0] * self.seq_len]
    
        # Decide in what order we place the positive and negative examples
        FLIP = random.random() > 0.5
    
        # Package candidate inputs and masks, and make label
        if FLIP:
            pos_cand_id = 1
            input_ids_cands = [input_ids_neg, input_ids_pos]
            input_mask_cands = [input_mask_neg, input_mask_pos]
        else:
            pos_cand_id = 0
            input_ids_cands = [input_ids_pos, input_ids_neg]
            input_mask_cands = [input_mask_pos, input_mask_neg]
    
        # Check data
        assert len(input_ids_query) == self.seq_len
        assert len(input_ids_cands) == 2
        assert len(input_ids_cands[0]) == self.seq_len
        assert len(input_ids_cands[1]) == self.seq_len
        assert len(input_mask_query) == self.seq_len
        assert len(input_mask_cands) == 2
        assert len(input_mask_cands[0]) == self.seq_len
        assert len(input_mask_cands[1]) == self.seq_len
        assert len(segment_ids_query) == self.seq_len
        assert len(segment_ids_cands) == 2
        assert len(segment_ids_cands[0]) == self.seq_len
        assert len(segment_ids_cands[1]) == self.seq_len
        assert len(lm_label_ids_query) == self.seq_len

        # Print a few examples.
        if self.verbose and example.guid < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens_query: {}".format(tokens_query))
            logger.info("input_ids_query: {}".format(input_ids_query))
            logger.info("input_mask_query: {}".format(input_mask_query))
            logger.info("segment_ids_query: {}".format(segment_ids_query))        
            logger.info("tokens_pos: {}".format(tokens_pos))
            logger.info("tokens_neg: {}".format(tokens_neg))        
            logger.info("input_ids_cands: {}".format(input_ids_cands))
            logger.info("input_mask_cands: {}".format(input_mask_cands))        
            logger.info("segment_ids_cands: {}".format(segment_ids_cands))        
            logger.info("pos_cand_id: {}".format(pos_cand_id))
            logger.info("lm_label_ids_query: {}".format(lm_label_ids_query))

        features = InputFeaturesForSPCAndMLM(input_ids_query,
                                             input_mask_query,
                                             segment_ids_query,
                                             input_ids_cands,
                                             input_mask_cands,
                                             segment_ids_cands,
                                             pos_cand_id,
                                             lm_label_ids_query)
        return features


class InputExampleForClassification(object):
    """A single training/test example for classification, and optionally masked language modeling."""

    def __init__(self, guid, tokens, label):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens: list of strings. The tokens.
            label: string. The class label.
        """
        self.guid = guid
        self.tokens = tokens
        self.label = label
        

class InputFeaturesForClassification(object):
    """A single set of features of data for classification, and optionally masked language modeling. """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, masked_input_ids, lm_label_ids):
        self.input_ids = input_ids # List input token IDs
        self.input_mask = input_mask # List containing input mask 
        self.segment_ids = segment_ids # List containing token type (segment) IDs
        self.label_id = label_id # Label id
        self.masked_input_ids = masked_input_ids # List of masked input token IDs (for MLM)
        self.lm_label_ids = lm_label_ids # List containing LM label IDs 
    
    
class BertDatasetForClassification(BertDatasetForTraining):
    
    def __init__(self, train_paths, tokenizer, seq_len, include_mlm=False, sampling_distro="uniform", encoding="utf-8", seed=None, verbose=False):
        # Init parent class
        size = REL_SAMPLE_SIZE + CON_SAMPLE_SIZE + IRR_SAMPLE_SIZE
        super().__init__(train_paths, tokenizer, seq_len, size, unk_only=False, sampling_distro=sampling_distro, encoding=encoding, seed=seed, verbose=verbose)
        self.include_mlm = include_mlm
        
        # Sample a training set
        self.resample() 
        return
    
    
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
                data.append((text, lang))
        assert len(data) == self.sampled_dataset_size

        # Shuffle
        np.random.shuffle(data)
        self.sampled_dataset = data
        return None

    
    def __getitem__(self, item):
        text, lang = self.sampled_dataset[item]
        example_id = self.sample_counter
        self.sample_counter += 1
        tokens = self.tokenizer.tokenize(text)
        example = InputExampleForClassification(guid=example_id, tokens=tokens, label=lang)
        features = self._convert_example_to_features(example)
        tensors = [torch.tensor(features.input_ids),
                   torch.tensor(features.input_mask),
                   torch.tensor(features.segment_ids),
                   torch.tensor(features.label_id),
                   torch.tensor(features.masked_input_ids),
                   torch.tensor(features.lm_label_ids)]
        return tensors

    
    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for classification, and optionally MLM.
        
        :param example: InputExampleForClassification, containing sentence input as lists of tokens.

        :return: InputFeaturesForClassification, containing all inputs and labels of one sample as IDs (as used for model training)

        """
        tokens = example.tokens
        label = example.label
        
        # Truncate sequence if necessary. Account for [CLS] and [SEP] by subtracting 2.
        tokens = tokens[:self.seq_len-2]

        # Mask tokens for MLM
        if self.include_mlm:
            masked_tokens, lm_label_ids = mask_random_tokens(tokens, self.tokenizer)

        # Add CLS and SEP
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        if self.include_mlm:
            masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
            lm_label_ids = [NO_MASK_LABEL] + lm_label_ids + [NO_MASK_LABEL]
        
        # Get input token IDs (unpadded)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.include_mlm:
            masked_input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            
        # Zero-pad input token IDs
        input_ids += [0] * (self.seq_len - len(tokens))
        if self.include_mlm:
            masked_input_ids += [0] * (self.seq_len - len(masked_tokens))
            # Zero-pad labels too
            lm_label_ids += [NO_MASK_LABEL] * (self.seq_len - len(masked_tokens))

        # Make input mask (1 for real tokens and 0 for padding tokens)
        input_mask = [1] * len(tokens) + [0] * (self.seq_len - len(tokens))
    
        # Make segment IDs (padded)
        segment_ids = [0] * self.seq_len

        # Get label ID
        label_id = self.lang2id[label]
        
        # Check data
        assert len(input_ids) == self.seq_len
        assert len(input_mask) == self.seq_len
        assert len(segment_ids) == self.seq_len
        if self.include_mlm:
            assert len(masked_input_ids) == self.seq_len            
            assert len(lm_label_ids) == self.seq_len
        
        if self.verbose and example.guid < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(tokens))
            logger.info("input_ids: {}".format(input_ids))
            logger.info("input_mask: {}".format(input_mask))
            logger.info("segment_ids: {}".format(segment_ids))
            logger.info("label: {}".format(label))
            logger.info("label_id: {}".format(label_id))
            if self.include_mlm:
                logger.info("masked_tokens: {}".format(masked_tokens))
                logger.info("masked_input_ids: {}".format(masked_input_ids))
                logger.info("lm_label_ids: {}".format(lm_label_ids))

        # Get features
        if not self.include_mlm:
            masked_input_ids = []
            lm_label_ids = []
        features = InputFeaturesForClassification(input_ids=input_ids,
                                                  input_mask=input_mask,
                                                  segment_ids=segment_ids,
                                                  label_id=label_id,
                                                  masked_input_ids=masked_input_ids,
                                                  lm_label_ids=lm_label_ids)
        return features


class BertDatasetForTesting(Dataset):
    """ A class for evaluating classification on dev or test sets. """
    
    def __init__(self, path_data, tokenizer, label2id, seq_len, require_labels=False, encoding="utf-8", verbose=False):
        """ Constructor.

        Args:
        - path_data: path of a file in TSV format, with one or 2 columns, containing texts and optional labels.
        - tokenizer:
        - label2id: dict that maps labels2ids
        - seq_len: maximum sequence length (including CLS and SEP)

        """
        self.path_data = path_data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.seq_len = seq_len
        self.require_labels = require_labels
        self.encoding = encoding
        self.verbose = verbose
        self.data = []
        self.label_list = [None] * len(label2id)
        for label, label_id in label2id.items():
            self.label_list[label_id] = label
        assert None not in self.label_list
        
        # Load data
        with open(self.path_data, encoding=self.encoding) as f:
            for line in f:
                elems = line.strip().split("\t")
                if len(elems) == 0:
                    # Empty line
                    continue
                elif len(elems) == 1:
                    if self.requre_labels:
                        msg = "only once column found, but require_labels is True"
                        raise RuntimeError(msg)
                    text = elems[0]
                    label = None
                elif len(elems) == 2:
                    text = elems[0]
                    label = elems[1]
                else:
                    msg = "invalid number of columns (%d)" % len(elems)
                    raise RuntimeError(msg)
                self.data.append((text, label))

                
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, item):
        text, label = self.data[item]
        example_id = item
        tokens = self.tokenizer.tokenize(text)
        example = InputExampleForClassification(guid=example_id, tokens=tokens, label=label)
        features = self._convert_example_to_features(example)
        tensors = [torch.tensor(features.input_ids),
                   torch.tensor(features.input_mask),
                   torch.tensor(features.segment_ids)]
        if features.label_id is None:
            tensors.append(torch.empty(0))
        else:
            tensors.append(torch.tensor(features.label_id))
        return tensors


    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for classification.
        
        :param example: InputExampleForClassification.

        :return: InputFeaturesForClassification.

        """
        tokens = example.tokens
        label = example.label
        
        # Truncate sequence if necessary. Account for [CLS] and [SEP] by subtracting 2.
        tokens = tokens[:self.seq_len-2]

        # Add CLS and SEP
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        
        # Get input token IDs (unpadded)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
        # Zero-pad input token IDs
        input_ids += [0] * (self.seq_len - len(tokens))

        # Make input mask (1 for real tokens and 0 for padding tokens)
        input_mask = [1] * len(tokens) + [0] * (self.seq_len - len(tokens))
    
        # Make segment IDs (padded)
        segment_ids = [0] * self.seq_len

        # Get label ID
        if label is None:
            label_id is None
        else:
            label_id = self.label2id[label]
        
        # Check data
        assert len(input_ids) == self.seq_len
        assert len(input_mask) == self.seq_len
        assert len(segment_ids) == self.seq_len
        
        if self.verbose and example.guid < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(tokens))
            logger.info("input_ids: {}".format(input_ids))
            logger.info("input_mask: {}".format(input_mask))
            logger.info("segment_ids: {}".format(segment_ids))
            logger.info("label: {}".format(label))
            logger.info("label_id: {}".format(label_id))

        # Get features
        features = InputFeaturesForClassification(input_ids=input_ids,
                                                  input_mask=input_mask,
                                                  segment_ids=segment_ids,
                                                  label_id=label_id,
                                                  masked_input_ids=[],
                                                  lm_label_ids=[])
        return features
