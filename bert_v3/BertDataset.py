""" Dataset classes for training BERT """

import sys, os, random, logging
from io import open
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
sys.path.append("..")
from comp_utils import RELEVANT_LANGS, IRRELEVANT_LANGS, IRRELEVANT_URALIC_LANGS, ALL_LANGS


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


def line_to_data(line, is_labeled):
    """Takes a line from a dataset (labeled or unlabeled), and returns the
    text and label.

    """
    if is_labeled:
        elems = line.strip().split("\t")
        assert len(elems) == 2
        text = elems[0]
        label = elems[1]
    else:
        text = line.strip()
        # Make sure text is not labeled
        if len(text) > 3:
            assert text[-4] != "\t"
        label = None
    return (text, label)


def mask_random_tokens(tokens, tokenizer):
    """Masking some random tokens for masked language modeling with
    probabilities as in the original BERT paper.

    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction

    """
    output_label = []
    # Copy tokens so that we don't modify the input list.
    tokens = deepcopy(tokens)
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


class BertDatasetForPretraining(IterableDataset):

    """Abstract class for BertDataset used for pre-training. Implements
    sampling of data from disk, as we can not load all the training
    data into memory. Sets sampling probabilities. Subclasses must
    implement `__iter__`. 

    """

    def __init__(self, train_paths, tokenizer, seq_len, sampling_alpha=1.0, weight_relevant=1.0, encoding="utf-8", seed=None, verbose=False):
        super(BertDatasetForPretraining).__init__()
        assert sampling_alpha >= 0.0 and sampling_alpha <= 1.0
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.seq_len = seq_len # Includes CLS token
        self.sampling_alpha = sampling_alpha
        self.weight_relevant = weight_relevant   # Sampling weight of relevant examples wrt irrelevant examples
        self.encoding = encoding
        self.verbose = verbose
        self.sample_counter = 0  # total number of examples sampled by calling the iterator returned by __iter__ 
        self.size = 0 # Total size of dataset
        self.lang_list = []
        self.lang2id = {} # Maps to indices in lang_list
        self.lang2path = {}
        self.lang2file = {}
        self.lang2freq = {}
        self.lang2ix = {} # Maps to the current index in the training file
        self.sample_probs = [] # Language sampling probabilities

        # Seed RNG
        if seed:
            random.seed(seed)            
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Store paths of training files
        for path in sorted(train_paths):
            filename = os.path.split(path)[-1]
            assert filename[-6:] == ".train"
            cut = filename.rfind(".")
            lang = filename[:cut]
            assert lang in ALL_LANGS or lang == "unk"
            self.lang2path[lang] = path                            
        assert len(self.lang2path) > 0
        self.lang_list = sorted(self.lang2path.keys())
        self.lang2id = {x:i for i,x in enumerate(self.lang_list)}

        # For the moment, I do not foresee a use case where we include
        # both labeled and unlabeled data, so require that we only
        # have one or the other.
        if 'unk' in self.lang2id:
            if len(self.lang2id) != 1:
                msg = "Expected either 1 path to unlabeled training data (unk.train)"
                msg += " or n paths to labeled training data (*.train), but not both."
                raise RuntimeError(msg)
        
        # Prepare training files to sample lazily from disk
        for lang in self.lang_list:
            path = self.lang2path[lang]
            logger.info("Processing %s" % path)            
            # Open file to load lazily from disk later when we start iterating
            self.lang2file[lang] = open(path, 'r', encoding=self.encoding)
            self.lang2ix[lang] = 0
            # Count examples
            self.lang2freq[lang] = 0
            with open(path, 'r', encoding=self.encoding) as f:
                for line in f:
                    (text, label) = line_to_data(line, False)
                    assert text is not None
                    self.lang2freq[lang] += 1
        self.size = sum(self.lang2freq.values())
        logger.info("Dataset size: %d" % self.size)

        # Skip a random number of lines.
        logger.info("Skipping random number of lines...")
        for lang in self.lang_list:
            start = random.randint(0, self.lang2freq[lang])
            self.lang2ix[lang] = start
            for _ in range(start):
                next(self.lang2file[lang])
                
        # Compute sampling probabilities
        self.sample_probs = self._compute_sampling_probs()

        # Make buffer of sampled languages (if we have more than 1 language)
        nb_langs = len(self.lang_list)
        if nb_langs > 1:
            self.sampled_lang_buffer_size = 10**6
            self.sampled_lang_buffer = self._make_sampled_lang_buffer()
            self.sampled_lang_buffer_ix = 0
        return

    
    def __len__(self):
        return self.size


    def read_line(self, lang):
        # Check if we have reached EOF for sampled language
        if self.lang2ix[lang] >= (self.lang2freq[lang]-1):
            self.lang2file[lang].close()
            self.lang2file[lang] = open(self.lang2path[lang], "r", encoding=self.encoding)
            self.lang2ix[lang] = 0
        # Read next line for sampled language
        line = next(self.lang2file[lang])
        self.lang2ix[lang] += 1
        return line


    def sample_language(self):
        if len(self.lang_list) == 1:
            return self.lang_list[0]
        sampled_lang_id = self.sampled_lang_buffer[self.sampled_lang_buffer_ix]
        sampled_lang = self.lang_list[sampled_lang_id]
        self.sampled_lang_buffer_ix += 1
        if self.sampled_lang_buffer_ix >= self.sampled_lang_buffer_size:
            # Refresh buffer
            self.sampled_lang_buffer = self._make_sampled_lang_buffer()            
            self.sampled_lang_buffer_ix = 0
        return sampled_lang
    
    
    def _make_sampled_lang_buffer(self):
        nb_langs = len(self.lang_list)
        b = np.random.choice(np.arange(nb_langs),
                             size=self.sampled_lang_buffer_size,
                             replace=True,
                             p=self.sample_probs)
        return b
                 
    
    def _compute_sampling_probs(self):
        if len(self.lang_list) == 1:
            return [1]
        # We compute the sampling probabilities of the relevant and
        # irrelevant languages independently.
        rel_langs = sorted(RELEVANT_LANGS)
        irr_langs = sorted(IRRELEVANT_LANGS)
        logger.info("Computing sampling probabilities for relevant languages...")
        rel_probs = self._compute_sampling_probs_for_subgroup(rel_langs)
        logger.info("Computing sampling probabilities for irrelevant languages...")
        irr_probs = self._compute_sampling_probs_for_subgroup(irr_langs)
        # Weight the distribution of relevant languages, then renormalize
        rel_probs = rel_probs * self.weight_relevant
        sum_of_both = rel_probs.sum() + irr_probs.sum()
        rel_probs = rel_probs / sum_of_both
        irr_probs = irr_probs / sum_of_both
        sample_probs = [0 for lang in self.lang_list]            
        for lang, prob in zip(rel_langs, rel_probs):
            lang_id = self.lang2id[lang]
            sample_probs[lang_id] = prob
        for lang, prob in zip(irr_langs, irr_probs):
            lang_id = self.lang2id[lang]
            sample_probs[lang_id] = prob
        logger.info("Stats on sampling probabilities:")
        logger.info("- Min prob (relevant): %f" % (min(rel_probs)))
        logger.info("- Mean prob (relevant): %f" % (np.mean(rel_probs)))        
        logger.info("- Max prob (relevant): %f" % (max(rel_probs)))
        logger.info("- Cumulative prob (relevant): %f" % (sum(rel_probs)))        
        logger.info("- Min prob (irrelevant): %f" % (min(irr_probs)))
        logger.info("- Mean prob (irrelevant): %f" % (np.mean(irr_probs)))        
        logger.info("- Max prob (irrelevant): %f" % (max(irr_probs)))
        logger.info("- Cumulative prob (irrelevant): %f" % (sum(irr_probs)))        
        return sample_probs


    def _compute_sampling_probs_for_subgroup(self, lang_list):
        counts = np.array([self.lang2freq[k] for k in lang_list], dtype=np.float)
        probs = counts / counts.sum()
        probs_damp = probs ** self.sampling_alpha
        probs = probs_damp / probs_damp.sum()
        return probs

    
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
    
    
class BertDatasetForMLM(BertDatasetForPretraining):
    
    def __init__(self, train_paths, tokenizer, seq_len, sampling_alpha=1.0, weight_relevant=1.0, encoding="utf-8", seed=None, verbose=False):
        # Init parent class
        super().__init__(train_paths, tokenizer, seq_len, sampling_alpha=sampling_alpha, weight_relevant=weight_relevant, encoding=encoding, seed=seed, verbose=verbose)
        return


    def __iter__(self):
        sampler = self._generate_samples()
        return sampler

    
    def _generate_samples(self):
        while True:
            # Sample a language
            lang = self.sample_language()
            line = self.read_line(lang)
            (text,_) = line_to_data(line, False)
            assert text is not None and len(text)

            # Create input tensors
            example_id = self.sample_counter
            self.sample_counter += 1
            tokens = self.tokenizer.tokenize(text)
            example = InputExampleForMLM(guid=example_id, tokens=tokens)
            features = self._convert_example_to_features(example)
            tensors = (torch.tensor(features.input_ids),
                       torch.tensor(features.input_mask),
                       torch.tensor(features.segment_ids),
                       torch.tensor(features.lm_label_ids))
            yield tensors

            
    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for MLM only, with IDs, LM labels,
        input_mask, CLS token etc.
        
        :param example: InputExampleForMLM, containing sentence input as lists of tokens.
        :return: InputFeaturesForMLM, containing all inputs and labels of one sample as IDs (as used for model training)

        """
        tokens = example.tokens
        
        # Truncate sequence if necessary. Account for [CLS] by subtracting 1.
        tokens = tokens[:self.seq_len-1]
        
        # Mask tokens for MLM
        tokens, lm_label_ids = mask_random_tokens(tokens, self.tokenizer)

        # Add CLS
        tokens = ["[CLS]"] + tokens 
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
            logger.info("*** Example (BertDatasetForMLM) ***")
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


class BertDatasetForSPCAndMLM(BertDatasetForPretraining):
    
    def __init__(self, train_paths, tokenizer, seq_len, sampling_alpha=1.0, weight_relevant=1.0, encoding="utf-8", seed=None, verbose=False):
        # Init parent class
        super().__init__(train_paths, tokenizer, seq_len, sampling_alpha=sampling_alpha, weight_relevant=weight_relevant, encoding=encoding, seed=seed, verbose=verbose)

        # Training data should not include UNK, as we cannot do SPC on unlabeled data
        assert 'unk' not in self.lang_list

        # We need more than one language to do SPC
        assert len(self.lang_list) > 1
        

    def __iter__(self):
        sampler = self._generate_samples()
        return sampler

    
    def _generate_samples(self):
        while True:
            # Sample a language for which we have at least 2 examples (query and positive example)
            pos_lang = None
            while pos_lang is None:
                sampled_lang = self.sample_language()
                if self.lang2freq[sampled_lang] > 1:
                    pos_lang = sampled_lang

            # Read 2 lines: one for the query, the other for the positive example
            line_query = self.read_line(pos_lang)
            line_pos = self.read_line(pos_lang)
            (text_query,_) = line_to_data(line_query, False)
            (text_pos,_) = line_to_data(line_pos, False)
            assert text_query is not None and len(text_query)
            assert text_pos is not None and len(text_pos)

            # Sample a different language for the negative example
            neg_lang = None
            while neg_lang is None:
                sampled_lang = self.sample_language()
                if sampled_lang != pos_lang:
                    neg_lang = sampled_lang

            # Read a line
            line_neg = self.read_line(neg_lang)
            (text_neg,_) = line_to_data(line_neg, False)
            assert text_neg is not None and len(text_neg)
            
                    
            # Create input tensors
            example_id = self.sample_counter
            self.sample_counter += 1
            tokens_query = self.tokenizer.tokenize(text_query)
            tokens_pos = self.tokenizer.tokenize(text_pos)
            tokens_neg = self.tokenizer.tokenize(text_neg)        
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
            yield tensors
        

    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for both MLM and SPC, with IDs, LM labels,
        input_mask, candidate labels for sentence pair classification, CLS
        token, etc.

        :param example: InputExampleForSPCAndMLM, containing sentence inputs as lists of tokens.

        :return: InputFeaturesForSPCAndMLM, containing all inputs and labels of one sample as IDs (as used for model training)

        """
        tokens_query = example.tokens_query
        tokens_pos = example.tokens_pos
        tokens_neg = example.tokens_neg
    
        # Truncate sequence if necessary. Account for [CLS] by subtracting 1.
        tokens_query = tokens_query[:self.seq_len-1]
        tokens_pos = tokens_pos[:self.seq_len-1]
        tokens_neg = tokens_neg[:self.seq_len-1]    
    
        # Mask tokens for MLM
        tokens_query, lm_label_ids_query = mask_random_tokens(tokens_query, self.tokenizer)

        # Add CLS
        tokens_query = ["[CLS]"] + tokens_query 
        tokens_pos = ["[CLS]"] + tokens_pos 
        tokens_neg = ["[CLS]"] + tokens_neg 
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
    """A single training/test example for classification."""

    def __init__(self, guid, tokens, label, lang):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens: list of strings. The tokens.
            label: int. The class label (1 or 0)
            lang_id: string. The language (should be target language if label is 1). 
        """
        self.guid = guid
        self.tokens = tokens
        self.label = label
        self.lang = lang
        

class InputFeaturesForClassification(object):
    """A single set of features of data for classification. """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, lang_id, seq_len):
        self.input_ids = input_ids # List input token IDs
        self.input_mask = input_mask # List containing input mask 
        self.segment_ids = segment_ids # List containing token type (segment) IDs
        self.label_id = label_id # Label id (1 or 0)
        self.lang_id = lang_id # Lang id (should be target lang if label id is 1)
        self.seq_len # Real sequence length of input tokens (excluding padding and special token CLS).
        
    
class BertDatasetForClassification(Dataset):
    """Dataset for fine-tuning a language identifier for one specific
    language, with negative examples sampled from other languages.

    Note: examples are sorted by length, so that we can train more
    efficienlty (by setting seq_len to the length of the longest
    examples in a given batch).

    """
    
    def __init__(self, dir_data, target_lang, tokenizer, max_seq_len=256, sampling_alpha=1.0, weight_relevant=1.0, encoding="utf-8", seed=None, verbose=False):
        # Init parent class
        super().__init__()

        assert sampling_alpha >= 0 and sampling_alpha <= 1
        self.target_lang = target_lang
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len # Includes CLS token
        self.verbose = verbose
        self.sorted_examples = []
        self.sample_counter = 0

        # Seed RNG
        if seed:
            random.seed(seed)            
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Map languages to integer IDs
        assert self.target_lang in ALL_LANGS
        all_langs = sorted(ALL_LANGS)
        self.lang2id = {x:i for i,x in enumerate(all_langs)}
            
        # Load all positive examples
        data = []
        path = os.path.join(self.dir_data, "%s.train" % self.target_lang)
        logger.info("Loading positive examples")
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                (text, label) = line_to_data(line, False)
                assert text is not None
                data.append((text, 1, self.target_lang))
        nb_pos = len(data)
        lengths = [len(x) for (x,y,z) in data]
        min_pos_length = min(lengths)
        max_pos_length = max(lengths)
        logger.info("Nb positive examples: %d" % nb_pos)
        logger.info("Min length of positive examples: %d" % min_pos_length)
        logger.info("Max length of positive examples: %d" % max_pos_length)

        # Compute corpus sizes, ignoring examples that are shorter
        # than the shortest positive example or longer than the
        # longest
        lang2freq = {}
        for lang in all_langs:
            path = os.path.join(self.dir_data, "%s.train" % lang)
            # Count number of examples
            logger.info("Counting examples in %s" % path)
            lang2freq[lang] = 0
            with open(path, 'r', encoding=encoding) as f:
                for line in f:
                    (text, label) = line_to_data(line, False)
                    assert text is not None
                    if len(text) >= min_pos_length and len(text) <= max_pos_length:
                        lang2freq[lang] += 1
            if lang2freq[lang] == 0:
                msg = "No examples found for lang '%s'" % lang
                msg += " (ignoring texts shorter than %d or longer than %d chars)" % (min_pos_length, max_pos_length)
                logger.warning(msg)
        
        # Compute the sampling probabilities of the relevant and
        # irrelevant languages independently.
        rel_langs = sorted(RELEVANT_LANGS)
        irr_langs = sorted(IRRELEVANT_LANGS)
        if lang in rel_langs:
            rel_langs.remove(lang)
        else:
            irr_langs.remove(lang)
        logger.info("Computing sampling probabilities for relevant languages...")
        rel_probs = self._compute_sampling_probs_for_subgroup(rel_langs, lang2freq, sampling_alpha)
        logger.info("Computing sampling probabilities for irrelevant languages...")
        irr_probs = self._compute_sampling_probs_for_subgroup(irr_langs, lang2freq, sampling_alpha)
        # Weight the distribution of relevant languages, then renormalize
        rel_probs = rel_probs * weight_relevant
        sum_of_both = rel_probs.sum() + irr_probs.sum()
        rel_probs = rel_probs / sum_of_both
        irr_probs = irr_probs / sum_of_both
        sample_probs = [0 for lang in all_langs]
        for lang, prob in zip(rel_langs, rel_probs):
            lang_id = self.lang2id[lang]
            sample_probs[lang_id] = prob
        for lang, prob in zip(irr_langs, irr_probs):
            lang_id = self.lang2id[lang]
            sample_probs[lang_id] = prob
        logger.info("Stats on sampling probabilities:")
        logger.info("- Min prob (relevant): %f" % (min(rel_probs)))
        logger.info("- Mean prob (relevant): %f" % (np.mean(rel_probs)))        
        logger.info("- Max prob (relevant): %f" % (max(rel_probs)))
        logger.info("- Cumulative prob (relevant): %f" % (sum(rel_probs)))        
        logger.info("- Min prob (irrelevant): %f" % (min(irr_probs)))
        logger.info("- Mean prob (irrelevant): %f" % (np.mean(irr_probs)))        
        logger.info("- Max prob (irrelevant): %f" % (max(irr_probs)))
        logger.info("- Cumulative prob (irrelevant): %f" % (sum(irr_probs)))        

        # Count number of negative examples we sample from each of the
        # other languages. Avoid sampling the same example twice.
        sample_sizes = [0 for _ in all_langs]
        nb_neg = 0
        buffer_size = 10**6
        samples = np.random.choice(np.arange(len(all_langs)),
                                   size=buffer_size,
                                   replace=True,
                                   p=sample_probs)
        sample_ix = 0
        while nb_neg < nb_pos:
            lang_id = samples[sample_ix]
            lang = all_langs[lang_id]
            if lang == self.target_lang:
                msg = "Sampled target lang even though it should have 0 probability."
                raise RuntimeError(msg)
            if sample_sizes[lang_id] < lang2freq[lang]:
                sample_sizes[lang_id] += 1
                nb_neg += 1
            sample_ix += 1
            if sample_ix >= buffer_size:
                # Refresh buffer
                samples = np.random.choice(np.arange(len(all_langs)),
                                           size=buffer_size,
                                           replace=True,
                                           p=sample_probs)
                sample_ix = 0
                            
        # Sample negative examples one language at a time
        logger.info("Sampling negative examples")
        for lang, sample_size in zip(all_langs, sample_sizes):
            if lang == self.target_lang:
                assert sample_size == 0
                continue
            path = os.path.join(self.dir_data, "%s.train" % lang)
            if sample_size == 0:
                logger.info("  Not sampling from %s (no examples of appropriate length)" % path)
                continue
            logger.info("  Sampling %d examples from %s" % (sample_size, path))
            texts = []
            # Load all texts of appropriate length
            with open(path, 'r', encoding=self.encoding) as f:
                for line in f:
                    (text, label) = line_to_data(line, False)
                    assert text is not None
                    if len(text) >= min_pos_length and len(text) <= max_pos_length:
                        texts.append(text)
            # Sample
            np.random.shuffle(texts)
            for text in texts[:sample_size]:
                data.append((text, 0, lang))

        # Sort examples by length
        logger.info("Sorting examples")
        sorted_examples = sorted(data, key=lambda x:len(x[0]), reverse=False)

        # Shuffle examples of same length
        beg_ix = 0
        max_cont = self._get_max_contiguous_same_label(sorted_examples)
        logger.info("Max contiguous examples with same label: %d" % (max_cont))
        logger.info("Shuffling examples of same length")
        while end_ix < len(sorted_examples):
            beg_label = sorted_examples[beg_ix][1]
            end_ix = beg_ix
            while sorted_examples[end_ix+1][1] == beg_label:
                end_ix += 1
                if end_ix == len(sorted_examples) - 1:
                    break
            count = end_ix - beg_ix + 1
            if count > 1:
                # Shuffle this subsequence
                sub = sorted_examples[beg_ix:end_ix+1]
                np.random.shuffle(sub)
                sorted_examples[beg_ix:end_ix+1] = sub
            beg_ix = end_ix + 1
        max_cont = self._get_max_contiguous_same_label(sorted_examples)
        logger.info("Max contiguous examples with same label: %d" % (max_cont))
        
        # Done
        print("Dataset ready. Size: %d" % len(self))

            
    def _get_max_contiguous_same_label(examples):
        # Check max number of contiguous examples with same label
        cur_count = 0
        max_count = 0
        cur_label = None
        for (text, label, lang_id) in examples:
            if label == cur_label:
                cur_count += 1
            else:
                if cur_count > max_count:
                    max_count = cur_count
                cur_count = 0
                cur_label = label
        return max_count


    def _compute_sampling_probs_for_subgroup(self, lang_list, lang2freq, alpha):
        counts = np.array([lang2freq[k] for k in lang_list], dtype=np.float)
        probs = counts / counts.sum()
        probs_damp = probs ** alpha
        probs = probs_damp / probs_damp.sum()
        return probs

    
    def __len__(self):
        return len(sorted_examples)


    def __getitem__(self, item):
        text, label_id, lang = self.sorted_examples[item]
        example_id = self.sample_counter
        self.sample_counter += 1
        tokens = self.tokenizer.tokenize(text)
        example = InputExampleForClassification(guid=example_id, tokens=tokens, label=label_id, lang=lang)
        features = self._convert_example_to_features(example)
        tensors = [torch.tensor(features.input_ids),
                   torch.tensor(features.input_mask),
                   torch.tensor(features.segment_ids),
                   torch.tensor(features.label_id),
                   torch.tensor(features.lang_id),                   
                   torch.tensor(features.seq_len)]
        return tensors

    
    def _convert_example_to_features(self, example):
        """Convert a raw sample (a sentence as tokenized strings) into a
        proper training sample for classification.
        
        :param example: InputExampleForClassification, containing sentence input as lists of tokens.

        :return: InputFeaturesForClassification, containing all inputs and labels of one sample as IDs (as used for model training)

        """
        tokens = example.tokens
        label = example.label
        lang = example.lang
        lang_id = self.lang2id[lang]

        # Store original sequence length
        seq_len = len(tokens)
        
        # Truncate sequence if necessary. Account for [CLS] by subtracting 1.
        tokens = tokens[:self.max_seq_len-1]

        # Add CLS
        tokens = ["[CLS]"] + tokens
        
        # Get input token IDs (unpadded)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
        # Zero-pad input token IDs
        input_ids += [0] * (self.max_seq_len - len(tokens))

        # Make input mask (1 for real tokens and 0 for padding tokens)
        input_mask = [1] * len(tokens) + [0] * (self.max_seq_len - len(tokens))
    
        # Make segment IDs (padded)
        segment_ids = [0] * self.max_seq_len

        
        # Check data
        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        if self.verbose and example.guid < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(tokens))
            logger.info("input_ids: {}".format(input_ids))
            logger.info("input_mask: {}".format(input_mask))
            logger.info("segment_ids: {}".format(segment_ids))
            logger.info("label: {}".format(label))
            logger.info("lang: {}".format(lang))
            logger.info("lang_id: {}".format(lang_id))            
            logger.info("seq_len: {}".format(seq_len))
            
        # Get features
        features = InputFeaturesForClassification(input_ids=input_ids,
                                                  input_mask=input_mask,
                                                  segment_ids=segment_ids,
                                                  label_id=label,
                                                  lang_id=lang_id,
                                                  seq_len=seq_len)
        return features



class BertDatasetForTesting(Dataset):
    """ A class for evaluating classification on dev or test sets. """
    
    def __init__(self, path_data, tokenizer, label2id, seq_len, require_labels=False, encoding="utf-8", verbose=False):
        """ Constructor.

        Args:
        - path_data: path of a file in TSV format, with one or 2 columns, containing texts and optional labels.
        - tokenizer:
        - label2id: dict that maps labels2ids
        - seq_len: maximum sequence length (including CLS)

        """
        super(BertDatasetForTesting).__init__()
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
                    if self.require_labels:
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
        
        # Truncate sequence if necessary. Account for [CLS] by subtracting 1.
        tokens = tokens[:self.seq_len-1]

        # Add CLS
        tokens = ["[CLS]"] + tokens
        
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
            label_id = None
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
