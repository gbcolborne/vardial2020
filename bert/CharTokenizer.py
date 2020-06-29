class CharTokenizer():
    def __init__(self, training_data=None):
        """Make tokenizer. If training data is provided, then we compute the
        character frequencies to enable trimming the vocabulary.

        Args:
        - training_data: list of texts.

        """
        self.vocab = self._init_vocab()
        for i,char in enumerate(CUNEIFORM_CHARS):
            self.vocab[char] = len(self.vocab)
        self.char2count = None
        if training_data:
            self.char2count = {k:0 for k in self.vocab}
            for text in training_data:
                for char in text:
                    if char not in self.vocab:
                        msg = "Warning: encounter OOV char '{}' while counting chars.".format(char)
                        print(msg)
                    else:
                        self.char2count[char] += 1

    def _init_vocab(self):
        vocab = {}
        vocab["[PAD]"] = len(vocab)
        vocab["[UNK]"] = len(vocab)
        vocab["[MASK]"] = len(vocab)
        vocab["[CLS]"] = len(vocab)
        vocab["[SEP]"] = len(vocab)
        vocab[" "] = len(vocab)
        return vocab

    def trim_vocab(self, min_freq):
        if min_freq < 1:
            return
        if not self.char2count:
            msg = "Provide training data when initializing tokenizer if you want to then trim the vocab."
            raise NotImplementedError(msg)
        new_vocab = self._init_vocab()
        for char in CUNEIFORM_CHARS:
            if self.char2count[char] >= min_freq:
                new_vocab[char] = len(new_vocab)
        self.vocab = new_vocab

    def tokenize(self, chars):
        return [c if c in self.vocab else "[UNK]" for c in chars]
    
    def convert_tokens_to_ids(self, chars):
        return [self.vocab[c] if c in self.vocab else self.vocab["[UNK]"] for c in chars]
    


def load_labeled_data(path):
    """ Load labeled data """
    texts = []
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line):
                 text, label = line.split("\t")
                 texts.append(text)
                 labels.append(label)
    return texts, labels
