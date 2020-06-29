""" Simple tokenizer that splits texts into characters. """

class CharTokenizer():
    def __init__(self):
        """Make tokenizer. 

        Args:

        """
        self.vocab = self._init_vocab()
        self.char2count = {}
        return

    
    def _init_vocab(self):
        vocab = {}
        vocab["[PAD]"] = len(vocab)
        vocab["[UNK]"] = len(vocab)
        vocab["[MASK]"] = len(vocab)
        vocab["[CLS]"] = len(vocab)
        vocab["[SEP]"] = len(vocab)
        vocab[" "] = len(vocab)
        return vocab

    
    def update_vocab(self, text):
        """ Update vocab and character frequencies. """
        for char in text:
            if char not in self.vocab:
                vocab[char] = len(vocab)
                char2count[char] = 0
            char2count[char] += 1
        return

    
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
        return

    
    def tokenize(self, chars):
        return [c if c in self.vocab else "[UNK]" for c in chars]

    
    def convert_tokens_to_ids(self, chars):
        return [self.vocab[c] if c in self.vocab else self.vocab["[UNK]"] for c in chars]
    


