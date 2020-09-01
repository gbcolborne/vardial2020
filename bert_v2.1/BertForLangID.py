import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_size, nb_classes):
        super().__init__()
        self.dense = nn.Linear(hidden_size, nb_classes)
        
    def forward(self, encodings, cls=None):
        """Forward pass.

        Args:
        - Encodings: tensor of shape (batch size, hidden dim)
        
        Returns: logits, tensor of shape (batch size, nb classes)

        """
        output = self.dense(encodings)
        return output


class Pooler(nn.Module):
    def __init__(self, hidden_size, cls_only=True):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.cls_only = cls_only
        
    def forward(self, hidden_states):
        if self.cls_only:
            # In BertPooler, they "pool" by taking the hidden state
            # corresponding to the CLS token.
            pooled = hidden_states[:, 0]
        else:
            # We average pool the hidden states
            pooled = torch.mean(hidden_states, dim=1)
        output = self.activation(self.dense(pooled))    
        return output    


class BertForLangID(nn.Module):
    """ Bert-based language identifier. """
    
    def __init__(self, encoder, lang_list):
        """ Constructor. 

        Params:
        - encoder: a Bert model used for encoding
        - lang_list: list of languages handled (in order)

        """
        super().__init__()
        self.encoder = encoder
        self.pooler = Pooler(encoder.config.hidden_size, cls_only=False)
        self.classifier = Classifier(encoder.config.hidden_size, len(lang_list))
        self.lang2id = {x:i for i,x in enumerate(lang_list)}


    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
            

    def forward(self, input_ids, input_mask, segment_ids):
        """ Forward pass.

        Params:
        - input_ids: input token IDs
        - input_mask: attention mask (1 for real tokens, 0 for padding)
        - segment_ids: token type (i.e. sequence) IDs

        """
        outputs = self.encoder.bert(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids,
                                    position_ids=None)
        last_hidden_states = outputs[0]
        encodings = self.pooler(last_hidden_states)
        scores = self.classifier(encodings)
        return scores


    
