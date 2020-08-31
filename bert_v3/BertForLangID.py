import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_size, nb_classes):
        super().__init__()
        self.dense = nn.Linear(hidden_size, nb_classes)
        
    def forward(self, encodings, cls=None):
        """Forward pass.

        Args:
        - Encodings: tensor of shape (batch size, hidden dim) or shape
          (nb classes, batch size, hidden dim) if we have
          class-specific encodings; in this case, we do a batch
          multiply of the class-specific encodings and each of the
          classe specific output layers (each with a single output
          unit).
        - cls: (optional) integer ID of output unit we are training
          (to get all scores, leave this set to None)
        
        Returns: logits, tensor of shape (batch size, nb classes)

        """
        if cls:
            if len(encodings.shape) == 3:
                output = torch.bmm(encodings, self.dense.weight[:,cls].permute(1,0).unsqueeze(2)).squeeze().permute(1,0)
                output = output + self.dense.bias[cls]
            else:
                output = torch.matmul(encodings, self.dense.weight[:,cls].permute(1,0))
                output = output + self.dense.bias[cls]
        else:
            if len(encodings.shape) == 3:
                output = torch.bmm(encodings, self.dense.weight.unsqueeze(2)).squeeze().permute(1,0)
                output = output + self.dense.bias
            elif len(encodings.shape) < 3:
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


class Adapter(nn.Module):
    """Adapter layer designed to transform text encoding before passing it
    to the output layer.

    """
    def __init__(self, hidden_dim, nb_classes):
        super().__init__()
        self.dense = nn.Linear(nb_classes, hidden_size, hidden_size)
        self.activation = nn.Tanh()

        
    def forward(self, hidden_state, cls=None):
        """ Forward pass

        Params:
        - hidden_state
        - cls: (optional) integer ID of output unit we are training (to get all scores, leave this set to None)        

        Returns: tensor of shape (batch size, hidden size) if cls is provided, (nb classes, batch size, hidden size) otherwise.

        """
        if cls:
            output = torch.matmul(hidden_state, self.dense.weight[:,cls].permute(1,0)) + self.dense.bias[cls]
        else:
            output = torch.matmul(hidden_state, self.dense.weight.permuet(1,0)) + self.dense.bias
        output = self.activation(output)
        return output

    
class BertForLangID(nn.Module):
    """ Bert-based language identifier. """
    
    def __init__(self, encoder, lang_list, add_adapters=False):
        """ Constructor. 

        Params:
        - encoder: a Bert model used for encoding
        - lang_list: list of languages handled (in order)

        """
        super().__init__()
        self.encoder = encoder
        self.pooler = Pooler(encoder.config.hidden_size, cls_only=False)
        self.adapter = None
        if add_adapters:
            self.adapter = Adapter()
        self.classifier = Classifier(encoder.config.hidden_size, len(lang_list))
        self.lang2id = {x:i for i,x in enumerate(lang_list)}


    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
            

    def forward(self, input_ids, input_mask, segment_ids, cls=None):
        """ Forward pass.

        Params:
        - input_ids: input token IDs
        - input_mask: attention mask (1 for real tokens, 0 for padding)
        - segment_ids: token type (i.e. sequence) IDs
        - cls: integer ID of output unit we are training (to get all scores, leave this set to None)

        """
        outputs = encoder.bert(input_ids=input_ids,
                               attention_mask=input_mask,
                               token_type_ids=segment_ids,
                               position_ids=None)
        last_hidden_states = outputs[0]
        encodings = pooler(last_hidden_states)
        if self.adapter:
            encodings = self.adapter(encodings, cls=cls)
        scores = classifier(encodings, cls=cls)
        return scores


    
