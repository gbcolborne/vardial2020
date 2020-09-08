import torch
import torch.nn as nn

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
