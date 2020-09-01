import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_size, nb_classes):
        super().__init__()
        self.dense = nn.Linear(hidden_size, nb_classes)
        
    def forward(self, encodings):
        """ Forward pass.

        Args:
        - Encodings: tensor of shape (batch size, hidden_dim)
        
        Returns: logits, tensor of shape (batch size, nb classes)

        """
        output = self.dense(encodings)    
        return output
