import sys
sys.path.append("/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.init import xavier_normal_

def get_param(shape):
	param = nn.Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class InputPrompts(nn.Module):
    def __init__(self, prompt_len, input_dim):
        super().__init__()
        
        self.prompt_len = prompt_len
        self.input_dim = input_dim

        self.prefix_tokens = torch.arange(self.prompt_len).long()
        self.prefix_embedding = nn.Sequential(
            nn.Embedding(self.prompt_len, self.input_dim),
            nn.Linear(self.input_dim, self.input_dim),
            nn.Tanh(),
            nn.Linear(self.input_dim, self.input_dim),
        )

    def forward(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.prefix_embedding(input_tokens) # (B, L, D)
        
        return prefix_prompt