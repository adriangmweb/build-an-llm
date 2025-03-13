# A transformer block is a single unit of the transformer model
# It consists of a self-attention mechanism and a feed-forward network
# The output of the self-attention mechanism is added to the input to form a residual connection
# The output of the feed-forward network is added to the residual connection to form a second residual connection
# Here we combine the classes from the previous files to create a transformer block

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from layer_norm_class import LayerNorm
from feed_forward_class import FeedForward
from Chapter_3.multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["n_embd"],
            d_out=config["n_embd"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            qvk_bias=config["qkv_bias"]
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["n_embd"])
        self.norm2 = LayerNorm(config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        # Normalize the input before applying the attention mechanism
        # Old transformer blocks used layer norm post-attention
        # New transformer blocks use layer norm pre-attention
        # This is because it leads to better training stability
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut # Adds the original input back to the output

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut # Adds the previous layer's output back to the output
        return x
        
        
        