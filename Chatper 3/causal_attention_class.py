# Implementing a causal attention mechanism
# With is really similar to the self-attention class
# But we need to mask the future tokens
# So that the model cannot see the future tokens
# And apply dropout to the attention weights to prevent overfitting

import torch.nn as nn
import torch

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qvk_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_Query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_Key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_Value = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape # batch size, number of tokens, number of dimensions
        # For inputs where num_tokens exceeds the context length
        # results in errors in the mask creation

        # Compute the query, key, and value matrices
        queries = self.W_Query(x)
        keys = self.W_Key(x)
        values = self.W_Value(x)
        
        # Compute the attention scores
        attention_scores = queries @ keys.transpose(1, 2) # Changes the last two dimensions for matrix multiplication
        attention_scores = attention_scores.masked_fill(
            # :num_tokens to account for cases where num_tokens in the batch is smaller than the context length
            self.mask.bool()[:num_tokens, :num_tokens], 
            -torch.inf
        )
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        return context_vector
        
        
        
        
        