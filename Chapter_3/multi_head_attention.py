# Multi-head attention
# First, we'll create a multi-head attention module
# by stacking multiple causal attention modules each with a different set of weights
# And then combining their outputs

import torch
import torch.nn as nn
from .causal_attention_class import CausalAttention

# This class is a wrapper that stacks multiple causal attention modules
# Example: 2 heads, 2 dimensions per head -> 4 dimensions in total: (d_out*num_heads = 4)
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qvk_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qvk_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat(
            # They are processed sequentially which is not efficient
            # A better approach is to use parallel processing
            [head(x) for head in self.heads],
            dim=-1
        )

# Multi-head attention class with parallel processing
# Instead of maintaining the CausalAttention class and the MultiHeadAttentionWrapper class
# We can use the MultiHeadAttention class directly
# It will include other modifications to improve efficiency
# One key detail in this class is that we only perform a matrix multiplication for the queries, keys, and values
# This is more efficient than performing a matrix multiplication for each head as in the CausalAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 num_heads, context_length, dropout, qvk_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.d_out_per_head = d_out // num_heads # Reduces the dim to match the desired output dim

        self.W_Q = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_K = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qvk_bias)

        self.out_proj = nn.Linear(d_out, d_out) # Combines the heads outputs
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # tensor shape: (b, num_tokens, d_out)  
        keys = self.W_K(x)
        queries = self.W_Q(x)
        values = self.W_V(x)

        # Reshape the keys, queries, and values to add the num_heads dimension
        # This allows us to process each head in parallel 
        # and then combine the results
        # Tensor shape for each: (b, num_tokens, num_heads, d_out_per_head)
        keys = keys.view(b, num_tokens, self.num_heads, self.d_out_per_head)
        queries = queries.view(b, num_tokens, self.num_heads, self.d_out_per_head)
        values = values.view(b, num_tokens, self.num_heads, self.d_out_per_head)

        keys = keys.transpose(1, 2) # (b, num_heads, num_tokens, d_out_per_head)
        queries = queries.transpose(1, 2) # (b, num_heads, num_tokens, d_out_per_head)
        values = values.transpose(1, 2) # (b, num_heads, num_tokens, d_out_per_head)

        # Compute the attention scores for each head
        # This is done in parallel
        # The shape of the attention scores is (b, num_heads, num_tokens, num_tokens)
        # The last two dimensions are the same as the input
        # The first dimension is the batch size
        # The second dimension is the number of heads
        attention_scores = queries @ keys.transpose(2, 3) 
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # (num_tokens, num_tokens)

        attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        context_vector = context_vector.transpose(1, 2) # back to tensor shape: (b, num_tokens, num_heads, d_out_per_head)
        
        # Combines heads outputs and reshapes to tensor shape: (b, num_tokens, d_out)
        # Where self.d_out = self.num_heads * self.d_out_per_head
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)

        # Adds the final linear layer that reduces the dim to d_out
        context_vector = self.out_proj(context_vector)

        return context_vector
        