# Self-attention class
# The actual implementation of the self-attention mechanism to be used in a transformer model

import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_Query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_Key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_Value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        queries = x @ self.W_Query
        keys = x @ self.W_Key
        values = x @ self.W_Value

        # Compute the attention scores
        attention_scores = queries @ keys.T
        # We scale the attention scores by the square root of the number of dimensions in the key
        # This is a common practice to prevent the attention scores from becoming too large
        d_k = keys.shape[-1]
        attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
        
        # Compute the context vector
        context_vector = attention_weights @ values
        
        return context_vector
        

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qvk_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Using nn.Linear to create the query, key, and value matrices
        # nn.Linear is better than nn.Parameter for a few reasons:
        # 1. It handles both weights and biases in a single layer
        # 2. It automatically initializes the weights using Kaiming/Xavier initialization
        #    which helps with training stability compared to random initialization
        # 3. It handles the matrix multiplication internally in an optimized way
        # 4. It's more idiomatic PyTorch and integrates better with the rest of the framework
        # bias=qvk_bias means that the bias is not used
        self.W_Query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_Key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_Value = nn.Linear(d_in, d_out, bias=qvk_bias)
        
    def forward(self, x):
        queries = self.W_Query(x)
        keys = self.W_Key(x)
        values = self.W_Value(x)
        
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        
        context_vector = attention_weights @ values
        return context_vector
        
        
        
        