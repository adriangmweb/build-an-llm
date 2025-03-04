# Self-attention mechanism
# This one will contain the queries, keys, and values trainable matrices

import torch
from self_attention_class import SelfAttention_v1, SelfAttention_v2
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1] # journey
d_in = inputs.shape[1] # 3 input dimensions
d_out = 2 # 2 output dimensions (usually the same as the input dimensions)

torch.manual_seed(123)
# grad = False because we don't want to train these parameters for now
W_Query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_Key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_Value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Compute the query, key, and value for journey
query_2 = x_2 @ W_Query
key_2 = x_2 @ W_Key
value_2 = x_2 @ W_Value

print("Query for journey: \n", query_2)
print()

# Compute the key, and value for all inputs
keys = inputs @ W_Key
values = inputs @ W_Value

print("Key for all inputs: \n", keys.shape)
print()
print("Value for all inputs: \n", values.shape)
print()

# Compute the attention scores
attention_scores = query_2 @ keys.T # all attention scores for all inputs
print("Attention scores: \n", attention_scores)
print()

# Compute the attention weights
# We scale the attention scores by the square root of the number of dimensions in the key
# This is a common practice to prevent the attention scores from becoming too large
d_k = keys.shape[-1] # number of dimensions in the key
attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
print("Attention weights: \n", attention_weights)
print()

# Compute the context vector
context_vector = attention_weights @ values
print("Context vector: \n", context_vector)
print()

# Compute the context vector using the class
torch.manual_seed(123)
self_attention = SelfAttention_v1(d_in, d_out)
context_vector = self_attention(inputs)
print("Context vector using the class: \n", context_vector)
print()

# Compute the context vector using the second class
torch.manual_seed(789)
self_attention = SelfAttention_v2(d_in, d_out)
context_vector = self_attention(inputs)
print("Context vector using the second class: \n", context_vector)