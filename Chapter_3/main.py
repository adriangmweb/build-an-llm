# Self-attention mechanism
# This one will contain the queries, keys, and values trainable matrices

import torch
from self_attention_class import SelfAttention_v1, SelfAttention_v2
from causal_attention_class import CausalAttention
from multi_head_attention import MultiHeadAttentionWrapper, MultiHeadAttention

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
print()

# Masked self-attention
# We want to mask the future tokens so that the model cannot see the future tokens
# This is to prevent the model from cheating
print("***** Masked self-attention: ***** \n")

queries = self_attention.W_Query(inputs)
keys = self_attention.W_Key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
print("Attention weights: \n", attention_weights)
print()

# Create a simple mask

context_length = attention_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("Simple mask: \n", mask_simple)
print()

# Apply the mask to the attention weights to zero out the future tokens
attention_weights_masked = attention_weights * mask_simple
print("Attention weights after masking: \n", attention_weights_masked)
print()

# Normalize the attention weights to sum to 1
row_sums = attention_weights_masked.sum(dim=-1, keepdim=True)
attention_weights_masked = attention_weights_masked / row_sums
print("Attention weights after normalization: \n", attention_weights_masked)
print()

# More efficient masked self-attention
# We can mask the future tokens setting the attention weights to -inf
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked_attention_scores = attention_scores.masked_fill(mask.bool(), -torch.inf)
print("Masked attention scores to -inf: \n", masked_attention_scores)
print()

# Softmax the masked attention scores
attention_weights_masked = torch.softmax(masked_attention_scores / d_k**0.5, dim=-1)
print("Attention weights after masking with softmax: \n", attention_weights_masked)
print()

# Masking additional attention weights with dropout
# Dropout is a regularization technique that randomly drops out some neurons in the network
# This helps prevent overfitting
# It only applies to training
# It's usually applied after calculating the attention scores 
# or after computing the attention weights
# We'll apply it after computing the attention weights

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # 50% of the neurons will be dropped out
example = torch.ones(6, 6)
print("Example for a 50% dropout: \n", dropout(example))
print()

# Apply the dropout to the attention weights
# Values will compensate for the dropped out neurons
# Scaling their values by the inverse of the dropout rate
# In this case, 1 / 0.5 = 2
# So, the values will be scaled by 2
torch.manual_seed(123)
attention_weights_masked = dropout(attention_weights_masked)
print("Attention weights after masking with dropout: \n", attention_weights_masked)
print()

# Duplicate the input text to ensure the code can handle batches
# Of more than one input

batch = torch.stack([inputs, inputs], dim=0) # Stack the inputs twice to create a batch
print("Shape of the batch: \n", batch.shape)
print()

print("***** Causal attention: ***** \n")
# Use the causal attention class
torch.manual_seed(123) # Set the seed for reproducibility

context_length = batch.shape[1]
causal_attention = CausalAttention(d_in, d_out, context_length, 0.0)
context_vector = causal_attention(batch)

print("Context vector: \n", context_vector)
print()
print("Context vector shape for the batch: \n", context_vector.shape)
print()

print("***** Multi-head attention with wrapper: ***** \n")

# Use the multi-head attention class
torch.manual_seed(123)

context_length = batch.shape[1] # Number of tokens in the batch: 6
d_in, d_out, num_heads = 3, 2, 2
multi_head_attention = MultiHeadAttentionWrapper(d_in, d_out, num_heads, 
                                                 context_length, dropout=0.0)
context_vector = multi_head_attention(batch)

# Context vector is (2, 6, 4)
# First dimension is 2: is the number of batches, we are duplicating the input so they are identical
# Second dimension is 6: is the number of tokens in the batch
# Third dimension is 4 (2 * 2): is the number of dimensions in the output
# 4 = 2 (d_out) * 2 (num_heads)
print("Context vector: \n", context_vector)
print()
print("Context vector shape for the batch: \n", context_vector.shape)
print()

print("***** Multi-head attention: ***** \n")

# Use the multi-head attention class
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out, num_heads = 2, 2
multi_head_attention = MultiHeadAttention(d_in, d_out, num_heads, context_length, dropout=0.0)
context_vector = multi_head_attention(batch)

print("Context vector: \n", context_vector)
print()
print("Context vector shape for the batch: \n", context_vector.shape)
print()

