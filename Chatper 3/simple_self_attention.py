# Simple self-attention mechanism
# This one won't be trainable but to understand the mechanism

import torch

# Example computing the context vector for the query "journey"

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)

query = inputs[1] # journey
attention_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    # Compute the attention score between the query and each key
    # Using the dot product 
    attention_scores_2[i] = torch.dot(x_i, query)

print("Attention score for journey: \n", attention_scores_2)
print()

# Softmax the attention scores
attention_scores_2_softmax = torch.softmax(attention_scores_2, dim=0)
print("Attention weights for journey: \n", attention_scores_2_softmax)
print("Sum of attention weights for journey: \n", attention_scores_2_softmax.sum())
print()

# Calculate the context vector
# The context vector is the weighted sum of the values
query = inputs[1] # journey
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attention_scores_2_softmax[i] * x_i

print("Context vector for journey: \n", context_vec_2)
print()

# Example computing the context vector for all inputs

attention_scores = inputs @ inputs.T

print("Attention scores for all inputs: \n", attention_scores)
print()

# Softmax the attention scores
attention_scores_softmax = torch.softmax(attention_scores, dim=1) # dim=1 to softmax over the rows
print("Attention weights for all inputs: \n", attention_scores_softmax)
print("Sum of attention weights for all inputs: \n", attention_scores_softmax.sum(dim=1))
print()

# Calculate the context vector
context_vec = attention_scores_softmax @ inputs
print("Context vector for all inputs: \n", context_vec)
print()

print("Context vector for journey (position 1): \n", context_vec[1])
print()
