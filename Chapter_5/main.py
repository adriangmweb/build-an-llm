import torch
import tiktoken
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chapter_4.gpt2_models import GPTModel, generate_text_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, # Instead of 1024 as in the original GPT-2 124M model
    "n_embd": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": True
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension (1, seq_len)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist() # Remove batch dimension and convert to list
    text = tokenizer.decode(flat)
    return text

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Untrained model output: \n")
print(token_ids_to_text(token_ids, tokenizer))
print()

# Inputs and targets example

inputs = torch.tensor([[16833, 3626, 6100], # "Every effort moves"
                        [40, 1107, 588]]) # "I really like"

targets = torch.tensor([[3626, 6100, 345], # "effort moves you"
                        [1107, 588, 11311]]) # "really like chocolate"

with torch.no_grad(): # Disables gradient calculation as we are not training
    logits = model(inputs)
probabilities = torch.softmax(logits, dim=-1)

print("Probabilities shape: \n", probabilities.shape)
print()

token_ids = torch.argmax(probabilities, dim=-1, keepdim=True)
print("Token IDs: \n", token_ids)
print()

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Output batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
print()

# Target probabilities

text_idx = 0
target_probabilities_1 = probabilities[text_idx, [0, 1, 2], targets[text_idx]] # Get the probabilities of the target tokens
print("Target probabilities batch 1: \n", target_probabilities_1)
print()

text_idx = 1
target_probabilities_2 = probabilities[text_idx, [0, 1, 2], targets[text_idx]] # Get the probabilities of the target tokens
print("Target probabilities batch 2: \n", target_probabilities_2)
print()

log_probabilities = torch.log(torch.cat([target_probabilities_1, target_probabilities_2])) # Concatenate the probabilities
print("Log probabilities: \n", log_probabilities)
print()

average_log_probability = torch.mean(log_probabilities)
print(f"Average log probability: {average_log_probability.item():.4f}")
print()

negative_average_log_probability = average_log_probability * -1
print(f"Negative average log probability: {negative_average_log_probability.item():.4f}")
print()

print("Logits shape: \n", logits.shape)
print("Targets shape: \n", targets.shape)
print()

# Flatten the logits and targets for the loss computation

logits_flat = logits.flatten(0, 1) # Flatens the batch and sequence dimensions (batch_size * seq_len, vocab_size)
targets_flat = targets.flatten() # Flatens the batch dimension (batch_size * seq_len)

print("Logits flattened shape: \n", logits_flat.shape)
print("Targets flattened shape: \n", targets_flat.shape)
print()

# Compute the loss

print("Computing the loss with cross entropy facilitates the computation of the logits and targets")
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(f"Loss: {loss.item():.4f}")
print()
