# Tokenize the text file
import tiktoken
import torch
from gpt2_dummy_model import DummyGPTModel

tokenizer = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)
print("Text batch: \n", batch)
print()

print("**** Use Dummy Model **** \n")

# Set up the configuration for the model from the smallest GPT-2 model
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "n_embd": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "dropout": 0.1, # Dropout rate
    "qkv_bias": False # Query-key-value bias
}

# Initialize the model
torch.manual_seed(123)

model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Logits shape: \n", logits.shape)
print("Logits: \n", logits)