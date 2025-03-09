# Tokenize the text file
import tiktoken
import torch
tokenizer = tiktoken.get_encoding("gpt2")

with open("the-veredict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# Remove first 50 tokens from the dataset
enc_sample = enc_text[50:]

content_size = 4

x = enc_sample[:content_size] # The input token sequence
y = enc_sample[1:content_size+1] # The target token to predict
print(f"x: {x}")
print(f"y: {y}")
print()

for i in range(1, content_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"Context: {context} -----> Desired: {desired}")
print()

# Convert to strings
print("Same example but in strings:")
for i in range(1, content_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"Context: {tokenizer.decode(context)} -----> Desired: {tokenizer.decode([desired])}")
print()

# Create a dataset
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        tokens_ids = tokenizer.encode(txt)

        for i in range(0, len(tokens_ids) - max_length, stride):
            input_chunk = tokens_ids[i:i + max_length]
            target_chunk = tokens_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

print("Simple 1 step example:")
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1, 
    shuffle=False,
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)
print()

print("Example with batch size 8 and stride 4:")
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False,
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n ", inputs)
print("Targets:\n ", targets)
print()

# Create embeddings

inputs_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123) # Set the seed for reproducibility

embeddings = torch.nn.Embedding(vocab_size, output_dim) # Create the embeddings layer

print("Embeddings weights: \n", embeddings.weight) # Print the weights of the embeddings layer
print()

print("Embeddings of the inputs: \n", embeddings(inputs_ids)) # Print the embeddings of the inputs
print()

output_dim = 256
vocab_size = 50257 # The vocabulary size of the tokenizer
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("Token embedding layer: \n", token_embedding_layer)
print()

batch_size = 8
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=batch_size, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs: \n", inputs)
print()

print("Inputs shape: \n", inputs.shape)
print()

# Create the embeddings
embeddings = token_embedding_layer(inputs)
print("Embeddings: \n", embeddings)
print()

# Number of tokens in the batch (Items), number of tokens in the sequence (Rows), embedding dimension (Columns)
print("Embeddings shape: \n", embeddings.shape) 
print()

# Position embeddings
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Positional embeddings: \n", pos_embeddings)
print()

print("Positional embeddings shape: \n", pos_embeddings.shape)
print()

# Add the positional embeddings to the token embeddings
embeddings_with_pos = embeddings + pos_embeddings
print("Embeddings with positional embeddings: \n", embeddings_with_pos.shape)
print()
