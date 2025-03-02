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
