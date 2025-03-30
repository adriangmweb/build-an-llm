import sys
import os
import torch
import tiktoken

from torch.utils.data import DataLoader
from functools import partial
from InstructionDatasetClass import InstructionDataset
from functions import format_input, download_and_load_file, custom_collate_draft_fn

# Add the parent directory to the Python path to import the modules
sys.path.append("..")  # Add parent directory to Python path

from Chapter_5.gpt_download import download_and_load_gpt2
from Chapter_4.gpt2_models import GPTModel
from Chapter_5.gpt2_weights_download import load_weights_into_gpt


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)

print("Number of entries of the dataset:", len(data))


model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)

# Divide the dataset into training, validation, and testing sets
print("\n **** Dividing the dataset into training, validation, and testing sets **** \n")

train_portion = int(len(data) * 0.85) # 85% of the data for training
val_portion = int(len(data) * 0.1) # 10% of the data for validation
test_portion = len(data) - train_portion - val_portion # 5% of the data for testing

train_data = data[:train_portion]
val_data = data[train_portion:train_portion + val_portion]
test_data = data[train_portion + val_portion:]

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))
print("Number of testing samples:", len(test_data))
print()


inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

inputs, targets = custom_collate_draft_fn(batch)

# Print the inputs and targets of an example batch
print("inputs: \n", inputs)
print("targets: \n", targets)
print()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)
print()
# Customize the collate function to use the device and allowed maximum length
customized_collate_fn = partial(
    custom_collate_draft_fn,
    device=device,
    allowed_max_length=1024
)

# Set up the data loaders

num_workers = 0 # It can be increased to use multiple cores if supported by the system
batch_size = 8

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

print("Train loader inputs and targets shapes:")
for i, (inputs, targets) in enumerate(train_loader):
    print("inputs.shape:", inputs.shape)
    print("targets.shape:", targets.shape)
    print()
    if i >= 4:  # Show first 5 batches
        break


print("**** Loading the model GPT-2 Medium (355M) ****")

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True  # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

print("Model loaded successfully!")
print()


