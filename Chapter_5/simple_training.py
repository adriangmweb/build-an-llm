import torch
import tiktoken
import os
import sys
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chapter_4.gpt2_models import GPTModel
from training_functions import train_model_simple
from Chapter_2.embeddings import create_dataloader_v1

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, # Instead of 1024 as in the original GPT-2 124M model
    "n_embd": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": True
}

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print("***** Loading the dataset *****")

filepath = os.path.join(os.path.dirname(__file__), "..", "the-veredict.txt")
with open(filepath, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print(f"Total characters: {total_characters}")
print(f"Total tokens: {total_tokens}")
print()

train_ratio = 0.9 # 90% of the data will be used for training
split_idx = int(train_ratio * len(text_data)) # Split 10% of the data for validation

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print()

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True, # Drop the last batch if it's not full
    shuffle=True, # Shuffle the data
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False, # Don't drop the last batch
    shuffle=False,
)

print("***** Training the model with simple training *****")

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), # All trainable weights
    lr=0.0004, # Learning rate
    weight_decay=0.1 # L2 regularization
)

num_epochs = 10
train_losses, val_losses, track_tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs,
    eval_freq=5, # Evaluate every 5 epochs
    eval_iter=5, # Evaluate every 5 batches
    start_context="Every effort moves you",
    tokenizer=tokenizer
)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

# We can see that the validation loss is higher than the training loss, which means that the model is overfitting
# We can also see that the model is learning faster in the beginning
# plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses) # Uncomment to plot the losses
