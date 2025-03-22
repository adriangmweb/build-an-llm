import torch
import tiktoken
import sys
import os

from plot import plot_losses

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chapter_4.gpt2_models import GPTModel, generate_text_simple
from text2tokens import text_to_token_ids, token_ids_to_text

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

# Preparing the data for training

# Load the dataset

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

from Chapter_2.embeddings import create_dataloader_v1

torch.manual_seed(123)

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

# Iterate over the training data to see their shape

for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}")
    print("Inputs shape: \n", inputs.shape)
    print("Targets shape: \n", targets.shape)
    print()
    break

from training_functions import calculate_loss_loader

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    train_loss = calculate_loss_loader(train_loader, model, device)
    val_loss = calculate_loss_loader(val_loader, model, device)

print(f"Train loss: {train_loss}")
print(f"Validation loss: {val_loss}")
print()

print("***** Training the model with simple training *****")

from training_functions import train_model_simple

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
    
model.to("cpu") # Move the model to the CPU as the dataset is too small
model.eval()

print("***** Temperature scaling ***** \n")

model.to("cpu") # Move the model to the CPU as the dataset is too small
model.eval()

token_ids = generate_text_simple(
    model,
    idx=text_to_token_ids("every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("No temperature scaling model output: \n")
print(token_ids_to_text(token_ids, tokenizer))
print()

from temperature_scaling import generate

torch.manual_seed(123)
token_ids = generate(
    model,
    idx=text_to_token_ids("every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Temperature scaled model output: \n")
print(token_ids_to_text(token_ids, tokenizer))
print()

print("***** Save and load a pretrained model ***** \n")

torch.save(model.state_dict(), "model.pth")

# Disabling to save the model architecture as well
# loaded_model = GPTModel(GPT_CONFIG_124M)
# loaded_model.load_state_dict(torch.load("model.pth"), map_location=device)
# loaded_model.eval() # Set the model to evaluation mode to disable dropout as we want to disable any information from the training process

# Save the model using torch.save will save the optimizer state as well
# To save only the model parameters, we can use torch.save(model.state_dict(), "model.pth")
# We want to save the AdamW optimizer state as well as it stores additional information
# about the learning rate, weight decay, etc.

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)

checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
loaded_model = GPTModel(GPT_CONFIG_124M)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_optimizer = torch.optim.AdamW(
    loaded_model.parameters(),
    lr=5e-4,
    weight_decay=0.1
)
loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# loaded_model.train() # Set the model to training mode to enable dropout as we want to use the model for training