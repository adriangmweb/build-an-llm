import torch
import tiktoken
import sys
import os

from temperature_scaling import generate
from gpt_download import download_and_load_gpt2

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

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
filepath = os.path.join(project_root, "the-veredict.txt")

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

from training_functions import calculate_loss_loader, load_weights_into_gpt

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    train_loss = calculate_loss_loader(train_loader, model, device)
    val_loss = calculate_loss_loader(val_loader, model, device)

print(f"Train loss: {train_loss}")
print(f"Validation loss: {val_loss}")
print()

print("***** Training the model with simple training ***** \n")
print("To see it in action, please run the script simple_training.py \n")

print("***** Testing the model with GPT-2 weights ***** \n")

# Load GPT-2 weights
model_configs = {
    "gpt2_small (124M)": {"embd_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2_medium (355M)": {"embd_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2_large (760M)": {"embd_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2_xl (137M)": {"embd_dim": 1600, "n_layers": 48, "n_heads": 25}
}

# load the smallest model
model_size = "gpt2_small (124M)"
NEW_MODEL_CONFIG = GPT_CONFIG_124M.copy()
NEW_MODEL_CONFIG.update(model_configs[model_size])
NEW_MODEL_CONFIG.update({"context_length": 1024})
NEW_MODEL_CONFIG.update({"qkv_bias": True}) # Don't used in modern models but used in the original GPT-2 124M model

settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"
)

gpt2_small_model = GPTModel(NEW_MODEL_CONFIG)
gpt2_small_model.eval()

# Load the weights
load_weights_into_gpt(gpt2_small_model, params)
gpt2_small_model.to(device)

# Test the model
torch.manual_seed(123)
token_ids = generate(
    gpt2_small_model,
    idx=text_to_token_ids("every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_MODEL_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("GPT-2 small model output: \n")
print(token_ids_to_text(token_ids, tokenizer))
print()
