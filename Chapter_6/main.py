# Finetune the GPT-2 model on the SMS Spam Collection dataset
# First, we need to download and prepare the dataset using the download_and_prepare_dataset.py script

import os
import tiktoken
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append("..")  # Add parent directory to Python path
from functions import calc_accuracy_loader, calculate_loss_loader, classify_review, plot_values, train_classifier_simple
from classes import SpamDataSet

# check if the dataset files exist
if not os.path.exists("train_dataset.csv") or not os.path.exists("validation_dataset.csv") or not os.path.exists("test_dataset.csv"):
    print("Dataset files do not exist. Please run the download_and_prepare_dataset.py script first.")

# Load the datasets
tokenizer = tiktoken.get_encoding("gpt2")

print("***** Preparing the datasets ***** \n")

train_dataset = SpamDataSet(
    "train_dataset.csv", 
    max_length=None,
    tokenizer=tokenizer
)
validation_dataset = SpamDataSet(
    "validation_dataset.csv", 
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataSet(
    "test_dataset.csv", 
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

print("Dataset max length: ", train_dataset.max_length)

# Create a DataLoader

num_workers = 0 # Ensures compatibility with most operating systems
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=True
)

# Ensure the DataLoaders are working
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions: ", input_batch.shape) # Should be torch.Size([8, 1024]) because batch_size is 8 and max_length is 1024
print("Target batch dimensions: ", target_batch.shape) # Should be torch.Size([8]) because batch_size is 8 and it's only 0 or 1
print()

print(f"{len(train_loader)} training batches")
print(f"{len(validation_loader)} validation batches")
print(f"{len(test_loader)} test batches")

print("***** Loading the pretrained model ***** \n")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "dropout": 0.0,
    "qkv_bias": True
}

model_configs = {
"gpt2-small (124M)": {"n_embd": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"n_embd": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"n_embd": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"n_embd": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

from Chapter_5.gpt_download import download_and_load_gpt2
from Chapter_5.training_functions import load_weights_into_gpt
from Chapter_4.gpt2_models import GPTModel

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

print("Model initialized and weights loaded")

# Check if the model is generates coherent text

from Chapter_5.training_functions import generate_text_simple
from Chapter_5.text2tokens import text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)

print("Generated text: ", token_ids_to_text(token_ids, tokenizer))
print()

# Before finetuning, let's evaluate the model on classification accuracy
print("***** Evaluating the model on classification accuracy without finetuning ***** \n")

text_2 = (
"Is the following text 'spam'? Answer with 'yes' or 'no':"
" 'You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text_simple(
    model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)

print("Generated text: ", token_ids_to_text(token_ids, tokenizer))
print()

print("***** Preparing to finetune the model as a binary classifier ***** \n")

print("Model architecture:")
# Print only the last transformer block, final_layer_norm and out_head
print(model.trf_blocks[-1])
print(model.final_norm)
print(model.out_head)
print()

# Want to only finetune the output layer
# First, we freeze all the parameters of the model to disable training to all layers
for param in model.parameters():
    param.requires_grad = False

# Now, we replace the output layer with a new one that has 2 output classes (spam or not spam)
torch.manual_seed(123)
num_classes = 2
# requires_grad is set to True by default, so training is enabled
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["n_embd"], # Equals to 768 for gpt2-small
    out_features=num_classes
)

# We make final_layer_norm and the last transformer block trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

# Determine the classification accuracies across the datasets 
# We only evaluate the model on 10 batches for each dataset
# to save time
model.to(device)
torch.manual_seed(123)

print("***** Evaluating the model before finetuning ***** \n")

print("Calculate classification accuracy")
train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_acc = calc_accuracy_loader(validation_loader, model, device, num_batches=10)
test_acc = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Initial train accuracy: {train_acc:.2%}")
print(f"Initial validation accuracy: {val_acc:.2%}")
print(f"Initial test accuracy: {test_acc:.2%}")
print()

print("Calulate losses before finetuning")
with torch.no_grad():
    train_loss = calculate_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calculate_loss_loader(validation_loader, model, device, num_batches=5)
    test_loss = calculate_loss_loader(test_loader, model, device, num_batches=5)

print(f"Initial train loss: {train_loss:.3f}")
print(f"Initial validation loss: {val_loss:.3f}")
print(f"Initial test loss: {test_loss:.3f}")
print()

print("***** Finetuning the model ***** \n")

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.1
)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, validation_loader, optimizer, device, 
        num_epochs, eval_freq=50, 
        eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes")

# Plot the training and validation loss - Disabled for now
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
# plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# Plot the training and validation accuracy - Disabled for now
# epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
# examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
# plot_values(
#     epochs_tensor, examples_seen_tensor, train_accs, val_accs,
#     label="accuracy"
# )

# Calculate the classification accuracy after finetuning
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(validation_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Finetuned train accuracy: {train_accuracy:.2%}")
print(f"Finetuned validation accuracy: {val_accuracy:.2%}")
print(f"Finetuned test accuracy: {test_accuracy:.2%}")
print()

print("***** Classifying reviews with the finetuned model ***** \n")

text_1 = (
"You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award."
)

text_1_review = classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
)

print(f"Review: {text_1}")
print(f"Classification: {text_1_review}")
print()

text_2 = (
"Hey, just wanted to check if we're still on"
" for dinner tonight? Let me know!"
)

text_2_review = classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
)

print(f"Review: {text_2}")
print(f"Classification: {text_2_review}")
print()

# Save finetuned model
torch.save(model.state_dict(), "review_classifier.pth")

# Load finetuned model
model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)

print("Finetuned model loaded")

# Test the finetuned model on the test set
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Finetuned test accuracy: {test_accuracy:.2%}")
