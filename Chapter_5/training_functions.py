import torch

from Chapter_4.gpt2_models import generate_text_simple
from text2tokens import text_to_token_ids, token_ids_to_text

# Calculate the loss for a single batch
def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss

# Calculate the loss for a whole dataset of batches
def calculate_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # Iterates over all batches if num_batches is not specified
        num_batches = len(data_loader)
    else:
        # Reduces the number of batches to iterate over if num_batches is specified
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item() # Sums the loss of each batch
        else:
            break
    return total_loss / num_batches # Returns the average loss

def evaluate_model(model, train_loader, val_loader,
                   device, eval_iter):
    """
    Evaluates a model on the training and validation sets.
    """
    model.eval() # Sets the model to evaluation mode
    with torch.no_grad(): # Disables gradient calculation
        train_loss = calculate_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calculate_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train() # Sets the model back to training mode
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generates and prints a sample from the model.
    """
    model.eval() # Sets the model to evaluation mode
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", "")) # Compact print format
    model.train() # Sets the model back to training mode
    
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    Trains a model for a given number of epochs, evaluating it periodically.
    """
    train_losses, val_losses, track_tokens_seen = [], [], [] # Initialize lists to store losses and tokens seen
    tokens_seen, global_step = 0, -1 # Initialize tokens seen and global step

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Resets loss gradient from previous iteration
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculates loss gradients
            optimizer.step() # Updates the model weights using loss gradients

            global_step += 1

            if global_step % eval_freq == 0: # Evaluates the model periodically
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} (Step {global_step:06d}) : "
                      f"Train Loss: {train_loss:.3f}, "
                      f"Val Loss: {val_loss:.3f}"
                )
                
        # Prints a sample after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
                    