import pandas as pd
import torch
# Create a balanced dataset with equal number of spam and ham messages
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0] # number of spam messages
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123 
    ) # sample random subset of ham messages to match the number of spam messages
    balanced_df = pd.concat([
        # concat the ham subset and the spam messages
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    """
    Split the dataset into train, validation, and test sets
    """
    df = df.sample( # shuffle the dataset
        frac=1, random_state=123
    ).reset_index(drop=True) # reset the index
    train_end = int(len(df) * train_frac) # calculate the end of the train set
    validation_end = train_end + int(len(df) * validation_frac) # calculate the end of the validation set

    # split the dataset into train, validation, and test sets
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the accuracy of the model on a dataset
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                # Get the logits for the last output token
                # Because is the token will most context
                logits = model(input_batch)[:, -1, :] 
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break

    return correct_predictions / num_examples
                
        
# Calculate the loss for a single batch focusing on the last output token
def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # Get the logits for the last output token
    loss = torch.nn.functional.cross_entropy(
        logits,
        target_batch
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

def train_classifier_simple(
        model, train_loader, validation_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    """
    Train a classifier model
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous iteration
            loss = calculate_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward() # Calculate the loss gradients
            optimizer.step() # Update the model parameters using loss gradients

            examples_seen += input_batch.shape[0] # New: tracks examples instead of tokens
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, validation_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train Loss: {train_loss:.3f}, "
                      f"Validation Loss: {val_loss:.3f}")
        
        # Calculate accuracy after each epoch
        train_acc = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_acc = calc_accuracy_loader(
            validation_loader, model, device, num_batches=eval_iter
        )

        print(f"Train Accuracy: {train_acc:.2%} | ", end="")
        print(f"Validation Accuracy: {val_acc:.2%}")
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen
        
def evaluate_model(model, train_loader, validation_loader, device, eval_iter):
    """
    Evaluate the model on the train and validation sets
    """
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calculate_loss_loader(
            validation_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

import matplotlib.pyplot as plt

def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # Plots training and validation loss
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
    label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    # Creates a second x-axis for examples seen
    # Invisible plot for aligning ticks
    # Adjusts layout to make room
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
    """
    Classify a review as spam or ham
    """
    model.eval()
    
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_embd.weight.shape[1]
    
    input_ids = input_ids[:min( # Truncates sequence if too long
        max_length, supported_context_length
    )]
    input_ids += [pad_token_id] * (max_length - len(input_ids)) # Pads sequence if too short
    
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0) # Add batch dimension (1, seq_len)
    
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :] # Get logits for last token
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
    
    
    