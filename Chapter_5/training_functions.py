import torch

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
    