import json
import os
import urllib.request

import torch

def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input(entry):
    """
    Format the input for the model to follow Alpaca prompt style.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text


def custom_collate_draft_1(
    batch,
    pad_token_id=50256, # Token ID for "<|endoftext|>"
    device="cpu",
):
    """
    Custom collate function to pad the sequences to the longest sequence in the batch.
    """
    batch_max_length = max(len(item)+1 for item in batch) # finds the longest sequence in the batch and adds 1 for the response token
    inputs_list = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # removes the extra token added to the end of the sequence
        inputs_list.append(inputs)

    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor

def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu",
):
    """
    Custom collate function to pad the sequences to the longest sequence in the batch.
    It also generates the the target token IDs from the input token IDs.
    """
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # Truncates the last token for the inputs
        targets = torch.tensor(padded[1:]) # Shifts the targets by one token to the left

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor

def custom_collate_draft_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
):
    """
    Custom collate function to pad the sequences to the longest sequence in the batch.
    It also generates the the target token IDs from the input token IDs.
    Also replaces the pad token ID with -100 in the targets tensor, except for the first pad token.
    """
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1]) # Truncates the last token for the inputs
        targets = torch.tensor(padded[1:]) # Shifts the targets by one token to the left

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index # Replaces all the pad tokens with -100, except for the first pad token

        if allowed_max_length is not None: # Truncates the inputs and targets to the allowed maximum length
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor

