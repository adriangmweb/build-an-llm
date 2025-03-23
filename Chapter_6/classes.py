import pandas as pd
import torch
from torch.utils.data import Dataset

class SpamDataSet(Dataset):
    """
    A dataset class that loads CSV files containing text messages and their labels.
    It encodes the texts using a tokenizer and pads all sequences to match the length
    of the longest text in the dataset. If max_length is provided, sequences are truncated
    to that length instead.
    Default pad token id is 50256, which is the id for "<|endoftext|>" in the GPT-2 tokenizer.
    """
    def __init__(self, csv_file, tokenizer, max_length=None, 
                 pad_token_id=50256): # pad token id for "<|endoftext|>"
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        self.encoded_texts = [
            encoded_text[:self.max_length] # truncate to max length
            for encoded_text in self.encoded_texts
        ]

        # Pads sequences to the same length of the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long), 
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        