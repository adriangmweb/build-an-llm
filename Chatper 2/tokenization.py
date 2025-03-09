with open("the-veredict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])
print()

# Tokenization

import re

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))
print(preprocessed[:30])
print()


# Vocabulary size

# Create a sorted list of unique words from preprocessed text
# 1. set() removes duplicates from preprocessed list
# 2. list() converts the set back to a list
# 3. sorted() alphabetically orders the list
all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)

print(f"Vocabulary size: {vocab_size}")
print()

# Create a mapping from words to indices
vocab = {word: index for index, word in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 10:
        break
print()

# create a tokenizer class

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab # A dictionary that maps words to indices
        self.int_to_str = {index: word for word, index in vocab.items()} # A dictionary that maps indices to words

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[word] for word in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV1(vocab)

text = "It's the last he painted, you know' Mrs. Gisburn said."
ids = tokenizer.encode(text)

print("Encoding and decoding a text:")

print("Encoded:", ids)
print("Decoded:", tokenizer.decode(ids))
print()

# Add special tokens
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])

vocab = {word: index for index, word in enumerate(all_tokens)}

vocab_size = len(all_tokens)

print(f"Vocabulary size: {vocab_size}")
print()

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

print()

# Create a new tokenizer class

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {index: word for word, index in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[word] for word in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join([text1, text2])
print("Text:", text)
print()
tokenizer = SimpleTokenizerV2(vocab)
print("Encoded:", tokenizer.encode(text))
print()
print("Decoded:", tokenizer.decode(tokenizer.encode(text)))
print()

# Use Byte-Pair Encoding (BPE)
from importlib.metadata import version
import tiktoken

print("Using tiktoken version:", version("tiktoken"))
print()

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>","<|unk|>"})

print("Encoded:", encoded)
print()

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
print()
