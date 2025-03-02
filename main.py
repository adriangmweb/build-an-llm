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
