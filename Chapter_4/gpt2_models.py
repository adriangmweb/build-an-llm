import torch
import torch.nn as nn

# Create a dummy model
class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["context_length"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(config) 
              for _ in range(config["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(config["n_embd"])
        self.out_head = nn.Linear(
            config["n_embd"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    # eps is a small value to avoid division by zero
    # this one just mimic the behavior of LayerNorm
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
from transformer_block import TransformerBlock
from layer_norm_class import LayerNorm
# Create the GPT final model
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["context_length"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])

        # Create a transformer block for each layer
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        # Create a final layer normalization
        self.final_norm = LayerNorm(config["n_embd"])

        # Create a linear layer to output the logits
        self.out_head = nn.Linear(
            config["n_embd"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # The device setting allows to train on CPU or GPU
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
        
def generate_text_simple(model, idx, # idx is (batch_size, context_size) array of token indices
                         max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # logits is (batch_size, 1, vocab_size) array of logits
        # we want to get the logits for the last token
        logits = logits[:, -1, :] # (batch_size, vocab_size)
        probs = torch.softmax(logits, dim=-1) # (batch_size, vocab_size)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (batch_size, 1)
        idx = torch.cat((idx, idx_next), dim=-1) # (batch_size, context_size + 1)

    return idx

    