import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Applies the GELU activation function
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            # Expands the embedding dimension by 4x
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            # Applies the GELU activation function
            GELU(),
            # Shrinks the embedding dimension back to the original dimension
            nn.Linear(4 * config["n_embd"], config["n_embd"])
        )
        
    def forward(self, x):
        return self.layers(x)
        