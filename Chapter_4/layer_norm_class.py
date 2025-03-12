import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # eps is a small value to avoid division by zero
        # this one just mimic the behavior of LayerNorm
        self.eps = 1e-5
        # scale and shift are learnable parameters
        # scale is initialized to 1 and shift to 0
        # they are used to scale and shift the normalized output
        # this is equivalent to the gamma and beta in the original LayerNorm
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # normalize the input
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # scale and shift the normalized input
        return norm_x * self.scale + self.shift