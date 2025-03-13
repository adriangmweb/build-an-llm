# Tokenize the text file
import tiktoken
import torch
from gpt2_models import DummyGPTModel, GPTModel
import torch.nn as nn
from layer_norm_class import LayerNorm
from feed_forward_class import FeedForward
from shortcut_connections import ExampleDeepNeuralNetwork, print_gradients
from transformer_block import TransformerBlock

tokenizer = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)
print("Text batch: \n", batch)
print()

print("**** Use Dummy Model **** \n")

# Set up the configuration for the model from the smallest GPT-2 model
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "n_embd": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "dropout": 0.1, # Dropout rate
    "qkv_bias": False # Query-key-value bias
}

# Initialize the model
torch.manual_seed(123)

model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Logits shape: \n", logits.shape)
print("Logits: \n", logits)
print()

print("**** Normalization **** \n")

torch.manual_seed(123)
batch_example = torch.randn(2, 5) # 2 samples of 5 dimensions
layer_norm = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer_norm(batch_example)
print("Output: \n", out)

mean = out.mean(dim=-1, keepdim=True) # dim=-1 calculates the mean horizontally
var = out.var(dim=-1, keepdim=True)
print("Mean: \n", mean)
print("Variance: \n", var)
print()

# Apply the normalization formula
# (out - mean) / sqrt(var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)
print("Normalized LayerOutput: \n", out_norm)
print("Normalized Mean: \n", mean)
print("Normalized Variance: \n", var)
print()

print("**** Use LayerNorm class **** \n")

layer_norm = LayerNorm(emb_dim=5)
out_ln = layer_norm(batch_example)
mean_ln = out_ln.mean(dim=-1, keepdim=True)
var_ln = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print("Mean: \n", mean_ln)
print("Variance: \n", var_ln)
print()

print("**** Use FeedForward class **** \n")

feed_forward = FeedForward(config=GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out_ff = feed_forward(x)
print("FeedForward output shape: \n", out_ff.shape)
print()

print("**** Use Shortcut Connections **** \n")

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])

print("Without shortcut connections, gradient gets smaller <vanishing gradient problem>: \n")
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_skip_connections=False)
print_gradients(model_without_shortcut, sample_input)
print()

print("With shortcut connections, gradient is more stable: \n")
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_skip_connections=True)
print_gradients(model_with_shortcut, sample_input)
print()

print("**** Use TransformerBlock class **** \n")

torch.manual_seed(123)
config = GPT_CONFIG_124M
x = torch.randn(2, 4, 768)

model = TransformerBlock(config)
output = model(x)
print("Input shape: \n", x.shape)
print("TransformerBlock output shape: \n", output.shape)
print()

print("**** Use the final GPTModel class **** \n")

torch.manual_seed(123)
gpt_model = GPTModel(GPT_CONFIG_124M)

output = gpt_model(batch)
print("Input batch: \n", batch)
print("Output shape: \n", output.shape)
print("Output: \n", output)
print()

total_params = sum(p.numel() for p in gpt_model.parameters())
print(f"Total number of parameters: {total_params:,} \n")

# We are seeing more than 124M parameters due to weight tying
# Which means that the model is reusing the weights of the embedding layer
# and the output layer

# We can see that the weights of the embedding layer and the output layer
# are the same

print("Token embedding layer shape: \n", gpt_model.tok_emb.weight.shape)
print("Output layer shape: \n", gpt_model.out_head.weight.shape)
print()

# Let's try removing the output layer parameter count
total_params_without_output = (
    total_params - sum(p.numel() 
                       for p in gpt_model.out_head.parameters()
                       )
)

# Now it matches the 124M parameters of the original GPT-2 124M model
print(f"Total number of parameters without output layer: {total_params_without_output:,} \n")
print()

# Compute the memory requirements for the 163M parameters
total_size_bytes = total_params * 4 # 4 bytes per parameter (float32)
total_size_mb = total_size_bytes / (1024 * 1024) # Convert to MB
print(f"Total size of the model: {total_size_mb:.2f} MB \n")
print()

