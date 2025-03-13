# Also called skip connections or residual connections
# They add a shortcut from the input to the output
# Adding the previous layer's output to the next layer's input

import torch
from torch import nn
from feed_forward_class import GELU

# An example of a deep neural network
# To see how adding shortcut connections can help with the vanishing gradient problem
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_skip_connections=False):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x) # Compute the output of the current layer
            # If the network has skip connections, 
            # add the output of the current layer to the input of the next layer
            if self.use_skip_connections and x.shape == layer_output.shape:
                x = x + layer_output 
            else:
                # Otherwise, use the output of the current layer as the input for the next layer
                x = layer_output 
        return x


# function that computes the gradients in the model’s back-ward pass
def print_gradients(model, x):
    output = model(x) # forward pass
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    # Calculate the loss based on how close the target is to the output
    loss = loss(output, target)

    # Backward pass to compute the gradients of the loss with respect to the model's parameters
    loss.backward()

    # Print the gradients of the model's parameters
    for name, param in model.named_parameters():
        if 'weight' in name: # Only print the gradients of the model's weights
            # Print the mean of the absolute value of the gradient
            print(f"{name}: has gradient mean of {param.grad.abs().mean().item()}")
