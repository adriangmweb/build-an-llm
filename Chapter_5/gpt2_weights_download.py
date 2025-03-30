# Run the gpt_download.py file
import torch
from Chapter_4.gpt2_models import GPTModel
from Chapter_5.training_functions import load_weights_into_gpt
from .gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"
)

print("Settings: ", settings) # Similar to the GPT_CONFIG_124M dictionary
print("Parameter dictionary keys: ", params.keys()) # The actual weights are stored in the 'params' dictionary

print("Params WTE: ", params["wte"])
print("Params WTE shape: ", params["wte"].shape)

