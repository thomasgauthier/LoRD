#My dirty fix for offline LORA conversion on cpu on Windows
### Pick locations here:

base_model_id = 'C:/path/to/base-model' #eg C:/models/llama2-hf
target_model_id = 'C:/path/to/target-model' #eg C:/models/euryale
cache_dir = './models'
LORA_OUT_DIR = "./lora"

import os
from huggingface_hub import list_repo_files, snapshot_download

def init_transformers_model(local_path,cache_dir):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(local_path) if isfile(join(local_path, f))]
    has_safetensors = any(file.endswith('.safetensors') for file in onlyfiles)

    print(f"Model will be loaded from: {local_path}")
    if has_safetensors:
        print("Note: .safetensors found. You better don't have .bin in there.")
    return os.path.abspath(local_path), has_safetensors


# ### Downloading the base model

# In[4]:

base_model_download_path, base_model_has_safetensors = init_transformers_model(base_model_id, cache_dir)

models = {
    'base' : {
        'download_path' : base_model_download_path,
        'has_safetensors' : base_model_has_safetensors
    },
    'target' : None
}


# ### Identifying relevant model layers
# 
# Define functions to identify linear and embedding layers within transformer models. These layers are targets for LoRA adapters extraction.

# In[5]:


# This code has been modified from its original version on the Axolotl project.
# Copyright 2023 Axolotl contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft.tuners.lora import QuantLinear


def get_linear_embedding_layers(model_type):
    """
    returns the linear embedding layers needed for loras, dependent on the model arch
    """
    if model_type == "gpt_neox":
        return ["embed_in", "embed_out"]
    if model_type == "falcon":
        return ["word_embeddings", "lm_head"]
    return ["embed_tokens", "lm_head"]


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)

    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names.append(name)


    return names

def get_linear_module_names(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, state_dict={}, device_map="meta") #avoid loading weights as we won't need them
    return find_all_linear_names(model)

linear_module_names = get_linear_module_names(models['base']['download_path'])


# ### Downloading the target model

# In[6]:


target_model_download_path, target_model_has_safetensors = init_transformers_model(target_model_id, cache_dir)

models['target'] = {
    'download_path' : target_model_download_path,
    'has_safetensors' : target_model_has_safetensors
}


# ### Loading tensors from .bin files
# 
# Define functions to load PyTorch tensors from `.bin` files or `.safetensors` file.

# In[7]:


import torch
import glob

def load_pytorch_tensors(directory, device='cpu'):
    """
    Loads tensors from .bin files in the specified directory into a dictionary.

    Args:
    - directory (str): Path to the directory containing .bin files.
    - device (str): The device to load the tensors on ('cpu', 'cuda', etc.). Default is 'cpu'.

    Returns:
    - dict: A dictionary containing all tensors from the .bin files.
    """
    tensors_dict = {}
    # Use glob to find all .bin files in the directory
    file_paths = glob.glob(f"{directory}/*.bin")

    # Loop through each file and load its tensors into the dictionary
    for file_path in sorted(file_paths):
        loaded_tensors = torch.load(file_path, map_location=torch.device(device))
        for k, v in loaded_tensors.items():
            tensors_dict[k] = v

    return tensors_dict


# In[8]:


import glob
from safetensors import safe_open

def load_safetensors(directory, framework="pt", device='cpu'):
    """
    Loads tensors from .safetensors files in the specified directory into a dictionary.

    Args:
    - directory (str): Path to the directory containing .safetensors files.
    - framework (str): The framework to use ('pt' for PyTorch, 'tf' for TensorFlow, etc.). Default is 'pt'.
    - device (str): The device to load the tensors on ('cpu', 'cuda', etc.). Default is 'cpu'.

    Returns:
    - dict: A dictionary containing all tensors from the .safetensors files.
    """
    tensors_dict = {}
    # Use glob to find all .safetensors files in the directory
    file_paths = glob.glob(f"{directory}/*.safetensors")

    # Loop through each file and load its tensors into the dictionary
    for file_path in sorted(file_paths):
        with safe_open(file_path, framework=framework, device=device) as f:
            for k in f.keys():
                tensors_dict[k] = f.get_tensor(k)

    return tensors_dict


# ### Loading model weights
# 
# Load weights for both base and target models

# In[9]:


base_model_weights = load_safetensors(models['base']['download_path']) if models['base']['has_safetensors'] else load_pytorch_tensors(models['base']['download_path'])
print("Base model weights loaded.")
target_model_weights = load_safetensors(models['target']['download_path']) if models['target']['has_safetensors'] else load_pytorch_tensors(models['target']['download_path'])
print("Target model weights loaded.")

# ### Weight matrix decomposition
# 
# The crux of what we're doing here. We define a function to decompose weight matrices into low-rank matrices using SVD

# In[10]:


import torch

def _low_rank_decomposition(weight, reduced_rank=16):
    """
    Decompose a 2D matrix into low-rank matrices A and B using SVD.a

    :param weight: The matrix to decompose, of shape (H, W)
    :param reduced_rank: The final rank of the decomposition
    :return: A tuple of tensors (A, B)
    """
    if weight.dim() != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {weight.dim()} dimensions.")

    weight = weight.to(torch.float32)
    # SVD Decomposition
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Truncated matrices
    A = Vh[:reduced_rank, :]
    B = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])

    return A, B

def decompose_delta_weight(new_weight, base_weight, alpha, reduced_rank, device=None):
    if device is None:
        device = 'cpu'

    new_weight = new_weight.to(device)
    base_weight = base_weight.to(device)

    """
    Decompose the delta weight into low-rank matrices A and B, considering the alpha scaling factor.

    :param new_weight: The updated weight matrix after applying LoRA.
    :param base_weight: The original weight matrix before LoRA.
    :param alpha: The alpha scaling factor used in LoRA.
    :param reduced_rank: The rank for the low-rank decomposition.
    :return: A tuple of tensors (A, B)
    """
    delta_weight = new_weight - base_weight

    # Check if alpha is applied uniformly
    # Adjust the implementation if alpha is applied differently
    adjusted_delta_weight = delta_weight / alpha

    A, B = _low_rank_decomposition(adjusted_delta_weight, reduced_rank=reduced_rank)

    return A, B


# ## Extract the LoRAs

# In[11]:


from tqdm.notebook import tqdm

loras = {

}

# lower rank captures less of the original model, a rank of 32 is probably reasonable for small delta (task specific finetunes and such)
alpha = 1
rank = 32
print("Decomposing LORA...(This may take a few hours for larger models)")
for module in tqdm(linear_module_names):
  target_tensor = target_model_weights[module+".weight"]
  base_tensor = base_model_weights[module+".weight"]

  lora_A, lora_B = decompose_delta_weight(target_tensor, base_tensor, alpha, rank)
  loras[f"base_model.model.{module}.lora_A.weight"] = lora_A.to('cpu')
  loras[f"base_model.model.{module}.lora_B.weight"] = lora_B.to('cpu')

del target_model_weights
del base_model_weights
print("LORA decomposed.")
# ### Extracting correct module names for PEFT
# 
# PEFT config uses partial module names, let's extract them correctly

# In[12]:


def get_module_peft_name(module_name):
    return module_name.split('.')[-1]


# ### Configuring LoRA model with PEFT
# 
# Set up a PEFT LoRA configuration for the model. Load the base model and apply this configuration, saving the configuration on disk. The LoRA weights will be saved later from our SVD decomposition.

# In[13]:


from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig



lora_config = LoraConfig(
        lora_alpha=32, # Setting the alpha to the to decomposition rank value (instead of alpha value used) seems to give better performance. Further testing would be needed to understand what is the optimal alpha value to use
        lora_dropout=0,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= list(set([get_module_peft_name(e) for e in linear_module_names])),
)

print("Saving LORA to disk...")
model = AutoModelForCausalLM.from_pretrained(models['base']['download_path'], load_in_4bit=False)

peft_model = get_peft_model(model, lora_config)

# Save to disk
peft_model.save_pretrained(LORA_OUT_DIR)

del peft_model
del model
print("LORA saved to disk.")

# ### Saving LoRA adapters as SafeTensors
# 
# Save the decomposed LoRA weights along our PEFT adapter config

# In[14]:


import torch
from safetensors.torch import save_file

print("Saving LoRA adapters as SafeTensors...")
for key in loras.keys():
    loras[key] = loras[key].to('cpu').contiguous()

save_file(loras, os.path.join(LORA_OUT_DIR, 'adapter_model.safetensors'))

print("Saved LoRA adapters as SafeTensors.")
# First, let's replace the `base_model_name_or_path` value of the adapter config with the base model id instead of the local path

# In[22]:


import os
import json

print("Adding metadata...")
adapter_config_path = os.path.join(LORA_OUT_DIR, 'adapter_config.json')

# Load the configuration from the file
with open(adapter_config_path, 'r') as file:
    config = json.load(file)

# Update the base_model_name_or_path in the configuration
config['base_model_name_or_path'] = base_model_id

# Save the updated configuration back to the file
with open(adapter_config_path, 'w') as file:
    json.dump(config, file, indent=2)

print("Configuration updated successfully.")


# Now let's create a readme

# In[23]:


import yaml

# Define your metadata as a Python dictionary
metadata = {
    'library_name': 'peft',
    'base_model': base_model_id
}

# Convert the dictionary to YAML format
yaml_frontmatter = yaml.dump(metadata, sort_keys=False)

# Define your Markdown content
markdown_content = f"""
# Low-rank decomposition of [{target_model_id}](https://huggingface.co/{target_model_id}) using [{base_model_id}](https://huggingface.co/{base_model_id}) as base

Created using [LoRD](https://github.com/thomasgauthier/LoRD)
"""

# Combine the YAML frontmatter and Markdown content
full_content = f"---\n{yaml_frontmatter}---\n{markdown_content}"

adapter_readme_path = os.path.join(LORA_OUT_DIR, 'README.md')

# Write to a Markdown file
with open(adapter_readme_path, 'w') as md_file:
    md_file.write(full_content)

print("Markdown file successfully created.")
