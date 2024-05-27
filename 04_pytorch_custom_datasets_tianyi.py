#!/usr/bin/env python
# coding: utf-8

# # 04 PyTorch Custom Datasets Notebook
# 
# Get our own data into PyTorch - via custom datasets
# 
# 

# ## 0. Importing PyTorch and Setting Up Device Agnostic Code

# In[ ]:


import torch
from torch import nn

print(f"torch version {torch.__version__}")


# In[15]:


# setup device-agnostic code for gpu / mps / cpu
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"on device {device}")


# ## 1. Get data
# 
# Subset of Food101 dataset
# 
# Try things on a small scale and increase scale when necessary, speed up experiment

# In[16]:


import requests
import zipfile
from pathlib import Path

# setup device agnostic path
data_path = Path.home() / "data/" if device == "mps" else Path("data/")
image_path = data_path/ "pizza_steak_sushi"

if image_path.is_dir():
  print(f"{image_path} directory already exists...skipping download")
else:
  print(f"{image_path} does not exist, creating one...")
  image_path.mkdir(parents=True, exist_ok=True)

