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

# In[27]:


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

# Download pizza, steak and sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  print(f"Downloading pizza, steak, sushi data...")
  f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path/ "pizza_steak_sushi.zip", "r") as zip_ref:
  print("Unzipping pizza, steak and sushi data...")
  zip_ref.extractall(image_path)



# ## 2. Becoming one with the data (data prep and exploration)

# In[24]:


import os
def walk_through_dir(dir_path):
  """Walk throught dir_path returning its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'. ")


# In[28]:


walk_through_dir(image_path)


# In[26]:


# setup train and test paths
train_dir = image_path/ "train"
test_dir = image_path / "test"

train_dir, test_dir


# ### 2.1 Visualizing images
# 
# 1. get all of the image paths
# 2. pick a random image path using Python's `randome.choice()`
# 3. get the img class name using `pathlib.Path.parent.stem`
# 4. viz img with Python's Pillow  `PIL`
# 5. show the img metadata

# In[49]:


import random
from PIL import Image

# set seed
#random.seed(42)

# 1. get all the img paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. pick a random image path
random_image_path = random.choice(image_path_list)

# 3. get image class from path name (name of dir)
image_class = random_image_path.parent.stem

# 4. open img
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
# device agnostic img show
if device == "mps":
  img.show()
else:
  display(img)


# In[53]:


import numpy as np
import matplotlib.pyplot as plt

# Trun the img into an array
img_as_array = np.array(img)

# plot the img with matplotlib
plt.figure(figsize=(10,7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [h, w, c]")
plt.axis(False)
if device == "mps":
  plt.show()


# In[ ]:




