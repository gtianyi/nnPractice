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
# if device == "mps":
#   plt.show()


# In[ ]:





# ## 3. Trainsforming data (img /text / auido -> tensor)
# 
# 1. Trun data intot tensors
# 2. Trun it into a `torch.utils.data.Dataset` and subsequently a `torch.utils.data.DataLoader`, called `DataSet` and `DataLoader`

# In[55]:


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ## 3.1 Transforming data with torchvision.transforms

# In[56]:


# write a transform for image
data_transform = transforms.Compose([
    # Resize imgs to 64x64
    transforms.Resize(size=(64,64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn img into torch.Tensor
    transforms.ToTensor()
])


# In[62]:


print(f"data_transform shape  {data_transform(img).shape}")


# In[68]:


def plot_transformed_images(image_paths, transform, n=3, seed=42):
  """
  Selects random imgs from path and loads/transforms
  them then plots the original vs the transformed version.
  """
  if seed:
    random.seed(seed)
  random_image_paths = random.sample(image_paths, k=n)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig, ax= plt.subplots(nrows=1, ncols=2)
      ax[0].imshow(f)
      ax[0].set_title(f"Original\nSize:{f.size}")
      ax[0].axis(False)

      # Transform and plot target img
      transformed_image = transform(f).permute(1, 2, 0) # need to change shape for matplotlib (CHW -> HWC)
      ax[1].imshow(transformed_image)
      ax[1].set_title(f"transformed\nSize: {transformed_image.shape}")
      ax[1].axis("off")

      fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
      #plt.show()

plot_transformed_images(image_paths=image_path_list,
                        transform=data_transform,
                        n=3,
                        seed=None)



# ## 4. Option 1: Loading img data using ImageFolder

# In[70]:


# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, # transform for the data
                                  target_transform=None # transform for the labels/tarrgets, we don't need it bc we use dir_stem as lable)
                                 )

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

train_data, test_data


# In[72]:


# get class names as list
class_names = train_data.classes
print(class_names)


# In[73]:


# get class names as dict
class_dict = train_data.class_to_idx
print(class_dict)


# In[77]:


# check the len of our dataset
print(len(train_data), len(test_data))


# In[76]:


print(train_data.samples[0])


# In[81]:


# index on the train_data DataSet to get a single image and label
img, label = train_data[0][0], train_data[0][1]

print(f"Image tensor:\n {img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"label datatype: {type(label)}")


# In[82]:


# Rearrange the order of the dimensions
img_permute = img.permute(1,2,0)

# print out different shapes
print(f"Original shape: {img.shape} -> CHW")
print(f"Image permute shape: {img_permute.shape} -> HWC")

# plot the image
plt.figure(figsize=(10,7))
plt.imshow(img_permute)
plt.axis(False)
plt.title(class_names[label], fontsize=14)


# ## 4.1 Turn loaded images into `DataLoader`s
# 
# A `DataLoader` is used for customized `batch_size`

# In[83]:


import os
print(f"cpu count: {os.cpu_count()}")


# In[93]:


# Trun train and test datasets into DataLoader's
from torch.utils.data import DataLoader

#BATCH_SIZE=32
#CPU_COUNT = os.cpu_count()
BATCH_SIZE=1
CPU_COUNT =1

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    num_workers=CPU_COUNT,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    num_workers=CPU_COUNT,
    shuffle=False
)

print(f"dataloader size: {len(train_dataloader)}, {len(test_dataloader)}")


# In[99]:


# To avoid python spawn multiprocessing issue on python 3.8+ in MacOS
# we need to do the following to avoid Dataloader crashes if num_worker>0
# https://github.com/pytorch/pytorch/issues/46648
import multiprocessing

if device == "mps":
  multiprocessing.set_start_method("fork")

img, label = next(iter(train_dataloader))

print(f"Image shape: {img.shape} -> [batch size, C, H, W]")
print(f"Label shape: {label.shape}")


# ## 5. Option 2: Loading Image data with a Custom `DataSet`
# 
# 1. Want to be able to load img from file
# 2. want to ba able to get class name from the dataset
# 3. want to be able to get classes as dict from the dataset
# 
# By subclassing `torch.utils.data.Dataset`

# In[102]:


import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

# Instance of torchvision.datasets.ImageFolder()
print(train_data.classes, train_data.class_to_idx)


# ### 5.1 Creating a helper fn to get class names
# 
# We want to:
# 1. Get the class names using `os.scandir()` to traverse a target dirctory
# 2. Raise an error if the class names aren't found
# 3. Turn class names into a dict and list and return them

# In[106]:


# setup path for target directory
target_directory = train_dir
print(f"Target dir: {target_directory}")

# gethe class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
class_names_found


# In[109]:


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
  """Finds the classes folder names in a target directory."""
  # 1. get the class names by scanning the target directory
  classes = sorted([entry.name for entry in os.scandir(directory) if entry.is_dir()])

  # 2. raise an error if classes names could not be found
  if not classes:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}... please check file structure.")

  # 3. create a directory of index labels
  class_to_idx = {class_name: i for i , class_name in enumerate(classes)}
  return classes, class_to_idx

print(find_classes(target_directory))


# ### 5.2 Create a custom `Dataset` to replicate `ImageFolder`
# 
# To create it:
# 
# 1. subclass `Dataset`
# 2. init subclass with a target directory as well as a transform
# 3. create several attributis:
#   * path
#   * transform
#   * classes
#   * class_to_idx
# 4. Create a fn to `load_images()`
# 5. Overwrite the `__len()__` fn
# 6. Overwrite the `__getitem()__` fn

# In[110]:


# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:

        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)


# In[111]:


# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# In[112]:


train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                     transform=test_transforms)
train_data_custom, test_data_custom


# In[114]:


# Check for equality amongst our custom Dataset and ImageFolder Dataset
print((len(train_data_custom) == len(train_data)) & (len(test_data_custom) == len(test_data)))
print(train_data_custom.classes == train_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)


# ### 5.3 Create a fn to display random images from customizerd dataset

# In[116]:


# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):

    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    #plt.show()


# In[117]:


# Display random images from ImageFolder created Dataset
display_random_images(train_data,
                      n=5,
                      classes=class_names,
                      seed=None)


# In[118]:


# Display random images from ImageFolderCustom Dataset
display_random_images(train_data_custom,
                      n=12,
                      classes=class_names,
                      seed=None) # Try setting the seed for reproducible images


# ### 5.4 turn custom loaded img into DataLoader's

# In[120]:


# Turn train and test custom Dataset's into DataLoader's
from torch.utils.data import DataLoader
train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom created train Dataset
                                     batch_size=1, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=False) # don't usually need to shuffle testing data

train_dataloader_custom, test_dataloader_custom


# In[121]:


# Get image and label from custom DataLoader
img_custom, label_custom = next(iter(train_dataloader_custom))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")


# ## 6. Data augmentation - other forms of transformation
# 
# Adding diversity to training data, eg image -> rotation, darken, shift, zoom, etc
# 
# Hope our model learn more generalized

# In[122]:


from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# In[123]:


# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Plot random images
plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None
)
#plt.show()


# ## 7. Model 0: TinyVGG without a data augmentation

# ### 7.1 creating transforms and loading data for Model 0

# In[124]:


# Create simple transform
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# In[125]:


# 1. Load and transform data
from torchvision import datasets
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

# 2. Turn data into DataLoaders
import os
from torch.utils.data import DataLoader

# Setup batch size and number of workers
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

# Create DataLoader's
train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

train_dataloader_simple, test_dataloader_simple


# ### 7.2 Create TinyVGG model class

# In[126]:


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB)
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)
print(model_0)


# ### 7.3 fw pass on a single img

# In[127]:


# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader_simple))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))

# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")

