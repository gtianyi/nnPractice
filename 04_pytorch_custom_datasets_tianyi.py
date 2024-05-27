import torch
from torch import nn

torch.__version__

# setup device-agnostic code for gpu / mps / cpu
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
device

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
