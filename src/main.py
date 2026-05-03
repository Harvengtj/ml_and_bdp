import os
import glob
import random
import numpy as np                                  # For numerical operations
import matplotlib.pyplot as plt                     # For plotting
from PIL import Image                               # For image processing
from skimage.color import rgb2lab, lab2rgb          # Provides RGB <-> Lab conversion.
import torch
import torch.nn as nn                               # For building neural networks
import torch.nn.functional as F                     # For activation functions and other utilities
import torch.optim as optim                         # For optimization algorithms
from torch.utils.data import Dataset, DataLoader    # For handling datasets and batching
import torchvision                                  # Useful for image grids

# %% Experiment settings
# Start small. GANs are much easier to debug at 64x64 than at 256x256.
image_size = 64

# Use a small batch size first. Increase only after shapes and losses work.
batch_size = 8

# First use a small number of epochs for debugging.
num_epochs = 10

# Generator warm-up epochs: train with L1 only before adversarial training.
warmup_epochs = 2

# L1 weight from the paper and pix2pix-style colorization.
lambda_l1 = 100.0

# Learning rate. The paper uses 2e-4; the repo examples use 3e-4.
lr = 2e-4

# Adam beta1 from the paper/DCGAN convention.
beta1 = 0.5

# Device selection, same idea as your labs.
device = "cuda:0"

# -----------------------------
# Dataset settings
# -----------------------------

# Use CIFAR-10 as the first base dataset for experimentation.
# torchvision will store or download it under this folder.
data_root = "./data"

# Reproducibility
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class RGBToLabTransform:
    """
    Custom PyTorch Transform to convert a PIL RGB image into normalized L and ab tensors.
    
    Output:
        L_tensor:  [1, H, W], values roughly in [-1, 1]
        ab_tensor: [2, H, W], values roughly in [-1, 1]
    """
    def __call__(self, pil_img):
        # Force RGB so grayscale or RGBA images do not break the pipeline.
        pil_img = pil_img.convert("RGB")

        # Convert PIL image to NumPy array in [0, 1].
        rgb = np.asarray(pil_img).astype(np.float32) / 255.0

        # Convert RGB to Lab.
        # rgb2lab expects RGB values in [0, 1].
        lab = rgb2lab(rgb).astype(np.float32)           # lab shape is [H, W, 3]: 1 channel for L, 2 channels for ab

        # Split Lab channels.
        L = lab[:, :, 0:1]      # [H, W, 1]
        ab = lab[:, :, 1:3]     # [H, W, 2]

        # Normalize.
        L = (L / 50.0) - 1.0
        ab = ab / 128.0

        # Convert HWC -> CHW for PyTorch.
        L = np.transpose(L, (2, 0, 1))
        ab = np.transpose(ab, (2, 0, 1))

        # Convert NumPy arrays to PyTorch tensors.
        L_tensor = torch.from_numpy(L).float()
        ab_tensor = torch.from_numpy(ab).float()

        return L_tensor, ab_tensor

# Define our transform pipeline
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size), Image.BICUBIC),
    RGBToLabTransform()
])


def lab_tensors_to_rgb(L_tensor, ab_tensor):
    """
    Convert normalized L and ab tensors back to an RGB NumPy image.

    Input:
        L_tensor:  [1, H, W] or [B, 1, H, W]
        ab_tensor: [2, H, W] or [B, 2, H, W]

    Output:
        RGB NumPy image or batch in [0, 1].
    """
    # Move tensors to CPU and detach from computation graph.
    L = L_tensor.detach().cpu()
    ab = ab_tensor.detach().cpu()

    # If input is a single image, add batch dimension temporarily.
    single = False
    if L.ndim == 3:
        L = L.unsqueeze(0)
        ab = ab.unsqueeze(0)
        single = True

    # Convert from normalized range back to Lab range.
    L = (L + 1.0) * 50.0
    ab = ab * 128.0

    # Concatenate channels: [B, 1, H, W] + [B, 2, H, W] -> [B, 3, H, W].
    lab = torch.cat([L, ab], dim=1)

    # Convert BCHW -> BHWC for skimage.
    lab = lab.permute(0, 2, 3, 1).numpy()

    rgb_images = []
    for lab_img in lab:
        # lab2rgb returns values in [0, 1].
        rgb = lab2rgb(lab_img).astype(np.float32)
        rgb_images.append(rgb)

    rgb_images = np.stack(rgb_images, axis=0)

    if single:
        return rgb_images[0]

    return rgb_images


def show_rgb_image(rgb, title=None):
    """
    Display one RGB image represented as a NumPy array in [0, 1].
    """
    rgb = np.clip(rgb, 0.0, 1.0)
    plt.figure(figsize=(4, 4))
    plt.imshow(rgb)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()