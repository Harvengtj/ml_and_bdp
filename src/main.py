# %% Imports
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
image_size = 64
batch_size = 8
num_epochs = 10
warmup_epochs = 2           # Generator warm-up epochs (L1 only before adversarial training)
lambda_l1 = 100.0           # L1 weight (pix2pix-style colorization)
lr = 2e-4                   # Learning rate
beta1 = 0.5                 # Adam beta1
device = "cuda:0"           # Device to use for training

# %% Dataset settings
data_root = "./data"

def rgb_to_lab_transform(pil_img):
    """
    Transform function that converts a PIL RGB image into normalized L and ab tensors.

    Output:
        L_tensor:  [1, H, W], values roughly in [-1, 1]
        ab_tensor: [2, H, W], values roughly in [-1, 1]
    """
    # Force RGB so grayscale or RGBA images do not break the pipeline.
    pil_img = pil_img.convert("RGB")

    # Convert PIL image to NumPy array in [0, 1].
    rgb = np.asarray(pil_img).astype(np.float32) / 255.0

    # Convert RGB to Lab.
    # rgb2lab expects RGB values in [0, 1].
    lab = rgb2lab(rgb).astype(np.float32)

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

class LabColorizationDataset(Dataset):
    """
    Generic wrapper for grayscale-to-color training.

    Many torchvision datasets return:
        image, class_label

    The base CIFAR-10 dataset returns:
        (L, ab), class_label

    This wrapper ignores the class label and returns:
        L:  [1, H, W]
        ab: [2, H, W]
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Because CIFAR-10 received transform=..., the image has already become
        # the pair (L, ab). The original CIFAR-10 label is not used.
        (L, ab), _ = self.base_dataset[index]
        return L, ab

# Define transforms in the same place/style as Lab 5.
# Keep PIL image transforms before rgb_to_lab_transform.
# rgb_to_lab_transform is the final transform because it returns tensors.
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5), # This increases the dataset without needing for more samples. I don't think we will need it for now anyways
    rgb_to_lab_transform,
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
    rgb_to_lab_transform,
])


# Define the base datasets outside the colorization wrapper.
# This is the only part you change when you want a different dataset.
cifar_train_base = torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=train_transform,
)

cifar_val_base = torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=val_transform,
)

# Wrap the base datasets to discard CIFAR-10 labels.
trainset = LabColorizationDataset(
    base_dataset=cifar_train_base,
)

valset = LabColorizationDataset(
    base_dataset=cifar_val_base,
)

trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

valloader = DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

print(f"Training images: {len(trainset)}")
print(f"Validation images: {len(valset)}")
print(f"Device: {device}")