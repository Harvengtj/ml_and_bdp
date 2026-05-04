# %% [markdown]
# ## Imports
import os
import random
from zipfile import Path
import numpy as np                                  # For numerical operations
import matplotlib.pyplot as plt                     # For plotting
from PIL import Image                               # For image processing
from skimage.color import rgb2lab, lab2rgb          # Provides RGB <-> Lab conversion.
import torch
import torch.nn as nn                               # For building neural networks
import torch.nn.functional as F                     # For activation functions and other utilities
import torch.optim as optim                         # For optimization algorithms
from torch.utils.data import Dataset, DataLoader, random_split    # For handling datasets and batching
import torchvision                                  # Useful for image grids

# %% [markdown]
# ## Experiment settings
data_root = "C:/Cloud/OneDrive - Université Libre de Bruxelles/1 - School/ULB/Master/MA2/2025-26 - Q2/ELEC-Y591 - Machine Learning And Big Data Processes/ELEC-Y591 - Project Workspace/data"
image_size = 32
batch_size = 8
num_epochs = 10
warmup_epochs = 2           # Generator warm-up epochs (L1 only before adversarial training)
lambda_l1 = 100.0           # L1 weight (pix2pix-style colorization)
lr = 2e-4                   # Learning rate
beta1 = 0.5                 # Adam beta1
device = "cuda:0"           # Device to use for training
seed = 42
num_workers = 0
pin_memory = str(device).startswith("cuda")

# %% [markdown]
# ## Lab Conversion Helpers
def rgb_to_lab_transform(pil_img):
    """
    Transform function that converts a PIL RGB image into normalized L and lab tensors.

    Output:
        L_tensor:   [1, H, W], values roughly in [-1, 1]
        lab_tensor: [3, H, W], values roughly in [-1, 1]
    """
    # Handle grayscale or RGBA images
    pil_img = pil_img.convert("RGB")

    # Convert PIL image to NumPy array in [0, 1].
    rgb = np.asarray(pil_img).astype(np.float32) / 255.0

    # Convert RGB to Lab.
    lab = rgb2lab(rgb).astype(np.float32)

    # Split L channel for condition.
    L = lab[:, :, 0:1]

    # Normalize. L is [0, 100], ab is [-128, 127]
    L_norm = (L / 50.0) - 1.0
    ab_norm = lab[:, :, 1:3] / 128.0
    lab_norm = np.concatenate([L_norm, ab_norm], axis=2)

    # Convert HWC -> CHW for PyTorch.
    L_norm = np.transpose(L_norm, (2, 0, 1))
    lab_norm = np.transpose(lab_norm, (2, 0, 1))

    # Convert NumPy arrays to PyTorch tensors.
    L_tensor = torch.from_numpy(L_norm).float()
    lab_tensor = torch.from_numpy(lab_norm).float()

    return L_tensor, lab_tensor


def lab_tensors_to_rgb(L_tensor, lab_tensor):
    """
    Convert normalized lab tensor back to an RGB NumPy image.

    Input:
        L_tensor:   [1, H, W] or [B, 1, H, W] (Unused here, kept for API compatibility)
        lab_tensor: [3, H, W] or [B, 3, H, W]

    Output:
        RGB NumPy image or batch in [0, 1].
    """
    lab = lab_tensor.detach().cpu()

    # If input is a single image, add batch dimension temporarily.
    single = False
    if lab.ndim == 3:
        lab = lab.unsqueeze(0)
        single = True

    # Convert from normalized range back to Lab range.
    L_unnorm = (lab[:, 0:1, :, :] + 1.0) * 50.0
    ab_unnorm = lab[:, 1:3, :, :] * 128.0

    lab = torch.cat([L_unnorm, ab_unnorm], dim=1)
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

# %% [markdown]
# ## Build the Colorization Dataset
class LabColorizationDataset(Dataset):
    """
    Generic wrapper for grayscale-to-color training from torchvision datasets.

    Many torchvision datasets return:
        image, class_label

    If a base dataset returns:
        (L, ab), class_label

    This wrapper ignores the class label and returns:
        L:  [1, H, W]
        lab: [3, H, W]
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # The base dataset returns ((L, lab), label) because of the transform.
        # Discard the label, we only need the images for colorization.
        (L, lab), _ = self.base_dataset[index]
        return L, lab

# %% [markdown]
# ## Create DataLoaders
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5), # This increases the dataset without needing for more samples. I don't think we will need it for now anyways
    rgb_to_lab_transform,
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
    rgb_to_lab_transform,
])


# Base dataset.
# Use the smaller Places365 validation split for development.
# split="train-standard" downloads the full training archive, which is much larger.
places_base = torchvision.datasets.Places365(
    root=data_root,
    split="val",
    small=True,
    download=True,
    transform=train_transform,
)

train_size = int(0.8 * len(places_base))
val_size = len(places_base) - train_size

places_train_base, places_val_base = random_split(
    places_base,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed),
)

# Wrapper to discard Places365 labels.
trainset = LabColorizationDataset(base_dataset=places_train_base)
valset = LabColorizationDataset(
    base_dataset=places_val_base,
)

trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

valloader = DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

print(f"Training images: {len(trainset)}")
print(f"Validation images: {len(valset)}")
print(f"DataLoader workers: {num_workers}")
print(f"Device: {device}")

L_batch, lab_batch = next(iter(trainloader))
print(L_batch.shape)
print(lab_batch.shape)
print(L_batch.min().item(), L_batch.max().item())
print(lab_batch.min().item(), lab_batch.max().item())


# %% [markdown]
# ## Inspect One Batch

def show_colorization_batch(L_batch, lab_batch, max_images=4, title="Batch"):
    """
    Show L grayscale inputs and RGB images reconstructed from real ab.
    """
    L_batch = L_batch[:max_images]
    lab_batch = lab_batch[:max_images]

    rgb_batch = lab_tensors_to_rgb(L_batch, lab_batch)

    n = len(rgb_batch)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(n):
        # Show L channel as grayscale.
        L_img = L_batch[i, 0].detach().cpu().numpy()
        L_img = (L_img + 1.0) / 2.0

        axes[0, i].imshow(L_img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title("L input")

        axes[1, i].imshow(np.clip(rgb_batch[i], 0, 1))
        axes[1, i].axis("off")
        axes[1, i].set_title("Real RGB")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


L_batch, lab_batch = next(iter(trainloader))
show_colorization_batch(L_batch, lab_batch, max_images=4, title="Training examples")