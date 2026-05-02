"""
Model training on BSDS500 dataset
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

from dataset import ColoredDataset, GrayscaleBSDSDataset

# Hyperparameters
batch_size = 8
num_epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 10

# Image preprocessing
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# Load train and validation sets
trainset = GrayscaleBSDSDataset(
    "data/images/train",
    transform=transform
)

valset = GrayscaleBSDSDataset(
    "data/images/val",
    transform=transform
)

# Dataloaders for batching
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True
)

valloader = torch.utils.data.DataLoader(
    valset, 
    batch_size=batch_size,
    shuffle=False
)

# Helper function to display images
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Sanity check: show some random images
dataiter = iter(trainloader)
example_images = next(dataiter)

imshow(torchvision.utils.make_grid(example_images))
