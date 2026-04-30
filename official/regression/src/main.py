import os
import numpy as np
import scipy.io as sio
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

from dataset import BSDSDataset


# Parameters
batch_size = 8
num_epochs = 5
device = 'cuda:0'
num_classes = 10

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])


trainset = BSDSDataset(
    r"C:\Users\justi\OneDrive\Documents\Q10\machine_learning_and_big_data_processing\project\official\regression\dataset\images\train",
    r"C:\Users\justi\OneDrive\Documents\Q10\machine_learning_and_big_data_processing\project\official\regression\dataset\ground_truth\train",
    transform=transform
)

valset = BSDSDataset(
    r"C:\Users\justi\OneDrive\Documents\Q10\machine_learning_and_big_data_processing\project\official\regression\dataset\images\val",
    r"C:\Users\justi\OneDrive\Documents\Q10\machine_learning_and_big_data_processing\project\official\regression\dataset\ground_truth\val",
    transform=transform
)

train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(valset,
                          batch_size=batch_size,
                          shuffle=False)

print(train_loader)


