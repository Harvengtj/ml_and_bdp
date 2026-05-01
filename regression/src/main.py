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
    r"C:\Users\justi\OneDrive\Documents\Q10\machine_learning_and_big_data_processing\project\regression\dataset\images\train",
    transform=transform
)

print(trainset.images[0])

valset = BSDSDataset(
    r"C:\Users\justi\OneDrive\Documents\Q10\machine_learning_and_big_data_processing\project\regression\dataset\images\val",
    transform=transform
)


# Create dataloaders
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          shuffle=True)

valloader = torch.utils.data.DataLoader(valset, 
                                        batch_size=batch_size,
                                        shuffle=False)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize to show images correctly
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
# Print some samples of dataset as a sanity check
# Get some random training images
dataiter = iter(trainloader)
example_images = next(dataiter)

print(example_images.shape)

# Show images
imshow(torchvision.utils.make_grid(example_images))






