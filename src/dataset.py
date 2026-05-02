import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

class ColoredDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class GrayscaleBSDSDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

class LabBSDSDataset(Dataset):
    """
    Dataset that returns images in LAB color space.
    L channel: Input (Grayscale)
    AB channels: Target (Color information)
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms (Resize, etc.) before converting to LAB
        if self.transform:
            image = self.transform(image)
            
        # Convert to numpy for skimage (expects 0-1 or 0-255)
        img_np = np.array(image).transpose(1, 2, 0) # CHW to HWC
        img_lab = rgb2lab(img_np)
        
        # L channel is in range [0, 100], AB channels in [-128, 127]
        # Standardize to roughly [-1, 1] or [0, 1] for the model
        img_lab[:, :, 0] = img_lab[:, :, 0] / 100.0          # L: [0, 1]
        img_lab[:, :, 1:] = (img_lab[:, :, 1:] + 128) / 255.0 # AB: [0, 1]
        
        return torch.from_numpy(img_lab.transpose(2, 0, 1)).float()
