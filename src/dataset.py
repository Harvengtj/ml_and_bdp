import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

class ColoredDataset(Dataset):
    """Dataset for simple RGB loading."""
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

class LabColorDataset(Dataset):
    """
    Dataset that returns images in CIE LAB colour space.
    L channel: Input (Grayscale)
    AB channels: Target (Colour information)
    """
    def __init__(self, img_dir, transform=None, mode='regression', num_bins=100):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.mode = mode
        self.num_bins = num_bins
        self.grid_size = int(np.sqrt(num_bins))
        
    def __len__(self):
        return len(self.images)
    
    def ab_to_bin(self, ab):
        """Converts continuous ab values [-1, 1] to discrete class indices."""
        a = np.clip(((ab[:, :, 0] + 1) / 2.0) * self.grid_size, 0, self.grid_size - 1).astype(int)
        b = np.clip(((ab[:, :, 1] + 1) / 2.0) * self.grid_size, 0, self.grid_size - 1).astype(int)
        return a * self.grid_size + b

    def bin_to_ab(self, bin_idx):
        """Converts class indices back to continuous ab values [-1, 1]."""
        a = ((bin_idx // self.grid_size) / self.grid_size) * 2.0 - 1.0 + (1.0 / self.grid_size)
        b = ((bin_idx % self.grid_size) / self.grid_size) * 2.0 - 1.0 + (1.0 / self.grid_size)
        return np.stack([a, b], axis=-1)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        img_np = np.array(image).transpose(1, 2, 0) # CHW to HWC
        img_lab = rgb2lab(img_np)
        
        # Standardise to roughly [-1, 1] range
        L = (img_lab[:, :, 0] / 50.0) - 1.0
        ab = img_lab[:, :, 1:] / 128.0
        
        L_tensor = torch.from_numpy(L).unsqueeze(0).float()
        
        if self.mode == 'classification':
            bins = self.ab_to_bin(ab)
            return L_tensor, torch.from_numpy(bins).long()
        else:
            ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).float()
            return L_tensor, ab_tensor

def get_class_weights(dataset, num_bins=100):
    """Calculates inverse frequency weights for class rebalancing."""
    counts = np.zeros(num_bins)
    print("Calculating class weights for rebalancing...")
    # Sample subset for speed
    for i in range(min(len(dataset), 500)):
        _, bins = dataset[i]
        unique, bin_counts = np.unique(bins.numpy(), return_counts=True)
        counts[unique] += bin_counts
    counts += 1 # Avoid division by zero
    probs = counts / counts.sum()
    weights = 1.0 / (probs + 0.01)
    weights = weights / weights.sum() * num_bins
    return torch.from_numpy(weights).float()
