import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio

#=======================================================================================
#=== Dataset ===
#=======================================================================================
class ColoredDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir # image directory
        #self.gt_dir = gt_dir # ground truth directory
        self.transform = transform
        self.target_transform = target_transform
        
        self.images = sorted(os.listdir(img_dir))
        #self.gts = sorted(os.listdir(gt_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # === Image (.jpg) ===
        img_path = os.path.join(self.img_dir, self.images[idx]) # ex: dataset\images\test\2018
        image = Image.open(img_path).convert("RGB")
        
        # === Ground truth (.mat) ===
        #gt_path = os.path.join(self.gt_dir, self.gts[idx]) # ex: dataset\ground_truth\test\2018
        #mat = sio.loadmat(gt_path)
        #gt = mat['groundTruth']
        # Extraction
        #gt = gt[0][0]['Segmentation'][0][0]
        #gt = gt.astype(np.int64)
            
        # === Transforms ===
        if self.transform:
            image = self.transform(image)
            
        #gt = torch.tensor(gt, dtype=torch.long)
        
        return image
    
    
    
class GrayscaleBSDSDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir # image directory
        #self.gt_dir = gt_dir # ground truth directory
        self.transform = transform
        self.target_transform = target_transform
        
        self.images = sorted(os.listdir(img_dir))
        #self.gts = sorted(os.listdir(gt_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # === Image (.jpg) ===
        img_path = os.path.join(self.img_dir, self.images[idx]) # ex: dataset\images\test\2018
        image = Image.open(img_path).convert("L")
        
        # === Ground truth (.mat) ===
        #gt_path = os.path.join(self.gt_dir, self.gts[idx]) # ex: dataset\ground_truth\test\2018
        #mat = sio.loadmat(gt_path)
        #gt = mat['groundTruth']
        # Extraction
        #gt = gt[0][0]['Segmentation'][0][0]
        #gt = gt.astype(np.int64)
            
        # === Transforms ===
        if self.transform:
            image = self.transform(image)
            
        #gt = torch.tensor(gt, dtype=torch.long)
        
        return image
    
    
            
        
         
        
        
            