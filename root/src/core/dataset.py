import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab, lab2rgb


#===================================================================================================================================
#=== DATASET ===
#===================================================================================================================================
class LabColourDataset(Dataset):
    """
    Dataset that returns images in CIE LAB colour space.
        - L channel: Input (grayscale).
        - AB channels: Target (colour information).
    """
    def __init__(self, img_dir, transform=None, mode='regression', num_bins=100):
        self.img_dir = img_dir # directory where images are located
        self.transform = transform # for resizing, normalization etc.
        self.images = sorted(os.listdir(img_dir)) # sort list of filenames
        self.mode = mode # reg. or class.
        self.num_bins = num_bins # ex: 100 bins
        self.grid_size = int(np.sqrt(num_bins)) # ex: a = 10 bins, b = 10 bins
        
    def __len__(self):
        """Returns the number of images in the dataset."""
        
        return len(self.images)
    
    
    # UNUSED (check utils.py)
    def ab_to_bin(self, ab):
        """
        Converts continuous ab values [-1, 1] to discrete class indices.
            - a: [-1, 1] --> [0, 1] --> [0, grid_size] --> clipping --> float to integers.
            - b: [-1, 1] --> [0, 1] --> [0, grid_size] --> clipping --> float to integers.
            
        Args: 
            ab (list): ab values [H, W, 2].
        """
        a = ab[:, :, 0]
        a = (a + 1) / 2
        a = a * self.grid_size
        a = np.clip(a, 0, self.grid_size - 1)
        a = a.astype(int)
        
        b = ab[:, :, 1]
        b = (b + 1) / 2
        b = b * self.grid_size
        b = np.clip(b, 0, self.grid_size - 1)
        b = b.astype(int)
        
        return a * self.grid_size + b 
    
    
    # UNUSED (check utils.py)
    def bin_to_ab(self, bin_idx):
        """
        Converts class indices back to continuous ab values [-1, 1].
           - a: [0, num_bins] --> [0, grid_size] --> [0, 1] --> [-1, 1].
           - b: [0, num_bins] --> [0, grid_size] --> [0, 1] --> [-1, 1].
              
        Args:
            bin_idx (int): bin index. Note that bin_idx belongs to [0, num_bins].
        """
        a = bin_idx // self.grid_size # find the corresponding row
        a = a / self.grid_size
        a = a * 2 - 1
        a = a + (1 / self.grid_size) # place the value to the center of the bin (bin width is (1 - (-1)) / grid_size = 2/10, so need for 1/10 offset)
        
        b = bin_idx % self.grid_size #find the corresponding column
        b = b / self.grid_size
        b = b * 2 - 1
        b = b + (1 / self.grid_size)
        
        return np.stack([a, b], axis=-1)
    

    def __getitem__(self, idx):
        """Function to manage DataLoader and extract images.

        Args:
            idx (int): index of an image.

        Returns:
            (tuple): L_tensor, ab_tensor if regression. Else: L_tensor, bins, ab_tensor.
        """
        try:
            img_path = os.path.join(self.img_dir, self.images[idx]) # merge image directory and image filename
            image = Image.open(img_path).convert("RGB") # convert to RGB
            if self.transform:
                image = self.transform(image)
                
            img_np = np.array(image).transpose(1, 2, 0) # [C, H, W] to [H, W, C]
            img_lab = rgb2lab(img_np) # convert to Lab
            
            # Standardize to roughly [-1, 1] range
            L = (img_lab[:, :, 0] / 50) - 1 # [0, 100] --> [-1, 1]
            ab = img_lab[:, :, 1:] / 128 # [0, 255] --> [-1, 1]
            
            L_tensor = torch.from_numpy(L).unsqueeze(0).float() # 1D tensor
            ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).float() # [H, W, C] to [C, H, W]
            
            if self.mode == 'classification':
                bins = self.ab_to_bin(ab)
                # Return L, bins, and ab to avoid re-calculating ab from bins on CPU during training
                return L_tensor, torch.from_numpy(bins).long(), ab_tensor # .long() because PyTorch expects class indices in 64-bit integers
            else:
                return L_tensor, ab_tensor
        except Exception as e: # captures FileNotFoundError, ValueError, RuntimeError, ...
            # Silence the warning, just delete and try another one
            try:
                os.remove(img_path) # try to remove image
            except:
                pass
            new_idx = np.random.randint(0, len(self.images)) # replaces the broken sample with another random one
            return self.__getitem__(new_idx)


def get_class_weights(dataset, num_bins=100, sample_size=2000):
    """
    FOR CLASSIFICATION ONLY:
       Calculates inverse frequency weights for class rebalancing.
       
    Args:
        dataset (LabColourDataset): Dataset.
        num_bins (int): Number of bins. Defaults to 100.
        sample_size (int): Number of samples. Defaults to 2000.
    """
    
    counts = np.zeros(num_bins)
    print(f"Calculating class weights for rebalancing (sampling {sample_size} images)...")
    # Sample subset for speed
    for i in range(min(len(dataset), sample_size)):
        data = dataset[i]
        bins = data[1] # bins is the second element in classification mode
        unique, bin_counts = np.unique(bins.numpy(), return_counts=True) # ex: [1, 1, 2, 3, 3] --> [1, 2, 3], [2, 1, 2]
        counts[unique] += bin_counts
    counts += 1 # Avoid division by zero
    probs = counts / counts.sum() # ex: [bin0, bin1, bin2] --> [3, 2, 5] --> [3/10, 2/10, 5/10]
    weights = 1 / (probs + 0.1) # ex: [10/3, 10/2, 10/5]
    weights = weights / weights.sum() * num_bins # ex: [(10/3) / (31/10), (10/2) / (31/10), (10/5) / (31/10)] * num_bins => sum of probas = num_bins, but average of probas = 1
    
    return torch.from_numpy(weights).float()
