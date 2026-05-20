import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse

from core.networks import Generator
from core.dataset import LabColourDataset
from core.utils import bins_to_ab_differentiable, lab_to_rgb_tensor


#===============================================================================================================================
#=== COMPARE PLOTS: GRAYSCALE, GENERATED, GROUND TRUTH FOR LATEST SAVED EPOCH ===
#===============================================================================================================================
def generate_for_epoch(mode, num_samples=5):
    """
    Generate (5x3) figure as follows |grayscale|generated|ground truth|.
    Regression or classification has to be precised.
    5 examples are fetched from the validation set.
    
    Args:
        mode (str): 'regression' or 'classification'
        num_samples (int, optional): Number of samples from validation set. Defaults to 5.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating results for {mode} on {device}...")

    # --- Configuration ---
    image_size = 256
    num_bins = 100
    base_path = "data/coco" 
    val_path = os.path.join(base_path, "val")
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    if not os.path.exists(val_path):
        print(f"Error: Validation path {val_path} not found.")
        return

    # --- Load Dataset ---
    dataset = LabColourDataset(val_path, transform=transform)
    random_indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, random_indices) # take subset of validation set (5 images by default)
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)

    # --- Load Model ---
    use_cls = (mode == 'classification')
    netG = Generator(image_size=image_size, use_classification=use_cls, num_bins=num_bins).to(device)
    
    # We assume checkpoints are saved as 'models/checkpoint_{mode}.pth'
    # and contain 'netG_state_dict'. 
    # NOTE: Your current training script overwrites the same checkpoint file.
    # To load a specific epoch, you would need to have saved them separately 
    # (e.g., checkpoint_zhang_epoch_10.pth).
    # If you only have the LATEST checkpoint, this script will load that one.
    
    checkpoint_path = f'models/checkpoint_{mode}.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict']) # load model parameters
    actual_epoch = checkpoint['epoch'] + 1
    print(f"Loaded model from checkpoint (Actual epoch in file: {actual_epoch})")
    
    netG.eval()

    # --- Inference ---
    data = next(iter(loader))
    L = data[0].to(device).float()
    target_ab = data[1].to(device).float()

    with torch.no_grad():
        if use_cls:
            logits = netG(L)
            # Use standard Zhang temperature
            fake_ab = bins_to_ab_differentiable(logits, dataset, device, temperature=0.38)
        else:
            fake_ab = netG(L)

    # Convert to RGB
    real_rgb = lab_to_rgb_tensor(L, target_ab).cpu().numpy().transpose(0, 2, 3, 1)
    fake_rgb = lab_to_rgb_tensor(L, fake_ab).cpu().numpy().transpose(0, 2, 3, 1)
    L_np = L.cpu().numpy().squeeze(1)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    fig.suptitle(f"Results: {mode.capitalize()} | Epoch: {actual_epoch}", fontsize=18, fontweight='bold', y=1.02)
    
    titles = ["Input (L)", f"Prediction ({mode})", "Ground truth"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=14, pad=10)

    for i in range(num_samples): # gray, fake, real
        axes[i, 0].imshow(L_np[i], cmap='gray')
        axes[i, 1].imshow(fake_rgb[i])
        axes[i, 2].imshow(real_rgb[i])
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    save_dir = "results/epoch_visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{mode}_epoch_{actual_epoch:03d}.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results for a specific epoch")
    parser.add_argument("--mode", type=str, required=True, choices=['regression', 'classification'], help="Model mode")
    
    args = parser.parse_args()
    generate_for_epoch(args.mode)
