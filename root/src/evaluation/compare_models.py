import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import lpips
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from src.core.networks import Generator
from src.core.dataset import LabColourDataset
from src.core.utils import bins_to_ab_differentiable, lab_to_rgb_tensor, lab2rgb


#===============================================================================================================================
#=== VISUALIZE THE RESULTS OF THE MODEL AND COMPARE TO GROUND TRUTH ===
#===============================================================================================================================
def generate_comparison_plot():
    """ Load regression and classification models and generate 'results/final_comparison_random.pdf'. """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    image_size = 256
    num_bins = 100
    # Try to find dataset
    base_path = "data/coco"
    if not os.path.exists(os.path.join(base_path, "val")):
        base_path = "data/imagenet"
    
    val_path = os.path.join(base_path, "val")
    num_candidates = 2000
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    # --- Load Dataset ---
    if not os.path.exists(val_path):
        print(f"Error: Validation path {val_path} not found.")
        return

    full_val_dataset = LabColourDataset(val_path, transform=transform)
    
    # --- Load Models ---
    # Create models
    netG_reg = Generator(image_size=image_size, use_classification=False).to(device)
    netG_cls = Generator(image_size=image_size, use_classification=True, num_bins=num_bins).to(device)

    # Trained checkpoints
    reg_path = 'models/netG_regression_best.pth'
    cls_path = 'models/netG_classification_best.pth' 

    if os.path.exists(reg_path):
        netG_reg.load_state_dict(torch.load(reg_path, map_location=device)) # load model parameters
        print("Loaded regression model.")
    else:
        print(f"Warning: {reg_path} not found.")

    if os.path.exists(cls_path):
        netG_cls.load_state_dict(torch.load(cls_path, map_location=device)) # load model parameters
        print("Loaded classification model.")
    else:
        print(f"Warning: {cls_path} not found.")

    netG_reg.eval()
    netG_cls.eval()

    # --- Selection phase: find 5 diverse and perceptually accurate images (BUT 2000 candidates) ---
    print(f"Evaluating {num_candidates} images to find diverse, accurate results...")
    candidate_indices = random.sample(range(len(full_val_dataset)), min(num_candidates, len(full_val_dataset)))
    candidate_loader = DataLoader(Subset(full_val_dataset, candidate_indices), batch_size=16, shuffle=False)
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    candidate_data = []

    with torch.no_grad():
        for i, (L, target_ab) in enumerate(tqdm(candidate_loader, desc="Ranking images")):
            L, target_ab = L.to(device).float(), target_ab.to(device).float()
            real_rgb = lab_to_rgb_tensor(L, target_ab) * 2.0 - 1.0
            
            # 1. Colour stats
            mean_a = torch.mean(target_ab[:, 0, :, :], dim=(1, 2))
            mean_b = torch.mean(target_ab[:, 1, :, :], dim=(1, 2))
            std_ab = torch.std(target_ab, dim=(1, 2, 3))
            
            # 2. Perceptual distances
            fake_reg_ab = netG_reg(L)
            lpips_reg = loss_fn_vgg(lab_to_rgb_tensor(L, fake_reg_ab) * 2.0 - 1.0, real_rgb).view(-1)
            
            fake_cls_logits = netG_cls(L)
            fake_cls_ab = bins_to_ab_differentiable(fake_cls_logits, full_val_dataset, device, temperature=0.38)
            lpips_cls = loss_fn_vgg(lab_to_rgb_tensor(L, fake_cls_ab) * 2.0 - 1.0, real_rgb).view(-1)
            
            total_lpips = lpips_reg + lpips_cls
            
            for j in range(L.size(0)):
                idx = candidate_indices[i * 16 + j]
                # Filter out very desaturated (gray) images
                if std_ab[j] > 0.05:
                    candidate_data.append({
                        'idx': idx,
                        'lpips': total_lpips[j].item(),
                        'mean_a': mean_a[j].item(),
                        'mean_b': mean_b[j].item()
                    })

    # Greedy Selection for Diversity
    candidate_data.sort(key=lambda x: x['lpips'])
    selected_indices = []
    selected_colours = [] # List of (mean_a, mean_b)
    
    # Threshold for colour diversity (Euclidean distance in ab space)
    diversity_threshold = 0.15 

    for item in candidate_data:
        if len(selected_indices) >= 5:
            break
            
        is_diverse = True
        for colour in selected_colours:
            dist = np.sqrt((item['mean_a'] - colour[0])**2 + (item['mean_b'] - colour[1])**2)
            if dist < diversity_threshold:
                is_diverse = False
                break
        
        if is_diverse:
            selected_indices.append(item['idx'])
            selected_colours.append((item['mean_a'], item['mean_b']))

    if len(selected_indices) < 5:
        remaining_needed = 5 - len(selected_indices)
        for item in candidate_data:
            if item['idx'] not in selected_indices:
                selected_indices.append(item['idx'])
                if len(selected_indices) >= 5: break

    print(f"Diverse perceptual indices selected: {selected_indices}")
    best_indices = selected_indices[:5]

    # --- Final Inference for plotting ---
    subset_dataset = Subset(full_val_dataset, best_indices)
    loader = DataLoader(subset_dataset, batch_size=5, shuffle=False)
    
    data = next(iter(loader))
    L = data[0].to(device).float()
    target_ab = data[1].to(device).float()

    with torch.no_grad():
        fake_reg_ab = netG_reg(L)
        fake_cls_logits = netG_cls(L)
        fake_cls_ab = bins_to_ab_differentiable(fake_cls_logits, full_val_dataset, device, temperature=0.38)

    # Convert to RGB for plotting
    real_rgb = lab_to_rgb_tensor(L, target_ab).cpu().numpy().transpose(0, 2, 3, 1)
    reg_rgb = lab_to_rgb_tensor(L, fake_reg_ab).cpu().numpy().transpose(0, 2, 3, 1)
    cls_rgb = lab_to_rgb_tensor(L, fake_cls_ab).cpu().numpy().transpose(0, 2, 3, 1)
    L_np = L.cpu().numpy().squeeze(1)

    # --- Plotting Comparison ---
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(5, 4, figsize=(14, 18))
    
    fig.suptitle("Qualitative comparison: regression vs classification", fontsize=20, fontweight='bold', y=0.98)
    
    titles = ["Input (L)", "Regression", "Classification", "Ground truth"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=15, pad=15, fontweight='semibold')

    for i in range(5):
        axes[i, 0].imshow(L_np[i], cmap='gray')
        axes[i, 1].imshow(reg_rgb[i])
        axes[i, 2].imshow(cls_rgb[i])
        axes[i, 3].imshow(real_rgb[i])
        
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

    plt.subplots_adjust(wspace=0.1, hspace=0.15, top=0.92, bottom=0.02, left=0.05, right=0.95)
    os.makedirs("results", exist_ok=True)
    save_path = "results/final_comparison_random.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()
    
    

def plot_metrics_from_checkpoints():
    """ Load regression and classification models from checkpoints. Plot loss, PSNR, SSIM evolutions across epochs.
        Generate 'results/final_metrics_comparison.pdf'. """
    
    reg_ckpt = 'models/checkpoint_regression.pth'
    cls_ckpt = 'models/checkpoint_classification.pth'
    
    history_reg = None
    history_cls = None

    if os.path.exists(reg_ckpt):
        history_reg = torch.load(reg_ckpt, map_location='cpu').get('history')
    if os.path.exists(cls_ckpt):
        history_cls = torch.load(cls_ckpt, map_location='cpu').get('history')

    if not history_reg and not history_cls:
        print("No histories found in checkpoints.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Figure 1: Training Losses ---
    fig_loss, (ax_loss_reg, ax_loss_cls) = plt.subplots(1, 2, figsize=(16, 6))
    fig_loss.suptitle("Training progression: loss comparison", fontsize=18, fontweight='bold', y=1.02)

    # 1. Regression Loss Plot
    if history_reg and 'loss_G' in history_reg:
        ax_loss_reg.plot(history_reg['loss_G'], color='#2c7bb6', linewidth=2, label='Regression loss')
        ax_loss_reg.set_title('Regression', fontsize=14, pad=10)
        ax_loss_reg.set_ylabel('Loss', fontsize=12)
    else:
        ax_loss_reg.text(0.5, 0.5, 'No regression data', ha='center', va='center')
    
    # 2. Classification Loss Plot
    if history_cls:
        cls_loss_key = 'loss' if 'loss' in history_cls else 'loss_G'
        if cls_loss_key in history_cls:
            ax_loss_cls.plot(history_cls[cls_loss_key], color='#d7191c', linewidth=2, label='Classification loss')
            ax_loss_cls.set_title('Classification', fontsize=14, pad=10)
            ax_loss_cls.set_ylabel('Loss', fontsize=12)
    else:
        ax_loss_cls.text(0.5, 0.5, 'No classification data', ha='center', va='center')

    for ax in [ax_loss_reg, ax_loss_cls]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True, shadow=True)

    plt.tight_layout()
    loss_path = "results/final_losses_comparison.pdf"
    plt.savefig(loss_path, bbox_inches='tight')
    print(f"Losses plot saved to {loss_path}")
    plt.close(fig_loss)

    # --- Figure 2: Validation Metrics (PSNR/SSIM) ---
    fig_metrics, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(16, 6))
    fig_metrics.suptitle("Training progression: performance metrics", fontsize=18, fontweight='bold', y=1.02)

    # 3. PSNR Plot
    if history_reg and 'psnr' in history_reg:
        ax_psnr.plot(history_reg['psnr'], label='Regression', color='#2c7bb6', linewidth=2.5)
    if history_cls and 'psnr' in history_cls:
        ax_psnr.plot(history_cls['psnr'], label='Classification', color='#d7191c', linewidth=2.5)
    ax_psnr.set_title('Validation PSNR', fontsize=14, pad=10)
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=12)

    # 4. SSIM Plot
    if history_reg and 'ssim' in history_reg:
        ax_ssim.plot(history_reg['ssim'], label='Regression', color='#2c7bb6', linewidth=2.5)
    if history_cls and 'ssim' in history_cls:
        ax_ssim.plot(history_cls['ssim'], label='Classification', color='#d7191c', linewidth=2.5)
    ax_ssim.set_title('Validation SSIM', fontsize=14, pad=10)
    ax_ssim.set_ylabel('SSIM index', fontsize=12)

    for ax in [ax_psnr, ax_ssim]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True, shadow=True)

    plt.tight_layout()
    metrics_path = "results/final_metrics_comparison.pdf"
    plt.savefig(metrics_path, bbox_inches='tight')
    print(f"Metrics plot saved to {metrics_path}")
    plt.close(fig_metrics)
    
    

def plot_temperature_impact(num_samples=3):
    """CLASSIFICATION ONLY: Tune the temperature coefficient T and generates 'results/temperature_impact_analysis.pdf'

    Args:
        num_samples (int, optional): Number of image samples. Defaults to 3.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 256
    num_bins = 100
    base_path = "data/coco" 
    if not os.path.exists(os.path.join(base_path, "val")):
        base_path = "data/imagenet"
    val_path = os.path.join(base_path, "val")
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    if not os.path.exists(val_path):
        print(f"Error: Validation path {val_path} not found.")
        return

    full_val_dataset = LabColourDataset(val_path, transform=transform)
    random_indices = random.sample(range(len(full_val_dataset)), num_samples)
    subset_dataset = Subset(full_val_dataset, random_indices) # take a subset from the validation set
    loader = DataLoader(subset_dataset, batch_size=num_samples, shuffle=False)
    
    netG_cls = Generator(image_size=image_size, use_classification=True, num_bins=num_bins).to(device)
    cls_path = 'models/netG_classification_best.pth'

    if os.path.exists(cls_path):
        netG_cls.load_state_dict(torch.load(cls_path, map_location=device))
        print("Loaded classification model for temperature analysis.")
    else:
        print("Warning: Classification model not found for temperature analysis.")
        return

    netG_cls.eval()

    temperatures = [0.1, 0.25, 0.38, 0.6, 1.0]
    data = next(iter(loader))
    L = data[0].to(device).float()
    target_ab = data[1].to(device).float()
    
    with torch.no_grad():
        logits = netG_cls(L)
        
        results = []
        for T_val in temperatures:
            fake_ab = bins_to_ab_differentiable(logits, full_val_dataset, device, temperature=T_val)
            rgb = lab_to_rgb_tensor(L, fake_ab).cpu().numpy().transpose(0, 2, 3, 1)
            results.append(rgb)

    real_rgb = lab_to_rgb_tensor(L, target_ab).cpu().numpy().transpose(0, 2, 3, 1)
    L_np = L.cpu().numpy().squeeze(1)

    # Plotting
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(num_samples, len(temperatures) + 2, figsize=(20, 4 * num_samples))
    
    fig.suptitle("Impact of temperature (T) on classification results", fontsize=20, fontweight='bold', y=1.02)

    col_titles = ["Input (L)", "Ground truth"] + [f"T = {t}" for t in temperatures]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14, pad=10, fontweight='semibold')

    for i in range(num_samples):
        axes[i, 0].imshow(L_np[i], cmap='gray')
        axes[i, 1].imshow(real_rgb[i])
        
        for j, T_idx in enumerate(range(len(temperatures))):
            axes[i, j + 2].imshow(results[j][i])
            
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    save_path = "results/temperature_impact_analysis.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Temperature impact analysis saved to {save_path}")
    plt.close()
    

if __name__ == "__main__":
    generate_comparison_plot()
    plot_metrics_from_checkpoints()
    plot_temperature_impact()
