import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

def bins_to_ab_differentiable(logits, ds_obj, device):
    """
    Converts bin logits to AB values in a differentiable way using soft-argmax.
    """
    temperature = 0.05
    probs = F.softmax(logits / temperature, dim=1)
    
    num_bins = logits.shape[1]
    grid_size = int(np.sqrt(num_bins))
    
    # Pre-calculate bin centres as tensors
    bin_indices = torch.arange(num_bins, device=device)
    a_centers = ((bin_indices // grid_size).float() / grid_size) * 2.0 - 1.0 + (1.0 / grid_size)
    b_centers = ((bin_indices % grid_size).float() / grid_size) * 2.0 - 1.0 + (1.0 / grid_size)
    
    # Reshape for broadcasting: [1, NUM_BINS, 1, 1]
    a_centers = a_centers.view(1, num_bins, 1, 1)
    b_centers = b_centers.view(1, num_bins, 1, 1)
    
    # Expectation over bins
    a_out = torch.sum(probs * a_centers, dim=1, keepdim=True)
    b_out = torch.sum(probs * b_centers, dim=1, keepdim=True)
    
    return torch.cat([a_out, b_out], dim=1)

def visualize_results(L, fake_out, target, mode, ds_obj, device):
    """
    Visualises the input, predicted, and ground truth images.
    """
    L_np = L.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    if mode == 'classification':
        fake_bins = torch.argmax(fake_out, dim=1).cpu().numpy()
        fake_ab = ds_obj.bin_to_ab(fake_bins)
        true_ab = ds_obj.bin_to_ab(target.cpu().numpy())
    else:
        fake_ab = fake_out.detach().cpu().numpy().transpose(0, 2, 3, 1)
        true_ab = target.cpu().numpy().transpose(0, 2, 3, 1)
    
    num_images = min(L_np.shape[0], 3)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):
        L_chan = (L_np[i] + 1.0) * 50.0
        ab_pred = fake_ab[i] * 128.0
        lab_pred = np.concatenate([L_chan, ab_pred], axis=-1)
        lab_pred[:,:,0] = np.clip(lab_pred[:,:,0], 0, 100)
        rgb_pred = lab2rgb(lab_pred.astype(np.float64))
        
        ab_real = true_ab[i] * 128.0
        lab_real = np.concatenate([L_chan, ab_real], axis=-1)
        lab_real[:,:,0] = np.clip(lab_real[:,:,0], 0, 100)
        rgb_real = lab2rgb(lab_real.astype(np.float64))
        
        axes[i, 0].imshow(L_np[i].squeeze(), cmap='gray')
        axes[i, 0].set_title("Input (L)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(rgb_pred)
        axes[i, 1].set_title(f"Predicted ({mode})")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(rgb_real)
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

def compare_colourisations(netG_reg, netG_cls, val_loader, ds_obj, device, num_samples=10):
    """
    Displays a comparison grid: Ground Truth, Input L, Regression Result, Classification Result.
    """
    netG_reg.eval()
    netG_cls.eval()
    
    L_list, target_list, reg_list, cls_list = [], [], [], []
    
    with torch.no_grad():
        for L, target in val_loader:
            L = L.to(device).float()
            
            # Regression prediction
            out_reg = netG_reg(L).detach().cpu().numpy().transpose(0, 2, 3, 1)
            
            # Classification prediction
            out_cls_logits = netG_cls(L)
            out_cls_bins = torch.argmax(out_cls_logits, dim=1).cpu().numpy()
            out_cls = ds_obj.bin_to_ab(out_cls_bins)
            
            # Handle target shape for both modes
            L_np = L.cpu().numpy().transpose(0, 2, 3, 1)
            if target.ndim == 3: # classification bins
                target_ab = ds_obj.bin_to_ab(target.numpy())
            else: # regression ab channels
                target_ab = target.numpy().transpose(0, 2, 3, 1)
                
            L_list.append(L_np)
            target_list.append(target_ab)
            reg_list.append(out_reg)
            cls_list.append(out_cls)
            
            if len(np.concatenate(L_list)) >= num_samples:
                break
                
    L_all = np.concatenate(L_list)[:num_samples]
    target_all = np.concatenate(target_list)[:num_samples]
    reg_all = np.concatenate(reg_list)[:num_samples]
    cls_all = np.concatenate(cls_list)[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        L_chan = (L_all[i] + 1.0) * 50.0
        
        # 1. Ground Truth
        lab_gt = np.concatenate([L_chan, target_all[i] * 128.0], axis=-1)
        rgb_gt = lab2rgb(np.clip(lab_gt, [0,-128,-128], [100,128,128]).astype(np.float64))
        
        # 2. Input L
        input_l = L_all[i].squeeze()
        
        # 3. Regression
        lab_reg = np.concatenate([L_chan, reg_all[i] * 128.0], axis=-1)
        rgb_reg = lab2rgb(np.clip(lab_reg, [0,-128,-128], [100,128,128]).astype(np.float64))
        
        # 4. Classification
        lab_cls = np.concatenate([L_chan, cls_all[i] * 128.0], axis=-1)
        rgb_cls = lab2rgb(np.clip(lab_cls, [0,-128,-128], [100,128,128]).astype(np.float64))
        
        axes[i, 0].imshow(rgb_gt)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 1].imshow(input_l, cmap='gray')
        axes[i, 1].set_title("Input (L)")
        axes[i, 2].imshow(rgb_reg)
        axes[i, 2].set_title("Regression")
        axes[i, 3].imshow(rgb_cls)
        axes[i, 3].set_title("Classification")
        
        for ax in axes[i]: ax.axis('off')
        
    plt.tight_layout()
    plt.show()
