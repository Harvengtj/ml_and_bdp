import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr

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
    Visualises the input, predicted, and ground truth images with PSNR metrics.
    """
    plt.close('all') # Clear memory
    L_np = L.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    if mode == 'classification':
        fake_ab_tensor = bins_to_ab_differentiable(fake_out, ds_obj, device)
        fake_ab = fake_ab_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
        if target.ndim == 3: # bins
            true_ab = ds_obj.bin_to_ab(target.cpu().numpy())
        else: # ab tensor
            true_ab = target.cpu().numpy().transpose(0, 2, 3, 1)
    else:
        fake_ab = fake_out.detach().cpu().numpy().transpose(0, 2, 3, 1)
        true_ab = target.cpu().numpy().transpose(0, 2, 3, 1)
    
    num_images = min(L_np.shape[0], 3)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    if num_images == 1: axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):
        # PSNR Calculation
        current_psnr = psnr(true_ab[i], fake_ab[i], data_range=2.0)
        
        L_chan = (L_np[i] + 1.0) * 50.0
        
        # Predicted
        lab_pred = np.concatenate([L_chan, fake_ab[i] * 128.0], axis=-1)
        rgb_pred = lab2rgb(np.clip(lab_pred, [0, -128, -128], [100, 128, 128]).astype(np.float64))
        
        # Real
        lab_real = np.concatenate([L_chan, true_ab[i] * 128.0], axis=-1)
        rgb_real = lab2rgb(np.clip(lab_real, [0, -128, -128], [100, 128, 128]).astype(np.float64))
        
        axes[i, 0].imshow(L_np[i].squeeze(), cmap='gray')
        axes[i, 0].set_title("Input (L)")
        axes[i, 1].imshow(rgb_pred)
        axes[i, 1].set_title(f"Predicted ({mode})\nPSNR: {current_psnr:.2f} dB")
        axes[i, 2].imshow(rgb_real)
        axes[i, 2].set_title("Ground Truth")
        
        for ax in axes[i]: ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def compare_colourisations(netG_reg, netG_cls, val_loader, ds_obj, device, num_samples=10):
    """
    Comparison grid with higher quality rendering and PSNR metrics.
    """
    plt.close('all')
    netG_reg.eval()
    netG_cls.eval()
    
    L_list, target_list, reg_list, cls_list = [], [], [], []
    
    with torch.no_grad():
        for data in val_loader:
            L = data[0].to(device).float()
            target = data[1]
            
            out_reg = netG_reg(L).detach().cpu().numpy().transpose(0, 2, 3, 1)
            out_cls_logits = netG_cls(L)
            out_cls = bins_to_ab_differentiable(out_cls_logits, ds_obj, device).cpu().numpy().transpose(0, 2, 3, 1)
            
            L_np = L.cpu().numpy().transpose(0, 2, 3, 1)
            if target.ndim == 3: # classification bins
                target_ab = ds_obj.bin_to_ab(target.numpy())
            else: # regression ab channels
                target_ab = target.numpy().transpose(0, 2, 3, 1)
                
            L_list.append(L_np)
            target_list.append(target_ab)
            reg_list.append(out_reg)
            cls_list.append(out_cls)
            if len(np.concatenate(L_list)) >= num_samples: break
                
    L_all, target_all = np.concatenate(L_list)[:num_samples], np.concatenate(target_list)[:num_samples]
    reg_all, cls_all = np.concatenate(reg_list)[:num_samples], np.concatenate(cls_list)[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1: axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        L_chan = (L_all[i] + 1.0) * 50.0
        psnr_reg = psnr(target_all[i], reg_all[i], data_range=2.0)
        psnr_cls = psnr(target_all[i], cls_all[i], data_range=2.0)
        
        # Renderings
        img_gt = lab2rgb(np.clip(np.concatenate([L_chan, target_all[i] * 128.0], axis=-1), [0,-128,-128], [100,128,128]))
        img_reg = lab2rgb(np.clip(np.concatenate([L_chan, reg_all[i] * 128.0], axis=-1), [0,-128,-128], [100,128,128]))
        img_cls = lab2rgb(np.clip(np.concatenate([L_chan, cls_all[i] * 128.0], axis=-1), [0,-128,-128], [100,128,128]))
        
        axes[i, 0].imshow(img_gt)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 1].imshow(L_all[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Input (L)")
        axes[i, 2].imshow(img_reg)
        axes[i, 2].set_title(f"Regression\nPSNR: {psnr_reg:.2f}")
        axes[i, 3].imshow(img_cls)
        axes[i, 3].set_title(f"Classification\nPSNR: {psnr_cls:.2f}")
        for ax in axes[i]: ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_training_history(hist_reg, hist_cls):
    """
    Plots training loss and PSNR metrics with professional styling.
    """
    plt.close('all')
    # Try to use a nice style, fallback to default if not available
    try:
        plt.style.use('seaborn-v0_8-muted')
    except:
        try:
            plt.style.use('ggplot')
        except:
            pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Loss Plot
    if hist_reg and 'loss_G' in hist_reg:
        epochs = range(1, len(hist_reg['loss_G']) + 1)
        ax1.plot(epochs, hist_reg['loss_G'], label='Regression (L1)', color='#3498db', linewidth=2, marker='o', markersize=4, alpha=0.8)
    if hist_cls and 'loss_G' in hist_cls:
        epochs = range(1, len(hist_cls['loss_G']) + 1)
        ax1.plot(epochs, hist_cls['loss_G'], label='Classification (CE)', color='#e74c3c', linewidth=2, marker='s', markersize=4, alpha=0.8)
    
    ax1.set_title('Generator Training Loss', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, shadow=True)
    
    # 2. PSNR Plot
    if hist_reg and 'psnr' in hist_reg:
        epochs = range(1, len(hist_reg['psnr']) + 1)
        ax2.plot(epochs, hist_reg['psnr'], label='Regression (L1)', color='#2ecc71', linewidth=2, marker='o', markersize=4, alpha=0.8)
    if hist_cls and 'psnr' in hist_cls:
        epochs = range(1, len(hist_cls['psnr']) + 1)
        ax2.plot(epochs, hist_cls['psnr'], label='Classification (CE)', color='#f1c40f', linewidth=2, marker='s', markersize=4, alpha=0.8)
        
    ax2.set_title('Validation PSNR (Quality Index)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.show()
