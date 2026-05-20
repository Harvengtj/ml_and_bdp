import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr

# New metrics
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure


#===================================================================================================================================
#=== USEFUL FUNCTIONS (ADAPTED FOR TENSORS) ===
#===================================================================================================================================
def lab_to_rgb_tensor(L, ab):
    """Converts LAB tensors to RGB directly on the GPU (differentiable).
   
    Args:
        L (tensor): [-1, 1] -> shape [B, 1, H, W].
        ab (tensor): [-1, 1] -> shape [B, 2, H, W].

    Returns:
        (tensor): RGB [0, 1] -> shape [B, 3, H, W].
    """
    # Unnormalize LAB
    L_unnorm = (L + 1) * 50 # [-1, 1] --> [0, 100] 
    ab_unnorm = ab * 128 # [-1, 1] --> [-128, 128] 
    lab = torch.cat([L_unnorm, ab_unnorm], dim=1)

    # LAB -> XYZ
    y = (lab[:, 0:1] + 16) / 116
    x = (lab[:, 1:2] / 500) + y
    z = y - (lab[:, 2:3] / 200)

    xyz = torch.cat([x, y, z], dim=1)
    mask = xyz > 0.2068966
    xyz = torch.where(mask, xyz**3, (xyz - 16/116) / 7.787)

    # Reference White (D65)
    xyz[:, 0:1] = xyz[:, 0:1] * 0.95047
    xyz[:, 1:2] = xyz[:, 1:2] * 1.00000
    xyz[:, 2:3] = xyz[:, 2:3] * 1.08883

    # XYZ -> RGB
    r = xyz[:, 0:1] * 3.2404542 - xyz[:, 1:2] * 1.5371385 - xyz[:, 2:3] * 0.4985314
    g = -xyz[:, 0:1] * 0.9692660 + xyz[:, 1:2] * 1.8760108 + xyz[:, 2:3] * 0.0415560
    b = xyz[:, 0:1] * 0.0556434 - xyz[:, 1:2] * 0.2040259 + xyz[:, 2:3] * 1.0572252

    rgb = torch.cat([r, g, b], dim=1)
    mask = rgb > 0.0031308
    rgb = torch.where(mask, 1.055 * (rgb ** (1/2.4)) - 0.055, 12.92 * rgb)

    return torch.clamp(rgb, 0, 1)

def ab_to_bin_tensor(ab, num_bins):
    """Calculates target bins on GPU directly from the AB tensor.  

    Args:
        ab (tensor): [B, 2, H, W].
        num_bins (int): Number of bins.

    Returns:
        (tensor): [B, N, H, W].
    """
    
    grid_size = int(np.sqrt(num_bins))
    
    a = ab[:, 0:1]
    a = (a + 1) / 2
    a = a * grid_size
    a = torch.clamp(a, 0, grid_size - 1)
    a = a.long()
    
    b = ab[:, 1:2]
    b = (b + 1) / 2
    b = b * grid_size
    b = torch.clamp(b, 0, grid_size - 1)
    b = b.long()
   
    return (a * grid_size + b).squeeze(1) # 1D tensor

def bins_to_ab_differentiable(logits, ds_obj, device, temperature=0.5):
    """Calculates a and b from bins (like in paper 'Colorful Image Colorization').
        logits [B, N, H, W] --> for EACH pixel, probas to belong to a certain bin

    Args:
        logits (float): Raw values from the model output.
        ds_obj (LabColourDataset): Dataset.
        device (str): 'cpu' or 'gpu'.
        temperature (float, optional): Used in the softmax function to tune the color distribution (if big, distribution unchanged, else peaked). Defaults to 0.5.

    Returns:
        (tensor): Tensor with ab components.
    """
    with torch.amp.autocast('cuda', enabled=False):
        probs = F.softmax(logits.float() / temperature, dim=1) # ex: [prob0, prob1, ..., prob99]
        
        num_bins = logits.shape[1]
        grid_size = int(np.sqrt(num_bins))
        
        bin_indices = torch.arange(num_bins, device=device) # ex: [bin0, bin1, ..., bin99]
        
        # Recover a and b values
        a = bin_indices // grid_size
        a = a.float()
        a = a / grid_size
        a = a * 2 - 1
        a_centres = a + (1 / grid_size) # ex: [a_0, a_1, ..., a_99]
        
        b = bin_indices % grid_size
        b = b.float()
        b = b / grid_size
        b = b * 2 - 1
        b_centres = b + (1 / grid_size) # ex: [b_0, b_1, ..., b_99]
        
        
        # Einsum is the most memory-efficient way to do weighted sum
        a_out = torch.einsum('bchw,c->bhw', probs, a_centres).unsqueeze(1) # sum(prob0*a_0 + ... + prob99*a_99)
        b_out = torch.einsum('bchw,c->bhw', probs, b_centres).unsqueeze(1) # sum(prob0*b_0 + ... + prob99*b_99)
        
        return torch.cat([a_out, b_out], dim=1)



#===================================================================================================================================
#=== PLOTS, COMPARISONS ===
#===================================================================================================================================
def compare_colourisations(netG_reg, netG_cls, val_loader, ds_obj, device, num_samples=10):
    """Full evaluation: Calculates SSIM, LPIPS, and Accuracy across the whole dataset, 
       then generates the visualisation grid. Prints the results in the terminal and
       shows images as |GT|L|reg|cls| in "results/compare_colourisations.png".

    Args:
        netG_reg (Generator): Regression model.
        netG_cls (Generator): Classification model.
        val_loader (DataLoader): Validation set.
        ds_obj (LabColourDataset): Dataset.
        device (str): 'cpu' or 'gpu'.
        num_samples (int, optional): Number of image samples. Defaults to 10.
    """
    print("\n--- Launching full evaluation (LPIPS, SSIM, Accuracy) ---")
    plt.close('all')
    
    # Deactivate: dropout, batchnorm training mode
    netG_reg.eval()
    netG_cls.eval()
    
    # Metrics initialisation
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # perceptual measurement
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device) # structure, contrast, luminance measurement 
    
    # Store results
    metrics = {'reg_lpips': [], 'reg_ssim': [], 'cls_lpips': [], 'cls_ssim': [], 'top1': [], 'top5': []}
    L_list, target_list, reg_list, cls_list = [], [], [], []
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            L = data[0].to(device).float()
            
            # Extract targets (ground truth)
            if len(data) == 3: 
                target_ab = data[2].to(device).float() # classification (L, bins, ab)
            else:
                target_ab = data[1].to(device).float() # regression (L, ab)

            # Target bins for Accuracy
            target_bins = ab_to_bin_tensor(target_ab, ds_obj.num_bins) # [N, H, W]

            # Inference results
            out_reg_ab = netG_reg(L) # [B, 2, H, W]
            out_cls_logits = netG_cls(L)
            out_cls_ab = bins_to_ab_differentiable(out_cls_logits, ds_obj, device) # [B, N, H, W]
            
            # --- METRICS CALCULATION ---
            # 1. Top-1 and Top-5 Accuracy (Classification)
            _, pred_top1 = out_cls_logits.topk(1, dim=1) # idx, highest value => [B,1,H,W]
            metrics['top1'].append((pred_top1.squeeze(1) == target_bins).float().mean().item()) # check if target_bins = most likely predicted_bins --> ex: [True,False,True,True] => [1.0,0.0,1.0,1.0] => mean = 0.75 => convert tensor to float Python
            
            # Each pixel contains: 1 if the actual class is in the top 5, 0 otherwise
            _, pred_top5 = out_cls_logits.topk(5, dim=1) # idx, highest value => [B,5,H,W]
            correct_top5 = (pred_top5 == target_bins.unsqueeze(1)).float().sum(dim=1) # unsqueeze [N, H, W] => [B,1,H,W] to compare with [B,5,H,W], then ex: [False, False, False, True, False] => [0.0,0.0,0.0,1.0,0.0] => sum = 1.0
            metrics['top5'].append(correct_top5.mean().item()) # mean over all the pixels

            # 2. RGB conversion and normalization [-1, 1] for LPIPS/SSIM
            real_rgb = lab_to_rgb_tensor(L, target_ab) * 2 - 1
            reg_rgb = lab_to_rgb_tensor(L, out_reg_ab) * 2 - 1
            cls_rgb = lab_to_rgb_tensor(L, out_cls_ab) * 2 - 1

            # 3. LPIPS and SSIM
            metrics['reg_lpips'].append(loss_fn_vgg(reg_rgb, real_rgb).mean().item()) # smallest = best 
            metrics['reg_ssim'].append(ssim_fn(reg_rgb, real_rgb).item()) # closest to 1 = best
            
            metrics['cls_lpips'].append(loss_fn_vgg(cls_rgb, real_rgb).mean().item())
            metrics['cls_ssim'].append(ssim_fn(cls_rgb, real_rgb).item())

            # --- STORAGE FOR VISUALISATION ---
            # Avoid to store too many images
            if len(np.concatenate(L_list) if L_list else []) < num_samples:
                # [B,C,H,W] --> [B,H,W,C]
                L_list.append(L.cpu().numpy().transpose(0, 2, 3, 1))
                target_list.append(target_ab.cpu().numpy().transpose(0, 2, 3, 1)) 
                reg_list.append(out_reg_ab.cpu().numpy().transpose(0, 2, 3, 1))
                cls_list.append(out_cls_ab.cpu().numpy().transpose(0, 2, 3, 1))
            
            # Progression display
            if (i+1) % 10 == 0:
                print(f"Evaluation Batch [{i+1}/{len(val_loader)}]")

    # Terminal summary
    print("\n" + "="*40)
    print("EVALUATION RESULTS (Averages)")
    print("="*40)
    print(f"--- REGRESSION Model ---")
    print(f"SSIM  : {np.mean(metrics['reg_ssim']):.4f} (The closer to 1, the better)") # mean on all images
    print(f"LPIPS : {np.mean(metrics['reg_lpips']):.4f} (The closer to 0, the better)")
    print(f"\n--- CLASSIFICATION Model ---")
    print(f"SSIM  : {np.mean(metrics['cls_ssim']):.4f}")
    print(f"LPIPS : {np.mean(metrics['cls_lpips']):.4f}")
    print(f"Top-1 Acc : {np.mean(metrics['top1'])*100:.2f}%")
    print(f"Top-5 Acc : {np.mean(metrics['top5'])*100:.2f}%")
    print("="*40 + "\n")

    # --- GRID GENERATION (unchanged) ---
    L_all = np.concatenate(L_list)[:num_samples]
    target_all = np.concatenate(target_list)[:num_samples]
    reg_all = np.concatenate(reg_list)[:num_samples]
    cls_all = np.concatenate(cls_list)[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples)) # 4 columns: groundtruth, grayscale, reg, cls 
    
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        L_chan = (L_all[i] + 1.0) * 50.0 # unnormalize: [-1,1] -> [0,100], shape: [H,W,1]
        
        # ab: [-1,1] -> [-128,127], ab shape: [H,W,2] -> concatenation: [H,W,3] (we merge L and ab)
        img_gt = lab2rgb(np.clip(np.concatenate([L_chan, target_all[i] * 128.0], axis=-1), [0,-128,-128], [100,128,128]))
        img_reg = lab2rgb(np.clip(np.concatenate([L_chan, reg_all[i] * 128.0], axis=-1), [0,-128,-128], [100,128,128]))
        img_cls = lab2rgb(np.clip(np.concatenate([L_chan, cls_all[i] * 128.0], axis=-1), [0,-128,-128], [100,128,128]))
        
        # Show images 
        axes[i, 0].imshow(img_gt)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 1].imshow(L_all[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Input (L)")
        axes[i, 2].imshow(img_reg)
        axes[i, 2].set_title("Regression")
        axes[i, 3].imshow(img_cls)
        axes[i, 3].set_title("Classification")
        for ax in axes[i]: ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("results/compare_colourisations.png", bbox_inches="tight", dpi=150)
    plt.close()
    

def plot_training_history(hist_reg, hist_cls):
    """Plots training losses and metrics (PSNR, SSIM) for both models. Save the results in "results/history_comparison.png"
    
       Args:
        hist_reg (dict): history = {'loss_G': [], 'loss_D': [], 'psnr': [], 'ssim': []} from train_reg.py
        hist_cls (dict): history = {'loss': [], 'psnr': [], 'ssim': []} from train_cls.py
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Plot
    if 'loss_G' in hist_reg:
        axes[0].plot(hist_reg['loss_G'], label='Reg G Loss', color='blue', linestyle='--')
    if 'loss_D' in hist_reg:
        axes[0].plot(hist_reg['loss_D'], label='Reg D Loss', color='blue')
    if 'loss' in hist_cls:
        axes[0].plot(hist_cls['loss'], label='Cls Loss', color='red')
        
    axes[0].set_title('Training Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # 2. PSNR Plot
    if 'psnr' in hist_reg:
        axes[1].plot(hist_reg['psnr'], label='Reg PSNR', color='blue')
    if 'psnr' in hist_cls:
        axes[1].plot(hist_cls['psnr'], label='Cls PSNR', color='red')
    axes[1].set_title('Validation PSNR')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('dB')
    axes[1].legend()
    
    # 3. SSIM Plot
    if 'ssim' in hist_reg:
        axes[2].plot(hist_reg['ssim'], label='Reg SSIM', color='blue')
    if 'ssim' in hist_cls:
        axes[2].plot(hist_cls['ssim'], label='Cls SSIM', color='red')
    axes[2].set_title('Validation SSIM')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Index')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("results/history_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()


# Used in train_reg.py and train_cls.py every 5 epochs to create a .pdf to see the evolution across the epochs
def save_visualise_results(L, fake_out, target, mode, ds_obj, device, save_path=None, epoch=None, stats=None):
    """Saves a comparison of results to disk and displays PSNR for each sample.
       Used in train_reg.py or train_cls.py every 5 epochs to create .PNG images in the save_path to observe images' evolution across the epochs. 
       Save also the figure in "results/visual_comparison.png".

    Args:
        L (tensor): Grayscale image.
        fake_out (tensor): ab components of generated image.
        target (tensor): GT image.
        mode (str): 'classification' or 'regression'.
        ds_obj (LabColourDataset): Dataset.
        device (str): 'cpu' or 'gpu'.
        save_path (str, optional): Path where results are stored. Defaults to None.
        epoch (int, optional): Number of epochs. Defaults to None.
        stats (dict, optional): Dictionary with SSIM and PSNR values. Defaults to None.
    """
    
    from skimage.metrics import peak_signal_noise_ratio as psnr_sq
    
    real_rgb = lab_to_rgb_tensor(L, target).cpu().numpy().transpose(0, 2, 3, 1)
    fake_rgb = lab_to_rgb_tensor(L, fake_out).cpu().numpy().transpose(0, 2, 3, 1)
    L_np = L.cpu().numpy().squeeze(1)
    
    num_imgs = min(L.shape[0], 4)
    # Tighter: reduction of figsize and manual adjustment of spacing
    fig, axes = plt.subplots(num_imgs, 3, figsize=(10, 3.2 * num_imgs))
    
    if num_imgs == 1:
        axes = axes.reshape(1, 3)

    # Main title and subtitle
    method_name = mode.capitalize()
    if mode == 'classification': 
        method_name = 'Classification'
    
    if epoch is not None:
        fig.suptitle(f"Method: {method_name} | Epoch: {epoch}", fontsize=16, fontweight='bold', y=0.98)
    
    if stats:
        # Keeping only a few key stats
        stat_str = f"Avg PSNR: {stats.get('psnr', 0):.2f} | Avg SSIM: {stats.get('ssim', 0):.4f}"
        fig.text(0.5, 0.94, stat_str, ha='center', fontsize=11, style='italic')

    for i in range(num_imgs):
        axes[i, 0].imshow(L_np[i], cmap='gray')
        if i == 0: axes[i, 0].set_title("Input", fontsize=12, pad=8)
        
        axes[i, 1].imshow(fake_rgb[i])
        if i == 0: axes[i, 1].set_title(method_name, fontsize=12, pad=8)
        
        axes[i, 2].imshow(real_rgb[i])
        if i == 0: axes[i, 2].set_title("Ground Truth", fontsize=12, pad=8)
        
        for ax in axes[i]: 
            ax.axis('off')
    
    # Tightens the images
    plt.subplots_adjust(wspace=0.02, hspace=0.05, top=0.91, bottom=0.02, left=0.02, right=0.98)
    
    # Save BEFORE showing (important for some environments)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        
    plt.savefig("results/visual_comparison.png", bbox_inches="tight", dpi=150)  # necessary ?
    plt.close()