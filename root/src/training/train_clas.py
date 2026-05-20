"""
Classification-based colourisation.
Prediction of colour bin indices based on Zhang et al. (2016).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import numpy as np
import os

from src.core.networks import Generator
from src.core.dataset import LabColourDataset, get_class_weights
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.core.utils import bins_to_ab_differentiable, lab_to_rgb_tensor, save_visualise_results


#============================================================================================================
#=== FUNCTION FOR TRAINING THE MODEL - CLASSIFICATION ===
#============================================================================================================
def train_clas_loop(base_path, transform, device,
                     batch_size, num_epochs, num_bins, lr, beta1, **kwargs):
    """
    Classification training loop.
    Pure CrossEntropy — no GAN discriminator.
    
    Store graphical results every 5 epochs at 'results/classification/'
    
    Args:
        base_path: Directory containing /train and /val.
        transform: PyTorch transformation (resizing, ...).
        device: 'cuda' or 'cpu'.
        batch_size: Size of the batches.
        num_epochs: Number of epochs.
        lr: Learning rate.
        beta1: Hyperparameter for Adam optimization.
        kwargs: Optional parameters in the form of a dictionary.
        
    Returns:
        netG (Generator): Trained model.
        history (dict): Dictionary containing the history of important metrics.
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    image_size = kwargs.get('image_size', 256)

    print(f"\n--- Starting Classification Training ---")
    print(f"Target epochs: {num_epochs} | Num bins: {num_bins}")

    # --- Datasets and loaders ---
    train_dataset = LabColourDataset(
        os.path.join(base_path, "train"), transform=transform,
        mode='classification', num_bins=num_bins
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True, persistent_workers=True
    )

    val_loader = None
    val_path = os.path.join(base_path, "val")
    if os.path.exists(val_path):
        val_dataset = LabColourDataset(
            val_path, transform=transform, mode='classification', num_bins=num_bins
        )
        val_sampler = RandomSampler(val_dataset, replacement=False,
                                    num_samples=min(500, len(val_dataset)))
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    # --- Network ---
    netG = Generator(
        image_size=image_size, use_classification=True, num_bins=num_bins
    ).to(device)

    # --- Optimizer and scheduler ---
    optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=num_epochs, eta_min=1e-6)

    # --- Loss — CrossEntropy with class rebalancing ---
    # Class rebalancing is now automated
    weights = get_class_weights(train_dataset, num_bins).to(device) # use 2000 image samples by default
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Using automated class rebalancing weights.")

    # --- Metrics ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    scaler = torch.amp.GradScaler('cuda')

    # --- State ---
    history = {'loss': [], 'psnr': [], 'ssim': []}
    start_epoch = 0
    best_ssim = 0.0

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Resume ---
    # Resumption is now automated
    checkpoint_path = 'models/checkpoint_classification.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            netG.load_state_dict(ckpt['netG_state_dict'])
            optG.load_state_dict(ckpt['optG_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            history = ckpt.get('history', history)
            start_epoch = ckpt['epoch'] + 1
            best_ssim = ckpt.get('best_ssim', 0.0)
            print(f"Resuming from epoch {start_epoch} | Best SSIM: {best_ssim:.4f}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}). Starting from scratch.")
            start_epoch = 0

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs):
        netG.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for data in pbar:
            L = data[0].to(device, non_blocking=True).float()
            target_bins = data[1].to(device, non_blocking=True).long()

            optG.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = netG(L)
                loss = criterion(logits, target_bins)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            scaler.step(optG)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        scheduler.step()

        # --- Validation ---
        netG.eval()
        eval_loader = val_loader if val_loader else [data]
        val_psnr, val_ssim, num_val = 0.0, 0.0, 0

        with torch.no_grad():
            for v_data in eval_loader:
                v_L = v_data[0].to(device, non_blocking=True).float()
                v_target_ab = v_data[2].to(device, non_blocking=True).float()

                v_fake_ab = bins_to_ab_differentiable(
                    netG(v_L), train_dataset, device, temperature=0.38
                )

                real_rgb = lab_to_rgb_tensor(v_L, v_target_ab)
                fake_rgb = lab_to_rgb_tensor(v_L, v_fake_ab)

                val_psnr += psnr_metric(fake_rgb, real_rgb).item()
                val_ssim += ssim_metric(fake_rgb, real_rgb).item()
                num_val  += 1

        val_psnr /= num_val
        val_ssim /= num_val
        epoch_loss = running_loss / len(train_loader)

        history['loss'].append(epoch_loss)
        history['psnr'].append(val_psnr)
        history['ssim'].append(val_ssim)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
        )

        recent_ssim = float(np.mean(history['ssim'][-3:])) if len(history['ssim']) >= 3 else val_ssim

        if recent_ssim > best_ssim:
            best_ssim = recent_ssim
            torch.save(netG.state_dict(), 'models/netG_classification_best.pth')
            print(f"  → New best SSIM: {best_ssim:.4f} — best model saved.")

        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_path = f'results/classification/classification_epoch_{epoch+1:03d}.png'
            v_data      = next(iter(eval_loader))
            v_L         = v_data[0].to(device).float()
            v_target_ab = v_data[2].to(device).float()

            with torch.no_grad():
                v_fake_ab = bins_to_ab_differentiable(
                    netG(v_L), train_dataset, device, temperature=0.38
                )

            save_visualise_results(
                v_L, v_fake_ab, v_target_ab, 'classification', train_dataset, device,
                save_path=save_path, epoch=epoch+1, stats={'psnr': val_psnr, 'ssim': val_ssim}
            )

        # Save the states of the model as a dictionary
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'optG_state_dict': optG.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_ssim': best_ssim,
        }, checkpoint_path)

    return netG, history
