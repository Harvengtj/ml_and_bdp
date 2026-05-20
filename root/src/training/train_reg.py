import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import numpy as np
import os

from src.core.networks import Generator, Discriminator
from src.core.dataset import LabColourDataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.core.utils import lab_to_rgb_tensor, save_visualise_results


#============================================================================================================
#=== FUNCTION FOR TRAINING THE MODEL - REGRESSION ===
#============================================================================================================
def train_reg_loop(base_path, transform, device,
                   batch_size, num_epochs, lr, beta1, **kwargs):
    """
    GAN training loop for Image Colorization (Regression mode).
    Uses L1 loss + Adversarial loss.
    
    Store graphical results every 5 epochs at 'results/regression/'
    
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
        netG (Generator): Trained generator model.
        netD (Discriminator): Trained discriminator model.
        history (dict): Dictionary containing the history of important metrics.
    """
    
    # Clear memory before training
    import gc
    gc.collect() # forces the cleaning of RAM by removing unused objects
    torch.cuda.empty_cache() # empty unused cuda cache

    # If optional parameters provided
    image_size = kwargs.get('image_size', 256)
    lambda_l1  = kwargs.get('lambda_l1', 100)

    # --- Samples per epoch limit ---
    max_samples = kwargs.get('max_samples_per_epoch', None) # ex: len(dataset) = 200000 and max_samples = 5000

    print(f"\n--- Starting Regression GAN Training ---")
    print(f"Target epochs: {num_epochs}")

    # --- Datasets and loaders ---
    # Fetch training dataset
    train_dataset = LabColourDataset(
        os.path.join(base_path, "train"), transform=transform, mode='regression'
    )
    
    # Check if the specified maximum number of samples PER EPOCH is smaller than the size of the training dataset 
    if max_samples and max_samples < len(train_dataset):
        # Take a subset from the original dataset (but this subset can contain multiple times the same image)
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=max_samples) # replacement = True => same image can be reprinted multiple times
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=kwargs.get('num_workers', 4), pin_memory=True, persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=kwargs.get('num_workers', 4), pin_memory=True, persistent_workers=True
        )

    # Fetch training dataset
    val_loader = None # to avoid error later
    val_dataset = None
    val_path = os.path.join(base_path, "val")
    if os.path.exists(val_path):
        val_dataset = LabColourDataset(
            val_path, transform=transform, mode='regression'
        )
        val_sampler = RandomSampler(val_dataset, replacement=True,
                                    num_samples=min(500, len(val_dataset)))
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    # --- Networks ---
    netG = Generator(image_size=image_size, use_classification=False).to(device)
    netD = Discriminator(input_nc=3).to(device) # receives complete image Lab

    # --- Optimisers ---
    optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) # beta1 (gradient average): configurable, beta2 (gradient variance): 0.999
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # Learning rate gradually decreases: good to stabilize CV
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=num_epochs, eta_min=1e-6)
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=num_epochs, eta_min=1e-6)

    # --- Loss functions ---
    criterion_GAN = nn.BCEWithLogitsLoss() # Binary Cross Entropy
    criterion_task = nn.L1Loss() # creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y

    # --- Metrics ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    scaler = torch.amp.GradScaler('cuda') # uses FP16 automatically (faster, less VRAM)

    # --- State ---
    history = {'loss_G': [], 'loss_D': [], 'psnr': [], 'ssim': []} # store metrics evolutions
    start_epoch = 0
    best_ssim = 0.0

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Resume ---
    # Resumption is now automated
    checkpoint_path = 'models/checkpoint_regression.pth' # if PC crashes at epoch 48, we resume at 49 with reloaded models, optimizers, schedulers, history
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            # ckpt is a DICTIONARY: {'netG_state_dict': ..., 'optG_state_dict': ..., 'schedulerG_state_dict': ..., 'history': ..., 'epoch': ..., ...}
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False) # loads the tensors on the CPU or GPU depending on the device, loads the entire dictionary, not just the weights
            netG.load_state_dict(ckpt['netG_state_dict']) # reload generator weights
            netD.load_state_dict(ckpt['netD_state_dict']) # reload discriminator weights
            if 'optG_state_dict' in ckpt:
                optG.load_state_dict(ckpt['optG_state_dict']) # reload optimizer state: momentum, learning rate, internal gradients, etc
            if 'optD_state_dict' in ckpt:
                optD.load_state_dict(ckpt['optD_state_dict']) 
            if 'schedulerG_state_dict' in ckpt:
                schedulerG.load_state_dict(ckpt['schedulerG_state_dict']) # reload generator/discriminator scheduler state, allowing learning rate to resume exactly where it was before
                schedulerD.load_state_dict(ckpt['schedulerD_state_dict'])
            history = ckpt.get('history', history) # update
            start_epoch = ckpt['epoch'] + 1 # update: resumes at next epoch
            best_ssim = ckpt.get('best_ssim', 0.0) # update
            print(f"Resuming from epoch {start_epoch} | Best SSIM so far: {best_ssim:.4f}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}). Starting from scratch.")
            start_epoch = 0

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs):
        netG.train()
        netD.train()
        running_loss_D = 0.0
        running_loss_G = 0.0

        # Loading bar display
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        # Fetch image one by one
        for data in pbar:
            L = data[0].to(device, non_blocking=True).float() # [B,1,256,256]
            target_ab = data[1].to(device, non_blocking=True).float() # [B,2,256,256]

            with torch.amp.autocast('cuda'):
                fake_ab = netG(L) # generator prediction

            # --- Update Discriminator ---
            # max[ E(log(D(x))) + E(log(1 - D(G(z)))) ]
            optD.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_real   = netD(torch.cat([L, target_ab], dim=1)) # discriminator output for original image
                # Proba that image is original given the original image: L_real_D = -log(sigma(D(x)))
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real) * 0.9) # true = 0.9 to avoid strong confidence

                pred_fake   = netD(torch.cat([L, fake_ab.detach()], dim=1)) # discriminator output for generated image
                # Proba that image is original given the fake image: L_fake_D = -log(sigma(1 - D(G(z))))
                loss_D_fake = criterion_GAN(pred_fake, torch.ones_like(pred_fake) * 0.1) # false = 0.1 to avoid strong confidence

                # Average
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            scaler.scale(loss_D).backward()
            scaler.step(optD)
            scaler.update()

            # --- Update Generator ---
            # min[ E(log(1 − D(G(z)))) ]
            optG.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_fake_G = netD(torch.cat([L, fake_ab], dim=1)) # discriminator output for generated image
                loss_G_GAN  = criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G)) # torch.ones_like(pred_fake_G) = 1
                loss_G_task = criterion_task(fake_ab, target_ab) # L1
                loss_G = loss_G_GAN + (loss_G_task * lambda_l1)

            scaler.scale(loss_G).backward()
            scaler.step(optG)
            scaler.update()

            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()
            pbar.set_postfix({'Loss_D': f"{loss_D.item():.3f}", 'Loss_G': f"{loss_G.item():.3f}"})

        schedulerG.step()
        schedulerD.step()

        # --- Validation ---
        netG.eval() # deactivate dropout, use global average/variance for normalization
        eval_loader = val_loader if val_loader else [data]
        val_psnr, val_ssim, num_val = 0.0, 0.0, 0

        with torch.no_grad():
            for v_data in eval_loader:
                v_L = v_data[0].to(device, non_blocking=True).float() # L
                v_target_ab = v_data[1].to(device, non_blocking=True).float() # ab true
                v_fake_ab = netG(v_L) # ab predicted

                real_rgb = lab_to_rgb_tensor(v_L, v_target_ab) # to rgb
                fake_rgb = lab_to_rgb_tensor(v_L, v_fake_ab)

                val_psnr += psnr_metric(fake_rgb, real_rgb).item() # PSNR
                val_ssim += ssim_metric(fake_rgb, real_rgb).item() # SSIM
                num_val  += 1

        val_psnr /= num_val # average
        val_ssim /= num_val

        epoch_loss_D = running_loss_D / len(train_loader)
        epoch_loss_G = running_loss_G / len(train_loader)
        history['loss_D'].append(epoch_loss_D) # save epoch loss
        history['loss_G'].append(epoch_loss_G)
        history['psnr'].append(val_psnr) # save metrics
        history['ssim'].append(val_ssim)


        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss G: {epoch_loss_G:.4f} | Loss D: {epoch_loss_D:.4f} | "
            f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
        )

        recent_ssim = float(np.mean(history['ssim'][-3:])) if len(history['ssim']) >= 3 else val_ssim

        # Save best model in memory
        if recent_ssim > best_ssim:
            best_ssim = recent_ssim
            torch.save(netG.state_dict(), 'models/netG_regression_best.pth')
            print(f"  → New best SSIM (3-epoch avg): {best_ssim:.4f} — best model saved.")

        # Every 5 epochs, plot image results via function in utils.py
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_path = f'results/regression/regression_epoch_{epoch+1:03d}.png'
            v_data = next(iter(eval_loader))
            v_L = v_data[0].to(device).float()
            v_target_ab = v_data[1].to(device).float()

            with torch.no_grad():
                v_fake_ab = netG(v_L)

            save_visualise_results(
                v_L, v_fake_ab, v_target_ab, 'regression', train_dataset, device, 
                save_path=save_path, epoch=epoch+1, stats={'psnr': val_psnr, 'ssim': val_ssim}
            )
            
        # Save model state at this epoch
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optG_state_dict': optG.state_dict(),
            'optD_state_dict': optD.state_dict(),
            'schedulerG_state_dict': schedulerG.state_dict(),
            'schedulerD_state_dict': schedulerD.state_dict(),
            'history': history,
            'best_ssim': best_ssim,
        }, checkpoint_path)

    return netG, netD, history
