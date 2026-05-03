import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

from dataset import LabColorDataset, get_class_weights
from networks import Generator, Discriminator
from utils import bins_to_ab_differentiable, visualize_results

def train_gan_loop(
    base_path,
    transform,
    device,
    mode='regression', 
    use_rebalancing=False, 
    max_samples=1000,
    batch_size=4,
    num_epochs=10,
    lr=3e-4,
    beta1=0.5,
    lambda_l1=100,
    num_bins=100,
    image_size=256,
    resume=True,
    num_workers=4
):
    """
    Unified GAN training loop with checkpoint support for multi-day training.
    """
    os.makedirs('models', exist_ok=True)
    checkpoint_path = f'models/checkpoint_{mode}.pth'
    
    # 1. Datasets & Loaders
    train_ds = LabColorDataset(os.path.join(base_path, "train"), transform=transform, mode=mode, num_bins=num_bins)
    val_ds = LabColorDataset(os.path.join(base_path, "val"), transform=transform, mode=mode, num_bins=num_bins)
    
    if max_samples: 
        train_ds = Subset(train_ds, range(min(max_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(max_samples // 5, len(val_ds))))
    
    # Enable multi-processing and memory pinning for speed
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ds_obj = train_ds.dataset if isinstance(train_ds, Subset) else train_ds

    # 2. Models
    netG = Generator(image_size=image_size, use_classification=(mode=='classification'), num_bins=num_bins).to(device)
    netD = Discriminator(input_nc=3).to(device)
    
    # 3. Losses & Optimisers
    criterionGAN = nn.BCELoss()
    if mode == 'classification':
        weights = get_class_weights(ds_obj, num_bins=num_bins).to(device) if use_rebalancing else None
        criterionContent = nn.CrossEntropyLoss(weight=weights)
    else:
        criterionContent = nn.L1Loss()

    optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    history = {'loss_G': [], 'psnr': []}
    start_epoch = 0

    # --- RESUME TRAINING ---
    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optG.load_state_dict(checkpoint['optG_state_dict'])
        optD.load_state_dict(checkpoint['optD_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch}")

    print(f"\n--- Starting GAN Training: {mode.upper()} ---")
    print(f"Target Epochs: {num_epochs} | Already completed: {start_epoch}")

    # 4. Training Loop
    for epoch in range(start_epoch, num_epochs):
        netG.train()
        netD.train()
        total_loss_G = 0
        
        for data in train_loader:
            if mode == 'classification':
                L, target, real_ab = [x.to(device) for x in data]
                L = L.float()
            else:
                L, target = [x.to(device) for x in data]
                L, target = L.float(), target.float()
                real_ab = target
            
            # --- Update Discriminator ---
            optD.zero_grad()
            pred_real = netD(torch.cat([L, real_ab], 1))
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
            
            fake_out = netG(L).float()
            fake_ab = fake_out if mode == 'regression' else bins_to_ab_differentiable(fake_out, ds_obj, device)
            pred_fake = netD(torch.cat([L, fake_ab.detach()], 1))
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optD.step()
            
            # --- Update Generator ---
            optG.zero_grad()
            pred_fake_G = netD(torch.cat([L, fake_ab], 1))
            loss_G_GAN = criterionGAN(pred_fake_G, torch.ones_like(pred_fake_G))
            
            loss_G_Content = criterionContent(fake_out, target)
            if mode == 'regression':
                loss_G_Content *= lambda_l1
            
            loss_G = loss_G_GAN + loss_G_Content
            loss_G.backward()
            optG.step()
            
            total_loss_G += loss_G.item()

        # 5. Validation
        netG.eval()
        val_psnr = 0
        samples_count = 0
        max_val_samples = 200
        with torch.no_grad():
            for data_v in val_loader:
                if samples_count >= max_val_samples: break
                
                L_v = data_v[0].to(device).float()
                target_v = data_v[1]
                
                out_v = netG(L_v).float()
                if mode == 'classification':
                    # Use soft-argmax for smoother validation metrics and results
                    pred_ab_tensor = bins_to_ab_differentiable(out_v, ds_obj, device)
                    pred_ab = pred_ab_tensor.cpu().numpy().transpose(0, 2, 3, 1)
                    true_ab = target_v.cpu().numpy().transpose(0, 2, 3, 1)
                else:
                    pred_ab = out_v.cpu().numpy().transpose(0, 2, 3, 1)
                    true_ab = target_v.cpu().numpy().transpose(0, 2, 3, 1)
                
                for i in range(len(L_v)): 
                    if samples_count >= max_val_samples: break
                    val_psnr += psnr(true_ab[i], pred_ab[i], data_range=2.0)
                    samples_count += 1
        
        avg_psnr = val_psnr / samples_count if samples_count > 0 else 0
        avg_loss_G = total_loss_G / len(train_loader)
        history['loss_G'].append(avg_loss_G)
        history['psnr'].append(avg_psnr)
        
        print(f"Epoch {epoch+1}/{num_epochs} | LossG: {avg_loss_G:.4f} | PSNR: {avg_psnr:.2f}")

        # --- SAVE CHECKPOINT ---
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optG_state_dict': optG.state_dict(),
            'optD_state_dict': optD.state_dict(),
            'history': history,
        }, checkpoint_path)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            # For visualization, we use the last batch from training loop
            visualize_results(L, fake_out, target, mode, ds_obj, device)
            
    # Save final model
    torch.save(netG.state_dict(), f'models/netG_{mode}_final.pth')
    print(f"Final model saved to models/netG_{mode}_final.pth")
            
    return netG, netD, history
