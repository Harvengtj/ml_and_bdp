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
    lr=2e-4,
    beta1=0.5,
    lambda_l1=100,
    num_bins=100,
    image_size=256
):
    """Unified GAN training loop for both Regression and Classification outputs."""
    
    # 1. Datasets & Loaders
    train_ds = LabColorDataset(os.path.join(base_path, "train"), transform=transform, mode=mode, num_bins=num_bins)
    val_ds = LabColorDataset(os.path.join(base_path, "val"), transform=transform, mode=mode, num_bins=num_bins)
    
    if max_samples: 
        train_ds = Subset(train_ds, range(min(max_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(max_samples // 5, len(val_ds))))
    
    print(f"\n--- GAN Training: {mode.upper()} ---")
    print(f"Training on {len(train_ds)} images.")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    ds_obj = train_ds.dataset if isinstance(train_ds, Subset) else train_ds

    # 2. Models
    netG = Generator(image_size=image_size, use_classification=(mode=='classification'), num_bins=num_bins).to(device)
    netD = Discriminator(input_nc=3).to(device)
    
    # 3. Losses & Optimizers
    criterionGAN = nn.BCELoss()
    if mode == 'classification':
        weights = get_class_weights(ds_obj, num_bins=num_bins).to(device) if use_rebalancing else None
        criterionContent = nn.CrossEntropyLoss(weight=weights)
    else:
        criterionContent = nn.L1Loss()

    optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # 4. Training Loop
    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        total_loss_G = 0
        
        for L, target in train_loader:
            L, target = L.to(device).float(), target.to(device)
            if mode == 'regression': target = target.float()
            
            # --- Update Discriminator ---
            optD.zero_grad()
            if mode == 'regression':
                real_ab = target
            else:
                real_ab = torch.from_numpy(ds_obj.bin_to_ab(target.cpu().numpy())).permute(0, 3, 1, 2).to(device).float()
            
            # Real pass
            pred_real = netD(torch.cat([L, real_ab], 1))
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
            
            # Fake pass
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
            
            # Content Loss (L1 or Cross-Entropy)
            loss_G_Content = criterionContent(fake_out, target)
            if mode == 'regression':
                loss_G_Content *= lambda_l1
            
            loss_G = loss_G_GAN + loss_G_Content
            loss_G.backward()
            optG.step()
            
            total_loss_G += loss_G.item()

        # 5. Validation & Visualization
        if (epoch + 1) == num_epochs:
            netG.eval()
            val_psnr = 0
            with torch.no_grad():
                for L_v, target_v in val_loader:
                    L_v = L_v.to(device).float()
                    out_v = netG(L_v).float()
                    
                    if mode == 'classification':
                        pred_ab = ds_obj.bin_to_ab(torch.argmax(out_v, 1).cpu().numpy())
                        true_ab = ds_obj.bin_to_ab(target_v.numpy())
                    else:
                        pred_ab = out_v.cpu().numpy().transpose(0, 2, 3, 1)
                        true_ab = target_v.cpu().numpy().transpose(0, 2, 3, 1)
                    
                    for i in range(len(L_v)): 
                        val_psnr += psnr(true_ab[i], pred_ab[i], data_range=2.0)
            
            avg_psnr = val_psnr / len(val_ds)
            print(f"Completed {num_epochs} Epochs. Final LossG: {total_loss_G/len(train_loader):.4f}, Final Validation PSNR: {avg_psnr:.2f}")
            visualize_results(L, fake_out, target, mode, ds_obj, device)
        else:
            print(f"Epoch {epoch+1}/{num_epochs} completed | LossG: {total_loss_G/len(train_loader):.4f}", end='\r')
            
    return netG, netD
