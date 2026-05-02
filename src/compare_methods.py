import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr

from dataset import LabColorDataset, get_class_weights
from networks import Generator

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-3
NUM_BINS = 100

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor()
])

def train_model(mode='regression', use_rebalancing=False):
    print(f"\nTraining Mode: {mode} (Rebalancing: {use_rebalancing})")
    
    # Dataset detection
    possible_paths = ["data/coco", "data/images"]
    base_path = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "train")):
            base_path = p
            break
    
    if not base_path:
        print("Dataset not found. Please ensure data is correctly placed in data/coco.")
        return
    
    # Dataset
    train_dataset = LabColorDataset(os.path.join(base_path, "train"), transform=transform, mode=mode, num_bins=NUM_BINS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = LabColorDataset(os.path.join(base_path, "val"), transform=transform, mode=mode, num_bins=NUM_BINS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    netG = Generator(image_size=IMAGE_SIZE, use_classification=(mode=='classification'), num_bins=NUM_BINS).to(DEVICE)
    
    # Loss & Optimizer
    if mode == 'classification':
        weights = get_class_weights(train_dataset, num_bins=NUM_BINS).to(DEVICE) if use_rebalancing else None
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.L1Loss()
        
    optimizer = optim.Adam(netG.parameters(), lr=LR)
    
    history = {'loss': [], 'psnr': []}
    
    for epoch in range(NUM_EPOCHS):
        netG.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = netG(inputs)
            
            if mode == 'classification':
                # outputs: [B, NUM_BINS, H, W], targets: [B, H, W]
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation PSNR
        netG.eval()
        total_psnr = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = netG(inputs)
                
                # Convert back to AB for PSNR calculation
                if mode == 'classification':
                    # Take argmax or annealed mean? Argmax for simplicity
                    pred_bins = torch.argmax(outputs, dim=1).cpu().numpy()
                    pred_ab = val_dataset.bin_to_ab(pred_bins) # [B, H, W, 2]
                    
                    # Target bins to AB
                    target_ab = val_dataset.bin_to_ab(targets.numpy())
                else:
                    pred_ab = outputs.cpu().numpy().transpose(0, 2, 3, 1)
                    target_ab = targets.cpu().numpy().transpose(0, 2, 3, 1)
                
                # PSNR (on AB channels normalized to [0,1])
                for j in range(len(inputs)):
                    total_psnr += psnr(target_ab[j], pred_ab[j], data_range=1.0)
                    
        epoch_psnr = total_psnr / len(val_dataset)
        history['loss'].append(epoch_loss)
        history['psnr'].append(epoch_psnr)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.2f}")
        
    return history

if __name__ == "__main__":
    if not os.path.exists("data/coco/train"):
        print("Dataset not found at data/coco/train. Please ensure data is correctly placed.")
    else:
        history_reg = train_model(mode='regression')
        history_cls = train_model(mode='classification', use_rebalancing=True)
        
        # Plotting
        plt.figure(figsize=(12, 5))
        
        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, NUM_EPOCHS + 1), history_reg['loss'], label='Regression (L1)')
        plt.plot(range(1, NUM_EPOCHS + 1), history_cls['loss'], label='Classification (CE)')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # PSNR Plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, NUM_EPOCHS + 1), history_reg['psnr'], label='Regression (L1)')
        plt.plot(range(1, NUM_EPOCHS + 1), history_cls['psnr'], label='Classification (CE)')
        plt.title('Validation PSNR')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('docs/pictures/comparison_metrics.png')
        plt.show()
        print("\nComparison plot saved to docs/pictures/comparison_metrics.png")
