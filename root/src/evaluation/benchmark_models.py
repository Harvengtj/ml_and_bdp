import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time
import pandas as pd
import lpips
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from core.networks import Generator
from core.dataset import LabColourDataset
from core.utils import bins_to_ab_differentiable, lab_to_rgb_tensor

#===============================================================================================================================
#=== BENCHMARK BETWEEN REGRESSION AND CLASSIFICATION (INFERENCE TIME, PSNR, SSIM, LPIPS, COLOURFULNESS) ===
#===============================================================================================================================
def calculate_colourfulness(img_tensor):
    """
    Hasler and Suesstrunk (2003) metric for image colourfulness. It measures how “colorful” an image is.
    
    Args:
        img_tensor (tensor): Expects RGB tensor [3, H, W] in range [0, 1].
        
    Returns:
        (tensor): Scalar value for coloufulness. 
    """
    
    # Extract RGB channels
    R = img_tensor[0, :, :]
    G = img_tensor[1, :, :]
    B = img_tensor[2, :, :]
    
    
    rg = torch.abs(R - G) # red >< green
    yb = torch.abs(0.5 * (R + G) - B) # yellow >< blue
    
    std_rg = torch.std(rg)
    mean_rg = torch.mean(rg)
    
    std_yb = torch.std(yb)
    mean_yb = torch.mean(yb)
    
    std_root = torch.sqrt(std_rg**2 + std_yb**2)
    mean_root = torch.sqrt(mean_rg**2 + mean_yb**2)
    
    return std_root + 0.3 * mean_root


def benchmark():
    """Compare regression and classification. Write benchmark results in 'results/model_benchmarks.csv'.
       Metrics are: inference time, PSNR, SSIM, LPIPS, coloufulness.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")

    # --- Configuration ---
    image_size = 256
    num_bins = 100
    base_path = "data/coco"
    val_path = os.path.join(base_path, "val")
    num_samples = 200 
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    if not os.path.exists(val_path):
        print(f"Error: Validation path {val_path} not found.")
        return

    dataset = LabColourDataset(val_path, transform=transform)
    indices = np.random.choice(len(dataset), num_samples, replace=False) # random samples generated from np.arange(len(dataset))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False) # take a subset from the dataset via indices

    # --- Metrics Setup ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # --- Models Definition ---
    # List of dictionaries
    models_to_test = [
        {
            'name': 'Regression',
            'path': 'models/netG_regression_best.pth',
            'is_cls': False
        },
        {
            'name': 'Classification',
            'path': 'models/netG_classification_best.pth',
            'is_cls': True
        }
    ]

    results = []
    gt_colourfulness = 0.0

    # 1 loop for each model
    for m_cfg in models_to_test:
        name = m_cfg['name']
        path = m_cfg['path']
        is_cls = m_cfg['is_cls']

        if not os.path.exists(path):
            print(f"Skipping {name}: path {path} not found.")
            continue

        print(f"\nTesting {name}...")
        model = Generator(image_size=image_size, use_classification=is_cls, num_bins=num_bins).to(device)
        model.load_state_dict(torch.load(path, map_location=device)) # load state dictionary (parameters) from the checkpoint
        model.eval()

        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_colourfulness = 0.0
        inference_times = []

        with torch.no_grad():
            for L, target_ab in tqdm(loader, desc=f"Benchmarking {name}"):
                L, target_ab = L.to(device).float(), target_ab.to(device).float()
                
                # Precise timing
                if device.type == 'cuda':
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()
                else:
                    start_time = time.time()

                # Inference
                output = model(L)
                if is_cls:
                    fake_ab = bins_to_ab_differentiable(output, dataset, device, temperature=0.38) # to get averaged a and b for each pixel
                else:
                    fake_ab = output

                if device.type == 'cuda':
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender) # in ms
                else:
                    curr_time = (time.time() - start_time) * 1000 # to ms

                inference_times.append(curr_time)

                # Metrics calculation
                real_rgb = lab_to_rgb_tensor(L, target_ab)
                fake_rgb = lab_to_rgb_tensor(L, fake_ab)

                total_psnr += psnr_metric(fake_rgb, real_rgb).item()
                total_ssim += ssim_metric(fake_rgb, real_rgb).item()
                total_lpips += lpips_fn(fake_rgb * 2.0 - 1.0, real_rgb * 2.0 - 1.0).item()
                
                # Colourfulness
                total_colourfulness += calculate_colourfulness(fake_rgb[0]).item()
                if m_cfg['name'] == 'Regression': # Baseline only once
                    gt_colourfulness += calculate_colourfulness(real_rgb[0]).item()

        # Average results
        avg_time = np.mean(inference_times)
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        avg_lpips = total_lpips / num_samples
        avg_color = total_colourfulness / num_samples

        print(f"Results for {name}:")
        print(f"  Avg Time: {avg_time:.2f} ms")
        print(f"  Avg PSNR: {avg_psnr:.2f} dB")
        print(f"  Avg SSIM: {avg_ssim:.4f}")
        print(f"  Avg LPIPS: {avg_lpips:.4f}")
        print(f"  Avg Colourfulness: {avg_color:.4f}")

        results.append({
            'Method': name,
            'Avg Inference Time (ms)': round(avg_time, 2),
            'Avg PSNR (dB)': round(avg_psnr, 2),
            'Avg SSIM': round(avg_ssim, 4),
            'Avg LPIPS': round(avg_lpips, 4),
            'Colourfulness': round(avg_color, 4)
        })

    # Add Ground Truth as baseline
    avg_gt_color = gt_colourfulness / num_samples
    results.append({
        'Method': 'Ground Truth',
        'Avg Inference Time (ms)': 0.0,
        'Avg PSNR (dB)': 0.0,
        'Avg SSIM': 1.0,
        'Avg LPIPS': 0.0,
        'Colourfulness': round(avg_gt_color, 4)
    })

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    save_path = "results/model_benchmarks.csv"
    df.to_csv(save_path, index=False)
    print(f"\nFull benchmark saved to {save_path}")
    print(df)

if __name__ == "__main__":
    benchmark()
