import argparse
import gc
import os
import sys
import random
import matplotlib
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.core.dataset import LabColourDataset
from src.core.networks import Generator
from src.core.utils import lab_to_rgb_tensor, bins_to_ab_differentiable
from src.training.train_reg import train_reg_loop
from src.training.train_clas import train_clas_loop


#================================================================================================================
#=== MAIN FILE ===
#================================================================================================================

# --- Configuration ---
CONFIG = {
    "image_size":      256,
    "num_bins":        100,
    "num_epochs":      200,
    "lr":              5e-5,
    "beta1":           0.5,
    "num_workers":     12,
    "batch_size_reg":  128,
    "batch_size_cls":  320,
}

TRANSFORM = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
])

#base_path = "data/imagenet"
base_path = "data/coco"

def get_training_config():
    """Returns the training parameters shared by both models.

    The batch sizes are excluded because regression and classification
    use different values.

    Returns:
        config (dict): dictionary containing the parameters needed to train the models.
    """

    config = CONFIG.copy()

    del config["batch_size_reg"]
    del config["batch_size_cls"]

    return config


def load_model(mode: str, device: torch.device):
    """Load the best model for a given mode.

    Args:
        mode (str): 'regression' or 'classification'.
        device (torch.device): 'cpu' or 'gpu'.

    Returns:
        model (Generator): Trained model.
    """
    
    path = f"models/netG_{mode}_best.pth"
    if not os.path.exists(path):
        return None
    model = Generator(
        image_size=CONFIG["image_size"],
        use_classification=(mode == 'classification'),
        num_bins=CONFIG["num_bins"]
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model



def run_test(device: torch.device):
    """Generates a 5-image comparison plot from the validation set.

    Args:
        device (torch.device): 'cpu' or 'gpu'.
    """
    
    val_path = os.path.join(base_path, "val")
    if not os.path.exists(val_path):
        print(f"Error: Validation path {val_path} not found.")
        return

    netG_reg = load_model("regression", device) # load models
    netG_clas = load_model("classification", device)

    if not netG_reg and not netG_clas:
        print("Error: No trained models found in models/.")
        return

    print("Loading validation samples...")
    full_val_dataset = LabColourDataset(val_path, transform=TRANSFORM)
    indices = random.sample(range(len(full_val_dataset)), 5) # take 5 random images
    subset = Subset(full_val_dataset, indices)
    loader = DataLoader(subset, batch_size=5) # batch: [5, 2, H, W] or batch: [5, N, H, W]
    
    L, target_ab = next(iter(loader))
    L, target_ab = L.to(device).float(), target_ab.to(device).float()

    with torch.no_grad():
        # Regression results
        reg_rgb = None
        if netG_reg:
            out_reg_ab = netG_reg(L)
            reg_rgb = lab_to_rgb_tensor(L, out_reg_ab).cpu().numpy().transpose(0, 2, 3, 1)
        
        # Classification results
        cls_rgb = None
        if netG_clas:
            out_cls_logits = netG_clas(L)
            out_cls_ab = bins_to_ab_differentiable(out_cls_logits, full_val_dataset, device)
            cls_rgb = lab_to_rgb_tensor(L, out_cls_ab).cpu().numpy().transpose(0, 2, 3, 1)

    # Plotting
    real_rgb = lab_to_rgb_tensor(L, target_ab).cpu().numpy().transpose(0, 2, 3, 1)
    L_np = L.cpu().numpy().squeeze(1)

    fig, axes = plt.subplots(5, 4, figsize=(14, 18))
    titles = ["Input (L)", "Regression", "Classification", "Ground Truth"]
    
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=15, pad=10)

    for i in range(5):
        axes[i, 0].imshow(L_np[i], cmap='gray')
        if reg_rgb is not None: axes[i, 1].imshow(reg_rgb[i])
        if cls_rgb is not None: axes[i, 2].imshow(cls_rgb[i])
        axes[i, 3].imshow(real_rgb[i])
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = "results/test_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Test comparison plot saved to {save_path}")
    plt.close()
    

def main():
    """Main function that allows:
        - to test a model with 5 validation samples (--test).
        - to train a model for regression (--train_reg).
        - to train a model for classification (--train_cls).
    """
    parser = argparse.ArgumentParser(description="Image Colourisation - Main Entry Point")
    parser.add_argument("--train-reg", action="store_true", help="Run Regression GAN training.")
    parser.add_argument("--train-clas", action="store_true", help="Run Classification training.")
    parser.add_argument("--test", action="store_true", help="Generate a 5-image comparison plot.")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) == 1: # if there is only 'main.py' => problem
        parser.print_help()
        return

    # Test already trained model (via checkpoints)
    if args.test: # --test
        run_test(device)
        return

    if not os.path.exists(os.path.join(base_path, "train")):
        print(f"Error: Dataset not found at {base_path}")
        return
    
    # Train with regression
    if args.train_reg: # --train_reg
        print("\n" + "=" * 60)
        print("Regression GAN Training Mode")
        print("=" * 60)
        train_reg_loop(
            base_path=base_path,
            transform=TRANSFORM,
            device=device,
            batch_size=CONFIG["batch_size_reg"],
            **get_training_config(),
        )
        gc.collect() # free memory
        torch.cuda.empty_cache()

    # Train with classification
    if args.train_clas: # --train-class
        print("\n" + "=" * 60)
        print("Classification Training Mode")
        print("=" * 60)
        train_clas_loop(
            base_path=base_path,
            transform=TRANSFORM,
            device=device,
            batch_size=CONFIG["batch_size_cls"],
            **get_training_config(),
        )
        gc.collect() # free memory
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
