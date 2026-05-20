import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")

#================================================================================================================
#=== FILE TO GENERATE ALL IMPORTANT FILES FOR COMPARISONS ===
#================================================================================================================

# Ensure we are in project root
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')

from src.core.dataset import LabColourDataset
from src.core.networks import Generator
from src.core.utils import compare_colourisations, plot_training_history
from src.evaluation.compare_models import generate_comparison_plot, plot_metrics_from_checkpoints, plot_temperature_impact

CONFIG = {
    "image_size": 256,
    "num_bins": 100,
}

TRANSFORM = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
])


def load_model(mode: str, device: torch.device):
    """Load the best model for a given mode.

    Args:
        mode (str): 'regression' or 'classification'.
        device (torch.device): 'cpu' or 'gpu'.

    Returns:
        model (Generator): Trained model.
        history (dict): Dictionary containing important metrics' evolutions.
    """
    
    best_path = f"models/netG_{mode}_best.pth"
    ckpt_path = f"models/checkpoint_{mode}.pth"

    if os.path.exists(best_path):
        model = Generator(
            image_size=CONFIG["image_size"],
            use_classification=(mode == 'classification'),
            num_bins=CONFIG["num_bins"],
        ).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device)) # load model parameters
        model.eval()
        
        history = None
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            history = ckpt.get("history", None) # record history
        return model, history
    return None, None


def main():
    """Main function that allows to generate all important documents (plots, numerical assessment, ...)"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #base_path = "data/imagenet"
    base_path = "data/coco"
    
    # Functions from compare_models.py
    print("\n--- Generating Comparison Plots ---")
    generate_comparison_plot()
    plot_metrics_from_checkpoints()
    plot_temperature_impact()

    print("\n--- Running Qualitative Evaluation ---")
    netG_reg, hist_reg = load_model("regression", device)
    netG_clas, hist_clas = load_model("classification", device)

    # Functions from utils.py
    if hist_reg and hist_clas:
        plot_training_history(hist_reg, hist_clas)

    if netG_reg and netG_clas and os.path.exists(os.path.join(base_path, "val")):
        val_dataset = LabColourDataset(os.path.join(base_path, "val"), TRANSFORM)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
        compare_colourisations(netG_reg, netG_clas, val_loader, val_dataset, device, num_samples=10)
    else:
        print("Models or validation dataset missing for qualitative evaluation.")

if __name__ == "__main__":
    main()
