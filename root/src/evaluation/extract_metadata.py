import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json


#===============================================================================================================================
#=== GENERATE A MODEL TRAINING TRACKING REPORT ===
#===============================================================================================================================
def extract_metadata():
    """Generate a report for the current state of the model in the terminal.
       The function is also generates 'training_metadata.md' in code/results/.
    """
    
    
    device = torch.device("cpu")
    # Checkpoints to analyze
    checkpoints = {
        'Regression': 'models/checkpoint_regression.pth',
        'Classification': 'models/checkpoint_classification.pth'
    }
    
    report = "# Training Metadata Report\n\n"
    
    for name, path in checkpoints.items(): # key, value
        if not os.path.exists(path):
            report += f"## {name}\nCheckpoint not found at {path}\n\n"
            continue
            
        try:
            # Load with weights_only=False because we need history and metadata
            ckpt = torch.load(path, map_location=device) # ckpt is a DICTIONARY
            
            # Fetch stored data
            epochs_done = ckpt.get('epoch', 'N/A')
            best_metric = ckpt.get('best_ssim', 'N/A')
            history = ckpt.get('history', {})
            
            # Print report
            report += f"## {name} Model\n"
            report += f"- **Total Epochs Completed**: {epochs_done}\n"
            report += f"- **Best Validation Metric (SSIM)**: {best_metric:.4f}\n"
            
            if history:
                # Try both 'loss' (Classification) and 'loss_G' (Regression)
                train_loss = 0.0
                if 'loss' in history and history['loss']:
                    train_loss = history['loss'][-1] # last loss from the history
                elif 'loss_G' in history and history['loss_G']:
                    train_loss = history['loss_G'][-1] # last loss from the history
                
                # Print report
                report += "- **Final Training Loss**: {:.6f}\n".format(train_loss)
                report += "- **Final Val PSNR**: {:.2f} dB\n".format(history['psnr'][-1] if 'psnr' in history and history['psnr'] else 0)
                report += "- **Final Val SSIM**: {:.4f}\n".format(history['ssim'][-1] if 'ssim' in history and history['ssim'] else 0)
            
            # Count parameters
            state_dict = ckpt.get('netG_state_dict', {}) # ex: {"conv1.weight": tensor(...), "conv1.bias": tensor(...), "conv2.weight": tensor(...), "conv2.bias": tensor(...), "fc1.weight": tensor(...), "fc1.bias": tensor(...),...}
            num_params = sum(p.numel() for p in state_dict.values()) # number of weights of the model
            report += f"- **Generator Parameters**: {num_params:,}\n"
            
            report += "\n"
        except Exception as e:
            report += f"## {name}\nError loading checkpoint: {e}\n\n"

    save_path = "results/training_metadata.md"
    with open(save_path, "w") as f:
        f.write(report)
    print(f"Metadata report saved to {save_path}")
    print(report)
    
    
    

if __name__ == "__main__":
    extract_metadata()
