Image Colorization Project - Submission
========================================

Author: Julian De Sutter-André, Justin Harvengt, Cédric Sipakam
Date: June 7, 2026

Description
-----------
This project implements two deep learning approaches for automatic image colorization:
1. Regression-based GAN: Direct prediction of chrominance (ab) channels using L1 + Adversarial loss.
2. Classification-based: Prediction of color bin indices (based on Zhang et al. 2016).

The models are trained using a U-Net generator architecture. 
The best checkpoints are saved in the 'result' folder and represent regression and classification models trained by ourselves via the ImageNet dataset.
However, for memory reasons, this github is suited to download and to exploit the COCO dataset, which is much smaller (take a look at generate/setup_coco to choose the size of the dataset).
During the training phase, at each epoch, the state of the model is saved in the event the computer crashes or you want to stop.
You should suppress the 'models' folder to train the desired model from the beginning.

Project Structure
-----------------
docs/                    # Project references, guidelines
generate_data/ 
  ├── setup_coco.py      # Generate COCO dataset in 'data' folder 
models/                  # Best checkpoints and saved models (trained using ImageNet dataset)
results/                 # Generated plots, benchmarks, and training reports
src/
  ├── main.py            # Main entry point for training and evaluation
  ├── core/              # Core logic (networks, dataset, utils)
  ├── training/          # Training loops (regression, classification)
  └── evaluation/        # Benchmarks, comparisons, and metadata extraction
README.txt               # What you are reading now 
requirements.txt         # Required librairies to install
run_eval.py              # Full evaluation of the models 

How to Run
----------
All scripts should be executed from the project root directory.

1. Install dependencies:
   pip install -r requirements.txt

2. Install COCO dataset 
   python3 generate_data/setup_coco.py

3. Run Regression GAN training:
   python3 src/main.py --train-reg

4. Run Classification training:
   python3 src/main.py --train-clas

5. Generates a 5-image comparison plot from the validation set:
   python3 src/main.py --test

5. Run quantitative benchmark (detailed CSV results):
   python3 src/evaluation/benchmark_models.py

6. Generate training metadata report:
   python3 src/evaluation/extract_metadata.py

7. Run a full evaluation of the models (generates .png, .pdf, ...):
   python3 run_eval.py

Dependencies
------------
- torch, torchvision
- numpy, pandas
- matplotlib, seaborn
- scikit-image, lpips
- tqdm, torchmetrics
