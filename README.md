# Machine Learning and Big Data Processing
Repository for the project 'Grayscale Image Colorization'.

# Deep Learning Project Summary: Grayscale Image Colorization

## 1. Current Progress & Group Decisions
* **Selected Topic:** Grayscale Image Colorization exploring Generative Adversarial Networks (GANs) and class rebalancing.
* **Framework:** PyTorch (adapting from existing TensorFlow open-source code).
* **Code Status:** Preliminary draft of the Generator and Discriminator classes initialized, structured similarly to practical sessions.
* **Dataset:** ImageNet.
* **Report Status:** Overleaf document created; initial algorithm description drafted.

## 2. Official Project Expectations & Deliverables
* **Technical Objectives:** Process images in the CIE Lab color space, taking the L (luminance) channel as input to predict the ab (chrominance) channels.
* **Required Implementations:** At least two formulation variants (e.g., comparing regression against classification).
* **Experimental Focus:** Qualitative and quantitative comparisons showing how class rebalancing and architectural choices impact color vividness and reconstruction quality.
* **Code Originality:** Bulk of the project must be original Python code. External code must be explicitly acknowledged and cited.
* **Group Size:** Strictly 3 members per group.

## 3. Submission Details
* **Written Report:** 4 to 6 pages formatted using the IEEE double-column template. Obligatory sections: Title/Authors, Abstract, Introduction, Related Work, Data, Methods, Experiments, Conclusion.
* **Submission Format:** A single ZIP file containing the PDF report, source code, and a `README.txt` with running instructions sent to the Teaching Assistants.

## 4. Grading Structure (30% of final course score)
* **Technical Quality:** 35% (Group score).
* **Written Report:** 25% (Group score).
* **Oral Defense:** 40% (Individual score) – 30-minute session featuring a 15–20 minute group PowerPoint presentation followed by individual Q&A.

---

## 5. Artifacts & Resources

### Datasets
* **ImageNet:** The chosen image source for training the colorization model.

### Frameworks & Tools
* **PyTorch:** The mandatory framework chosen for the project implementation.
* **TensorFlow:** The framework used in the baseline GitHub repository (requires translation to PyTorch).

### Research Papers & Literature
* **Colorful Image Colorization (Zhang et al., 2016):** Key reference for the class-rebalancing technique and classification approach in the CIE Lab color space.
* **Image Colorization using Generative Adversarial Networks (Nazeri et al., 2018):** Core reference for the GAN-based implementation.
* **Deep Colorization (Cheng et al., 2015):** Additional baseline paper provided in the project guidelines.

### Code Repositories
* **TensorFlow GAN Baseline:** [https://github.com/ImagingLab/Colorizing-with-GANs](https://github.com/ImagingLab/Colorizing-with-GANs)
* **Class Rebalancing & Classification Reference:** [https://github.com/richzhang/colorization?tab=readme-ov-file](https://github.com/richzhang/colorization?tab=readme-ov-file)

### Group Workspace
* **Overleaf (Initial Draft):** [https://fr.overleaf.com/2338948479ybxkfzrddjkj#6f4e3c](https://fr.overleaf.com/2338948479ybxkfzrddjkj#6f4e3c)
* **Overleaf (Premium Link):** [https://fr.overleaf.com/3745371124qvcbvscryrjw#af1990](https://fr.overleaf.com/3745371124qvcbvscryrjw#af1990)

### Official Course Resources
* **IEEE Final Report Templates:** [https://www.ieee.org/conferences/publishing/templates](https://www.ieee.org/conferences/publishing/templates)

---

## Checklist: Project Management Tasks (Need approval)

- [ ] **Phase 1 - Project Scoping and Alignment**
  - [ ] **1.1 - Confirm Official Requirements**
    - [ ] **1.1.1 -** Extract the project expectations from `Deep_Learning_Projects_2025_2026.pdf`
    - [ ] **1.1.2 -** Confirm that the mandatory implementation variants are regression and classification
    - [ ] **1.1.3 -** Confirm that class rebalancing is a required experimental focus
    - [ ] **1.1.4 -** Confirm required report length, report sections, submission format, and oral defense format
    - [ ] **1.1.5 -** Create a short requirements checklist to use before final submission
  - [ ] **1.2 - Define the Main Research Question**
    - [ ] **1.2.1 -** State the problem as grayscale-to-color image colorization in CIE Lab space
    - [ ] **1.2.2 -** Define the input as the `L` luminance channel
    - [ ] **1.2.3 -** Define the output as the `ab` chrominance channels
    - [ ] **1.2.4 -** Define the central comparison as regression versus classification
    - [ ] **1.2.5 -** Define the main hypothesis: class rebalancing improves color vividness but may trade off pixel accuracy
  - [ ] **1.3 - Agree on Project Scope**
    - [ ] **1.3.1 -** Mark regression baseline as mandatory
    - [ ] **1.3.2 -** Mark classification without rebalancing as mandatory
    - [ ] **1.3.3 -** Mark classification with rebalancing as mandatory
    - [ ] **1.3.4 -** Mark GAN implementation as optional extension
    - [ ] **1.3.5 -** Decide that optional work cannot start before mandatory experiments are complete
  - [ ] **1.4 - Define Success Criteria**
    - [ ] **1.4.1 -** The data pipeline can reconstruct RGB images from Lab channels
    - [ ] **1.4.2 -** Regression training loss decreases and produces visible color outputs
    - [ ] **1.4.3 -** Classification training loss decreases and produces decoded color outputs
    - [ ] **1.4.4 -** Rebalanced classification changes the predicted color distribution measurably
    - [ ] **1.4.5 -** Final results include metrics, qualitative grids, and a failure-case analysis

- [ ] **Phase 2 - Literature Review and Theory Grounding**
  - [ ] **2.1 - Study Zhang et al., Colorful Image Colorization**
    - [ ] **2.1.1 -** Summarize why colorization is a multimodal prediction problem
    - [ ] **2.1.2 -** Explain why L2 regression tends to produce dull colors
    - [ ] **2.1.3 -** Document the Lab-space formulation
    - [ ] **2.1.4 -** Document `ab` color quantization
    - [ ] **2.1.5 -** Document class rebalancing for rare colors
    - [ ] **2.1.6 -** Document annealed-mean decoding
    - [ ] **2.1.7 -** Identify which ideas will be implemented exactly and which will be simplified
  - [ ] **2.2 - Study Cheng et al., Deep Colorization**
    - [ ] **2.2.1 -** Summarize the historical transition from reference-based methods to learned colorization
    - [ ] **2.2.2 -** Document the role of local, mid-level, and semantic features
    - [ ] **2.2.3 -** Document why semantic context helps reduce ambiguity
    - [ ] **2.2.4 -** Decide not to implement the full hand-engineered pipeline
    - [ ] **2.2.5 -** Extract useful discussion points for the Related Work section
  - [ ] **2.3 - Study Nazeri et al., Image Colorization using GANs**
    - [ ] **2.3.1 -** Summarize the conditional GAN formulation
    - [ ] **2.3.2 -** Document the U-Net generator idea
    - [ ] **2.3.3 -** Document the discriminator conditioning on grayscale input
    - [ ] **2.3.4 -** Document the combined adversarial and L1 reconstruction loss
    - [ ] **2.3.5 -** Decide whether the GAN will remain a related-work topic or become an optional extension
  - [ ] **2.4 - Connect Papers to Course Theory**
    - [ ] **2.4.1 -** Map regression baseline to supervised learning with continuous targets
    - [ ] **2.4.2 -** Map classification colorization to supervised learning with categorical targets
    - [ ] **2.4.3 -** Map class rebalancing to imbalanced learning
    - [ ] **2.4.4 -** Map CNN/U-Net architecture to convolutional feature extraction and image-to-image prediction
    - [ ] **2.4.5 -** Map GAN extension to adversarial learning if implemented
  - [ ] **2.5 - Produce Literature Deliverables**
    - [ ] **2.5.1 -** Write one paragraph per paper for the report
    - [ ] **2.5.2 -** Write one comparison paragraph explaining why Zhang et al. is the main reference
    - [ ] **2.5.3 -** Prepare two to three slides for related work
    - [ ] **2.5.4 -** Create a citation list for the report and code acknowledgements

- [ ] **Phase 3 - Repository and Development Setup**
  - [ ] **3.1 - Define Repository Structure**
    - [ ] **3.1.1 -** Create a `src/` directory for source code
    - [ ] **3.1.2 -** Create a `configs/` directory for experiment configuration files
    - [ ] **3.1.3 -** Create a `data/` directory or document the external dataset path
    - [ ] **3.1.4 -** Create a `checkpoints/` directory for trained model weights
    - [ ] **3.1.5 -** Create a `results/` directory for metrics, figures, and qualitative outputs
    - [ ] **3.1.6 -** Create a `report/` or Overleaf synchronization plan for the final writeup
  - [ ] **3.2 - Define Python Environment**
    - [ ] **3.2.1 -** Select Python version
    - [ ] **3.2.2 -** Install PyTorch
    - [ ] **3.2.3 -** Install torchvision
    - [ ] **3.2.4 -** Install scikit-image or OpenCV for Lab conversion
    - [ ] **3.2.5 -** Install NumPy, pandas, matplotlib, and tqdm
    - [ ] **3.2.6 -** Install scikit-learn if needed for metrics or splitting
    - [ ] **3.2.7 -** Create `requirements.txt`
  - [ ] **3.3 - Define Coding Standards**
    - [ ] **3.3.1 -** Use clear module names for data, models, training, metrics, and visualization
    - [ ] **3.3.2 -** Keep all random seeds configurable
    - [ ] **3.3.3 -** Keep all paths configurable
    - [ ] **3.3.4 -** Save experiment settings with each run
    - [ ] **3.3.5 -** Add citations or comments when a design is directly inspired by a paper
  - [ ] **3.4 - Define Experiment Tracking**
    - [ ] **3.4.1 -** Decide whether to use CSV logs, TensorBoard, or both
    - [ ] **3.4.2 -** Log training loss after each epoch
    - [ ] **3.4.3 -** Log validation metrics after each epoch
    - [ ] **3.4.4 -** Save qualitative validation grids at fixed intervals
    - [ ] **3.4.5 -** Save the best checkpoint according to validation loss

- [ ] **Phase 4 - Dataset Acquisition and Preprocessing**
  - [ ] **4.1 - Select Dataset**
    - [ ] **4.1.1 -** Choose BSDS for fast controlled experiments or ImageNet subset for broader variety
    - [ ] **4.1.2 -** Verify that the selected dataset license allows academic use
    - [ ] **4.1.3 -** Record the dataset source URL or citation
    - [ ] **4.1.4 -** Record the number of available images
    - [ ] **4.1.5 -** Decide final image resolution for training
  - [ ] **4.2 - Create Dataset Splits**
    - [ ] **4.2.1 -** Define train, validation, and test split proportions
    - [ ] **4.2.2 -** Generate split files with image filenames
    - [ ] **4.2.3 -** Save split files under `data/splits/`
    - [ ] **4.2.4 -** Verify that no image appears in more than one split
    - [ ] **4.2.5 -** Use the same splits for all model variants
  - [ ] **4.3 - Implement Image Loading**
    - [ ] **4.3.1 -** Load each image as RGB
    - [ ] **4.3.2 -** Resize or crop each image consistently
    - [ ] **4.3.3 -** Convert images to tensors
    - [ ] **4.3.4 -** Handle grayscale or corrupted source images safely
    - [ ] **4.3.5 -** Add basic data augmentation if dataset size is small
  - [ ] **4.4 - Implement Lab Conversion**
    - [ ] **4.4.1 -** Convert RGB images to CIE Lab
    - [ ] **4.4.2 -** Extract `L` as model input
    - [ ] **4.4.3 -** Extract `a,b` as chrominance targets
    - [ ] **4.4.4 -** Normalize `L` to the chosen input range
    - [ ] **4.4.5 -** Normalize `a,b` to the chosen target range
    - [ ] **4.4.6 -** Implement inverse conversion from predicted Lab to RGB
  - [ ] **4.5 - Validate Preprocessing**
    - [ ] **4.5.1 -** Visualize original RGB images
    - [ ] **4.5.2 -** Visualize grayscale images derived from `L`
    - [ ] **4.5.3 -** Reconstruct RGB images from original `L,a,b`
    - [ ] **4.5.4 -** Verify reconstruction quality
    - [ ] **4.5.5 -** Save a preprocessing sanity-check figure

- [ ] **Phase 5 - Regression Baseline**
  - [ ] **5.1 - Define Regression Formulation**
    - [ ] **5.1.1 -** Set input tensor shape to `[B, 1, H, W]`
    - [ ] **5.1.2 -** Set output tensor shape to `[B, 2, H, W]`
    - [ ] **5.1.3 -** Use normalized continuous `ab` targets
    - [ ] **5.1.4 -** Choose L1 loss as the main regression loss
    - [ ] **5.1.5 -** Optionally include L2 loss as a small ablation
  - [ ] **5.2 - Implement Regression Dataset**
    - [ ] **5.2.1 -** Return `(L, ab)` pairs
    - [ ] **5.2.2 -** Verify tensor shapes
    - [ ] **5.2.3 -** Verify tensor value ranges
    - [ ] **5.2.4 -** Add batch loading with PyTorch `DataLoader`
    - [ ] **5.2.5 -** Test one batch through the full inverse-visualization pipeline
  - [ ] **5.3 - Implement Regression Model**
    - [ ] **5.3.1 -** Build a simple encoder-decoder or U-Net backbone
    - [ ] **5.3.2 -** Use one input channel
    - [ ] **5.3.3 -** Use two output channels
    - [ ] **5.3.4 -** Add skip connections if using U-Net
    - [ ] **5.3.5 -** Add `tanh` output if targets are normalized to `[-1, 1]`
    - [ ] **5.3.6 -** Verify output shape with a dummy batch
  - [ ] **5.4 - Implement Regression Training**
    - [ ] **5.4.1 -** Create optimizer
    - [ ] **5.4.2 -** Create training loop
    - [ ] **5.4.3 -** Create validation loop
    - [ ] **5.4.4 -** Save epoch-level losses
    - [ ] **5.4.5 -** Save best checkpoint
    - [ ] **5.4.6 -** Save sample predictions during training
  - [ ] **5.5 - Evaluate Regression Baseline**
    - [ ] **5.5.1 -** Compute validation and test MAE in `ab` space
    - [ ] **5.5.2 -** Compute RGB PSNR
    - [ ] **5.5.3 -** Compute RGB SSIM
    - [ ] **5.5.4 -** Compute mean chroma or colorfulness
    - [ ] **5.5.5 -** Generate qualitative comparison grids
    - [ ] **5.5.6 -** Record observed failure modes

- [ ] **Phase 6 - Color Quantization for Classification**
  - [ ] **6.1 - Define Color-Bin Strategy**
    - [ ] **6.1.1 -** Decide whether to use a regular `ab` grid or sampled in-gamut bins
    - [ ] **6.1.2 -** Choose initial number of bins
    - [ ] **6.1.3 -** Define bin centers in `ab` space
    - [ ] **6.1.4 -** Remove unused or invalid bins if needed
    - [ ] **6.1.5 -** Save bin centers to disk for reproducibility
  - [ ] **6.2 - Implement Pixel-to-Bin Encoding**
    - [ ] **6.2.1 -** Map each `ab` pixel to the nearest bin center
    - [ ] **6.2.2 -** Return target tensor shape `[H, W]`
    - [ ] **6.2.3 -** Verify that all target IDs are valid
    - [ ] **6.2.4 -** Test encoding speed on a batch
    - [ ] **6.2.5 -** Cache encoded targets if runtime is too slow
  - [ ] **6.3 - Implement Bin-to-Color Decoding**
    - [ ] **6.3.1 -** Decode hard bin IDs to `ab` bin centers
    - [ ] **6.3.2 -** Decode probability distributions with argmax
    - [ ] **6.3.3 -** Implement annealed-mean decoding if time allows
    - [ ] **6.3.4 -** Convert decoded `ab` and input `L` back to RGB
    - [ ] **6.3.5 -** Save a quantization reconstruction sanity-check figure
  - [ ] **6.4 - Analyze Bin Distribution**
    - [ ] **6.4.1 -** Count training pixels per bin
    - [ ] **6.4.2 -** Plot bin-frequency histogram
    - [ ] **6.4.3 -** Identify low-saturation dominant bins
    - [ ] **6.4.4 -** Identify rare saturated bins
    - [ ] **6.4.5 -** Use this analysis to motivate class rebalancing in the report

- [ ] **Phase 7 - Classification Model Without Rebalancing**
  - [ ] **7.1 - Define Classification Formulation**
    - [ ] **7.1.1 -** Set input tensor shape to `[B, 1, H, W]`
    - [ ] **7.1.2 -** Set output tensor shape to `[B, K, H, W]`
    - [ ] **7.1.3 -** Set target tensor shape to `[B, H, W]`
    - [ ] **7.1.4 -** Use cross-entropy loss without class weights
    - [ ] **7.1.5 -** Keep architecture comparable to regression model
  - [ ] **7.2 - Implement Classification Dataset**
    - [ ] **7.2.1 -** Return `(L, bin_target, ab)` for training and evaluation
    - [ ] **7.2.2 -** Verify class-target shapes
    - [ ] **7.2.3 -** Verify class-target data type is integer
    - [ ] **7.2.4 -** Verify batch loading
    - [ ] **7.2.5 -** Verify decoded ground-truth bin images
  - [ ] **7.3 - Implement Classification Model**
    - [ ] **7.3.1 -** Reuse the shared backbone
    - [ ] **7.3.2 -** Replace the regression head with a `K`-channel logits head
    - [ ] **7.3.3 -** Do not apply softmax inside the model
    - [ ] **7.3.4 -** Verify logits shape with a dummy batch
    - [ ] **7.3.5 -** Verify one training step runs without errors
  - [ ] **7.4 - Train Classification Model**
    - [ ] **7.4.1 -** Train with unweighted cross-entropy
    - [ ] **7.4.2 -** Validate after every epoch
    - [ ] **7.4.3 -** Save best checkpoint
    - [ ] **7.4.4 -** Save decoded qualitative predictions
    - [ ] **7.4.5 -** Log classification loss and decoded image metrics
  - [ ] **7.5 - Evaluate Classification Model**
    - [ ] **7.5.1 -** Decode predictions to `ab`
    - [ ] **7.5.2 -** Compute `ab` MAE or RMSE
    - [ ] **7.5.3 -** Compute RGB PSNR
    - [ ] **7.5.4 -** Compute RGB SSIM
    - [ ] **7.5.5 -** Compute mean chroma or colorfulness
    - [ ] **7.5.6 -** Compare outputs visually against regression baseline

- [ ] **Phase 8 - Classification Model With Class Rebalancing**
  - [ ] **8.1 - Compute Class Weights**
    - [ ] **8.1.1 -** Count bin frequencies on the training split only
    - [ ] **8.1.2 -** Convert frequencies to inverse-frequency weights
    - [ ] **8.1.3 -** Add smoothing to avoid extreme weights
    - [ ] **8.1.4 -** Clip weights if needed for stability
    - [ ] **8.1.5 -** Normalize weights to mean 1
    - [ ] **8.1.6 -** Save class weights to disk
  - [ ] **8.2 - Implement Weighted Loss**
    - [ ] **8.2.1 -** Load class weights into the training script
    - [ ] **8.2.2 -** Pass class weights to cross-entropy loss
    - [ ] **8.2.3 -** Verify that weights are on the correct device
    - [ ] **8.2.4 -** Verify that training loss remains finite
    - [ ] **8.2.5 -** Log the exact weighting formula for the report
  - [ ] **8.3 - Train Rebalanced Classification Model**
    - [ ] **8.3.1 -** Use the same architecture as unweighted classification
    - [ ] **8.3.2 -** Use the same dataset split
    - [ ] **8.3.3 -** Use the same number of bins
    - [ ] **8.3.4 -** Train with weighted cross-entropy
    - [ ] **8.3.5 -** Save best checkpoint
    - [ ] **8.3.6 -** Save decoded qualitative predictions
  - [ ] **8.4 - Evaluate Rebalanced Classification Model**
    - [ ] **8.4.1 -** Compute the same metrics as other variants
    - [ ] **8.4.2 -** Compare mean chroma against unweighted classification
    - [ ] **8.4.3 -** Compare predicted bin-frequency distribution against training distribution
    - [ ] **8.4.4 -** Compare qualitative vividness against regression
    - [ ] **8.4.5 -** Identify cases where rebalancing improves results
    - [ ] **8.4.6 -** Identify cases where rebalancing introduces wrong colors

- [ ] **Phase 9 - Optional GAN Extension**
  - [ ] **9.1 - Decide Whether GAN Extension Is Feasible**
    - [ ] **9.1.1 -** Confirm that all mandatory variants are complete
    - [ ] **9.1.2 -** Confirm that evaluation scripts are complete
    - [ ] **9.1.3 -** Confirm that enough time remains for unstable training
    - [ ] **9.1.4 -** Decide whether GAN results will strengthen or distract from the final report
  - [ ] **9.2 - Implement Conditional Generator**
    - [ ] **9.2.1 -** Reuse the regression U-Net as generator
    - [ ] **9.2.2 -** Use `L` as input
    - [ ] **9.2.3 -** Predict continuous `ab`
    - [ ] **9.2.4 -** Initialize from regression checkpoint if useful
    - [ ] **9.2.5 -** Verify generated RGB images after inverse Lab conversion
  - [ ] **9.3 - Implement Conditional Discriminator**
    - [ ] **9.3.1 -** Concatenate `L` and real `ab` for real examples
    - [ ] **9.3.2 -** Concatenate `L` and generated `ab` for fake examples
    - [ ] **9.3.3 -** Use a PatchGAN-style convolutional discriminator
    - [ ] **9.3.4 -** Output a patch-level real/fake map
    - [ ] **9.3.5 -** Verify discriminator output shape
  - [ ] **9.4 - Train GAN**
    - [ ] **9.4.1 -** Define discriminator adversarial loss
    - [ ] **9.4.2 -** Define generator adversarial loss
    - [ ] **9.4.3 -** Add L1 reconstruction loss to generator objective
    - [ ] **9.4.4 -** Tune reconstruction weight if needed
    - [ ] **9.4.5 -** Save qualitative samples frequently
    - [ ] **9.4.6 -** Stop GAN work if training becomes too unstable for the deadline
  - [ ] **9.5 - Evaluate GAN**
    - [ ] **9.5.1 -** Compare GAN outputs against regression outputs
    - [ ] **9.5.2 -** Compute the same quantitative metrics where applicable
    - [ ] **9.5.3 -** Discuss perceptual quality separately from pixel accuracy
    - [ ] **9.5.4 -** Include GAN only as an extension in the report if results are coherent

- [ ] **Phase 10 - Comparative Experiments and Ablations**
  - [ ] **10.1 - Define Final Experiment Matrix**
    - [ ] **10.1.1 -** Include regression baseline
    - [ ] **10.1.2 -** Include classification without rebalancing
    - [ ] **10.1.3 -** Include classification with rebalancing
    - [ ] **10.1.4 -** Include optional GAN only if complete
    - [ ] **10.1.5 -** Use the same dataset split for all comparisons
  - [ ] **10.2 - Run Mandatory Experiments**
    - [ ] **10.2.1 -** Train final regression model
    - [ ] **10.2.2 -** Train final unweighted classification model
    - [ ] **10.2.3 -** Train final rebalanced classification model
    - [ ] **10.2.4 -** Evaluate all models on validation set
    - [ ] **10.2.5 -** Evaluate all models on test set
  - [ ] **10.3 - Run Focused Ablations**
    - [ ] **10.3.1 -** Test at least one alternative number of color bins if time allows
    - [ ] **10.3.2 -** Test argmax versus annealed-mean decoding if implemented
    - [ ] **10.3.3 -** Test U-Net skip connections versus no skip connections if time allows
    - [ ] **10.3.4 -** Test L1 versus L2 regression if time allows
    - [ ] **10.3.5 -** Keep only meaningful ablations in the final report
  - [ ] **10.4 - Create Final Tables**
    - [ ] **10.4.1 -** Create a metrics table with one row per model
    - [ ] **10.4.2 -** Include `ab` MAE or RMSE
    - [ ] **10.4.3 -** Include RGB PSNR
    - [ ] **10.4.4 -** Include RGB SSIM
    - [ ] **10.4.5 -** Include mean chroma or colorfulness
    - [ ] **10.4.6 -** Add brief interpretation notes for each metric
  - [ ] **10.5 - Create Final Figures**
    - [ ] **10.5.1 -** Create training-loss curves
    - [ ] **10.5.2 -** Create validation-metric curves
    - [ ] **10.5.3 -** Create qualitative comparison grid
    - [ ] **10.5.4 -** Create color-bin frequency plot
    - [ ] **10.5.5 -** Create failure-case figure
    - [ ] **10.5.6 -** Export figures in report-ready resolution

- [ ] **Phase 11 - Analysis and Interpretation**
  - [ ] **11.1 - Analyze Regression Results**
    - [ ] **11.1.1 -** Identify whether colors are desaturated
    - [ ] **11.1.2 -** Identify whether spatial structure is preserved
    - [ ] **11.1.3 -** Compare pixel metrics against classification models
    - [ ] **11.1.4 -** Explain results using the averaging effect of regression losses
  - [ ] **11.2 - Analyze Classification Results**
    - [ ] **11.2.1 -** Identify whether outputs are more vivid than regression
    - [ ] **11.2.2 -** Identify whether quantization artifacts are visible
    - [ ] **11.2.3 -** Compare argmax and annealed decoding if available
    - [ ] **11.2.4 -** Explain results using multimodal color prediction
  - [ ] **11.3 - Analyze Class Rebalancing Results**
    - [ ] **11.3.1 -** Compare chroma before and after rebalancing
    - [ ] **11.3.2 -** Compare rare-color prediction before and after rebalancing
    - [ ] **11.3.3 -** Identify examples where rebalancing improves saturated objects
    - [ ] **11.3.4 -** Identify examples where rebalancing creates unrealistic colors
    - [ ] **11.3.5 -** Explain the tradeoff between vividness and exact reconstruction
  - [ ] **11.4 - Analyze Failure Cases**
    - [ ] **11.4.1 -** Identify semantic ambiguity failures
    - [ ] **11.4.2 -** Identify long-range consistency failures
    - [ ] **11.4.3 -** Identify dataset-bias failures
    - [ ] **11.4.4 -** Identify low-texture region failures
    - [ ] **11.4.5 -** Connect failure cases to limitations discussed in the papers

- [ ] **Phase 12 - Final Report**
  - [ ] **12.1 - Draft Abstract**
    - [ ] **12.1.1 -** State the task
    - [ ] **12.1.2 -** State the implemented variants
    - [ ] **12.1.3 -** State the dataset
    - [ ] **12.1.4 -** State the main result
    - [ ] **12.1.5 -** Keep abstract under 300 words
  - [ ] **12.2 - Draft Introduction**
    - [ ] **12.2.1 -** Motivate grayscale image colorization
    - [ ] **12.2.2 -** Explain why the problem is ill-posed
    - [ ] **12.2.3 -** Explain why regression can be insufficient
    - [ ] **12.2.4 -** Preview the regression, classification, and rebalancing comparison
    - [ ] **12.2.5 -** Summarize the main findings
  - [ ] **12.3 - Draft Related Work**
    - [ ] **12.3.1 -** Summarize Cheng et al.
    - [ ] **12.3.2 -** Summarize Zhang et al.
    - [ ] **12.3.3 -** Summarize Nazeri et al.
    - [ ] **12.3.4 -** Explain how this project differs from full paper reproductions
    - [ ] **12.3.5 -** Include proper citations
  - [ ] **12.4 - Draft Data Section**
    - [ ] **12.4.1 -** Describe dataset source
    - [ ] **12.4.2 -** Describe train, validation, and test splits
    - [ ] **12.4.3 -** Describe image resolution
    - [ ] **12.4.4 -** Describe Lab conversion
    - [ ] **12.4.5 -** Describe normalization
  - [ ] **12.5 - Draft Methods Section**
    - [ ] **12.5.1 -** Describe shared CNN/U-Net architecture
    - [ ] **12.5.2 -** Describe regression objective
    - [ ] **12.5.3 -** Describe color quantization
    - [ ] **12.5.4 -** Describe classification objective
    - [ ] **12.5.5 -** Describe class rebalancing formula
    - [ ] **12.5.6 -** Describe decoding from bins to `ab`
  - [ ] **12.6 - Draft Experiments Section**
    - [ ] **12.6.1 -** Describe training hyperparameters
    - [ ] **12.6.2 -** Present final quantitative table
    - [ ] **12.6.3 -** Present qualitative comparison grid
    - [ ] **12.6.4 -** Present class rebalancing ablation
    - [ ] **12.6.5 -** Present failure-case analysis
  - [ ] **12.7 - Draft Conclusion**
    - [ ] **12.7.1 -** Summarize main experimental conclusion
    - [ ] **12.7.2 -** State limitations
    - [ ] **12.7.3 -** Suggest future work such as larger datasets, semantic conditioning, or GANs
  - [ ] **12.8 - Finalize Report**
    - [ ] **12.8.1 -** Enforce 4 to 6 page limit
    - [ ] **12.8.2 -** Check IEEE formatting
    - [ ] **12.8.3 -** Check figure readability
    - [ ] **12.8.4 -** Check citation formatting
    - [ ] **12.8.5 -** Proofread for clarity
    - [ ] **12.8.6 -** Export final PDF

- [ ] **Phase 13 - Oral Presentation and Defense**
  - [ ] **13.1 - Create Presentation Structure**
    - [ ] **13.1.1 -** Allocate 15 to 20 minutes total
    - [ ] **13.1.2 -** Introduce the problem and motivation
    - [ ] **13.1.3 -** Explain the three papers briefly
    - [ ] **13.1.4 -** Explain the implemented models
    - [ ] **13.1.5 -** Present experimental results
    - [ ] **13.1.6 -** Discuss conclusions and limitations
  - [ ] **13.2 - Prepare Visual Slides**
    - [ ] **13.2.1 -** Include Lab-space explanation figure
    - [ ] **13.2.2 -** Include model architecture diagram
    - [ ] **13.2.3 -** Include color-bin quantization diagram
    - [ ] **13.2.4 -** Include class rebalancing histogram
    - [ ] **13.2.5 -** Include qualitative result grid
    - [ ] **13.2.6 -** Include final metrics table
  - [ ] **13.3 - Assign Speaking Roles**
    - [ ] **13.3.1 -** Assign introduction and motivation speaker
    - [ ] **13.3.2 -** Assign methods speaker
    - [ ] **13.3.3 -** Assign experiments and results speaker
    - [ ] **13.3.4 -** Ensure every member can answer questions outside their speaking section
  - [ ] **13.4 - Prepare Q&A**
    - [ ] **13.4.1 -** Prepare answer for why Lab space is used
    - [ ] **13.4.2 -** Prepare answer for why regression gives dull colors
    - [ ] **13.4.3 -** Prepare answer for why classification handles ambiguity better
    - [ ] **13.4.4 -** Prepare answer for how class rebalancing works
    - [ ] **13.4.5 -** Prepare answer for why quantitative metrics may not match perceptual quality
    - [ ] **13.4.6 -** Prepare answer for why GANs were or were not implemented
  - [ ] **13.5 - Rehearse Presentation**
    - [ ] **13.5.1 -** Run one full timed rehearsal
    - [ ] **13.5.2 -** Check transitions between speakers
    - [ ] **13.5.3 -** Remove low-value slides if over time
    - [ ] **13.5.4 -** Practice explaining equations clearly
    - [ ] **13.5.5 -** Practice defending experimental choices

- [ ] **Phase 14 - Packaging and Submission**
  - [ ] **14.1 - Clean Codebase**
    - [ ] **14.1.1 -** Remove unused scripts
    - [ ] **14.1.2 -** Remove temporary debug outputs
    - [ ] **14.1.3 -** Ensure paths are relative or documented
    - [ ] **14.1.4 -** Ensure external code inspiration is acknowledged
    - [ ] **14.1.5 -** Ensure code runs from a fresh checkout or clean folder
  - [ ] **14.2 - Create README**
    - [ ] **14.2.1 -** Describe project purpose
    - [ ] **14.2.2 -** Describe environment setup
    - [ ] **14.2.3 -** Describe dataset placement
    - [ ] **14.2.4 -** Describe how to train each model
    - [ ] **14.2.5 -** Describe how to evaluate each model
    - [ ] **14.2.6 -** Describe how to reproduce figures
  - [ ] **14.3 - Prepare Submission ZIP**
    - [ ] **14.3.1 -** Include final PDF report
    - [ ] **14.3.2 -** Include source code
    - [ ] **14.3.3 -** Include README.txt
    - [ ] **14.3.4 -** Include requirements file
    - [ ] **14.3.5 -** Include small sample outputs if allowed
    - [ ] **14.3.6 -** Exclude large datasets and checkpoints unless explicitly required
  - [ ] **14.4 - Final Submission Check**
    - [ ] **14.4.1 -** Open the ZIP and verify contents
    - [ ] **14.4.2 -** Verify report PDF opens correctly
    - [ ] **14.4.3 -** Verify README instructions are complete
    - [ ] **14.4.4 -** Verify all group member names are included
    - [ ] **14.4.5 -** Verify submission email addresses
    - [ ] **14.4.6 -** Submit before the deadline

- [ ] **Phase 15 - Final Risk Review**
  - [ ] **15.1 - Technical Risks**
    - [ ] **15.1.1 -** If training is too slow, reduce image size
    - [ ] **15.1.2 -** If classification is too hard, reduce number of bins
    - [ ] **15.1.3 -** If rebalancing is unstable, clip class weights
    - [ ] **15.1.4 -** If metrics look worse for vivid models, emphasize multimodality and qualitative analysis
    - [ ] **15.1.5 -** If GAN is unstable, remove it from main claims
  - [ ] **15.2 - Report Risks**
    - [ ] **15.2.1 -** If report exceeds page limit, reduce related work detail
    - [ ] **15.2.2 -** If figures are unreadable, simplify grids
    - [ ] **15.2.3 -** If methods section is too long, move hyperparameters to a compact table
    - [ ] **15.2.4 -** If experiments section is weak, prioritize mandatory comparisons over optional work
  - [ ] **15.3 - Presentation Risks**
    - [ ] **15.3.1 -** If presentation is too long, reduce paper summaries
    - [ ] **15.3.2 -** If Q&A preparation is weak, rehearse key theory questions
    - [ ] **15.3.3 -** If one member knows too little, redistribute explanation responsibilities
    - [ ] **15.3.4 -** If results are imperfect, clearly explain limitations and what was learned
