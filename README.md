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

# Implementation Plan: Grayscale Image Colorization

## 1. Project Positioning

The project should be framed as automatic grayscale image colorization: given only the luminance information of an image, the model must predict plausible chrominance values. The key point from the project statement and the papers is that this is not a deterministic reconstruction problem. Many valid color images can share the same grayscale image, so a successful project should show how different learning formulations handle this ambiguity.

The project statement explicitly expects:

- A grayscale/color image dataset from ImageNet, BSDS, or another public image source.
- Conversion to CIE Lab color space.
- A CNN that predicts chrominance from luminance.
- At least two formulation variants, especially regression versus classification.
- Experiments studying class rebalancing and architectural choices.
- Qualitative and quantitative evaluation.

Therefore, the core project should not be centered only on a GAN. The strongest and safest structure is:

1. Implement a direct regression baseline.
2. Implement a classification-based colorization model inspired by Zhang, Isola, and Efros.
3. Add class rebalancing to the classification model.
4. Optionally add a conditional GAN if the required models and experiments are already complete.

This directly matches the project statement and gives a defensible experimental story.

## 2. Paper-Grounded Design Choices

### 2.1 Zhang et al., Colorful Image Colorization

This should be the main technical reference. Its central claim is that colorization is multimodal, so predicting a single continuous color with an L2-style loss tends to produce averaged, desaturated colors. The paper addresses this by turning colorization into per-pixel classification over quantized `ab` color bins in Lab space.

Ideas to adopt:

- Use CIE Lab color space.
- Use `L` as input and `ab` as prediction target.
- Compare regression with classification.
- Quantize `ab` into color bins.
- Use soft or hard color-bin labels.
- Apply class rebalancing to rare saturated colors.
- Decode predicted color distributions back into `ab`.
- Evaluate not only pixel accuracy, but also visual realism and color vividness.

For the course project, we do not need to reproduce the full 313-bin ImageNet-scale setup. A simplified implementation is acceptable if clearly justified and experimentally evaluated.

### 2.2 Cheng et al., Deep Colorization

This paper is useful mainly for motivation and historical context. It shows the transition from hand-engineered descriptors, semantic features, clustering, and post-processing toward learned automatic colorization.

Ideas to discuss, but probably not implement:

- Colorization benefits from semantic and contextual information.
- Local grayscale patches alone are ambiguous.
- Post-processing can improve spatial smoothness.
- Scene context can reduce ambiguity.

We should not implement the full Cheng pipeline because it relies on hand-designed descriptors, scene parsing, adaptive clustering, and filtering. That would distract from the course requirement to implement deep learning variants.

### 2.3 Nazeri et al., Image Colorization using GANs

This paper is useful as an optional extension. It shows that adversarial training can make outputs more visually realistic than pure reconstruction losses, but GAN training is harder to stabilize and evaluate.

Ideas to adopt only after the required models work:

- Use a U-Net-like generator.
- Use a conditional discriminator that sees both the grayscale input and either the real or generated color image.
- Combine adversarial loss with an L1 reconstruction loss.
- Compare whether adversarial training improves color vividness at the cost of stability.

The GAN variant should be positioned as an extension, not the core deliverable, because the project statement specifically asks for regression versus classification and class rebalancing.

## 3. Proposed Scope

### Minimum Complete Project

The minimum complete project should include:

- Dataset loader.
- Lab conversion pipeline.
- Regression baseline.
- Classification model without class rebalancing.
- Classification model with class rebalancing.
- Training scripts.
- Evaluation scripts.
- Qualitative result grids.
- Quantitative comparison table.
- Short ablation on the effect of class rebalancing.

This is enough to satisfy the official expectations.

### Strong Project

A stronger version should add:

- Soft encoding of `ab` labels rather than hard bin labels.
- Annealed-mean decoding instead of simple argmax decoding.
- A small architecture comparison, such as plain encoder-decoder versus U-Net.
- Colorfulness or mean chroma metric.
- Failure-case analysis.

### Optional Extension

Only if the required work is complete:

- Conditional GAN trained on the regression target.
- Comparison against regression baseline.
- Short discussion of training instability and perceptual quality.

## 4. Dataset Plan

### 4.1 Dataset Choice

Use either:

- BSDS if we want fast experiments and simple setup.
- ImageNet subset if we want a more representative dataset.

Recommended practical choice:

- Start with BSDS or a small ImageNet subset for development.
- Keep the code dataset-agnostic so a larger image folder can be used later.

The project statement allows ImageNet or Berkeley Segmentation Dataset. BSDS is smaller, which makes it suitable for a course project and for running several controlled variants.

### 4.2 Splits

Use fixed splits:

- Training set: 80%.
- Validation set: 10%.
- Test set: 10%.

If using a predefined dataset split, preserve it. Save split filenames to text files so every model uses the same images.

### 4.3 Preprocessing

For each RGB image:

1. Load image as RGB.
2. Resize or center-crop to a fixed resolution, for example `128 x 128` or `256 x 256`.
3. Convert RGB to Lab.
4. Normalize:
   - `L`: scale from `[0, 100]` to `[-1, 1]` or `[0, 1]`.
   - `a,b`: scale from approximately `[-128, 127]` to `[-1, 1]`.
5. Return:
   - Input: tensor of shape `[1, H, W]`.
   - Regression target: tensor of shape `[2, H, W]`.
   - Classification target: tensor of shape `[H, W]` containing color-bin IDs.

All model variants should use the same preprocessing, except for the target representation.

## 5. Color Space and Target Representation

### 5.1 Why Lab Space

Lab separates luminance from chrominance:

- `L` contains grayscale/lightness information.
- `a` and `b` contain color information.

This makes it natural to use `L` as the input and predict only `ab`. It also follows the project statement and Zhang et al.

### 5.2 Regression Target

The regression baseline predicts two continuous channels:

```text
input:  L       shape [B, 1, H, W]
output: ab_hat  shape [B, 2, H, W]
target: ab      shape [B, 2, H, W]
```

Recommended loss:

```text
L_reg = mean absolute error(ab_hat, ab)
```

L1 is preferred over L2 because it is less sensitive to outliers and often gives sharper outputs. We can still mention that L2-style regression is the classical dull-color baseline discussed by Zhang et al.

### 5.3 Classification Target

The classification model predicts a distribution over quantized `ab` bins:

```text
input:  L            shape [B, 1, H, W]
output: logits       shape [B, K, H, W]
target: bin indices  shape [B, H, W]
```

Here `K` is the number of color bins.

A full Zhang-style implementation uses 313 in-gamut bins. For this project, a simpler scheme is acceptable:

- Build a regular grid over `a,b`, for example 10 x 10 or 16 x 16.
- Remove bins that are never or rarely used.
- Map every pixel's `ab` value to the nearest valid bin.

Recommended starting point:

- `K` between 64 and 128 bins.
- Ignore extremely rare bins or merge them into nearby bins.

This keeps the classification problem manageable on a small dataset.

### 5.4 Soft Encoding

A stronger implementation should use soft labels:

- For each ground-truth `ab` value, find the nearest `k` color bins.
- Assign weights based on distance in `ab` space.
- Train with soft cross-entropy.

If time is limited, use hard bin labels first. Soft encoding can be added later as an improvement.

### 5.5 Decoding

Classification outputs must be converted back to continuous `ab`.

Implement at least:

- Argmax decoding: choose the most probable bin.

If time allows, add:

- Annealed-mean decoding inspired by Zhang et al.

Annealed mean:

1. Take predicted probabilities over bins.
2. Sharpen the distribution with temperature `T < 1`.
3. Compute the weighted average of bin centers.

This is a good compromise between mean decoding, which can be dull, and argmax decoding, which can be unstable.

## 6. Model Architecture Plan

### 6.1 Shared Backbone

Use one shared CNN architecture family for both regression and classification so the comparison focuses on the prediction formulation rather than unrelated architectural differences.

Recommended architecture:

- U-Net-style encoder-decoder.
- Input channel: 1.
- Encoder: convolution blocks with downsampling.
- Decoder: upsampling blocks.
- Skip connections from encoder to decoder.
- Output head changes depending on formulation:
  - Regression head: 2 output channels with `tanh`.
  - Classification head: `K` output channels with raw logits.

This architecture is simple, familiar from image-to-image tasks, and consistent with the GAN paper's generator design.

### 6.2 Regression Model

Output:

```text
[B, 2, H, W]
```

Final activation:

- `tanh` if `ab` is normalized to `[-1, 1]`.
- No activation if using unnormalized Lab values, though normalized targets are cleaner.

Loss:

- Main: L1 loss.
- Optional: L2 loss comparison if time allows.

### 6.3 Classification Model

Output:

```text
[B, K, H, W]
```

Final activation:

- No softmax inside the model.
- Use cross-entropy loss, which applies log-softmax internally.

Loss:

- Cross-entropy without rebalancing.
- Cross-entropy with class weights.

### 6.4 Optional GAN Model

Only after required variants are complete:

Generator:

- Reuse the regression U-Net.
- Input `L`, output `ab`.

Discriminator:

- Conditional PatchGAN-style discriminator.
- Input concatenation: `[L, ab]`, so 3 channels total.
- Output patch-level real/fake map.

Loss:

```text
L_G = L_adv + lambda * L1(ab_hat, ab)
L_D = real/fake adversarial loss
```

Use `lambda = 100` as in the GAN paper unless experiments suggest otherwise.

## 7. Class Rebalancing Plan

Class rebalancing is central to the project. The reason is that natural images contain many low-saturation pixels. Without rebalancing, the model learns that grayish colors are usually safe.

Implementation:

1. Before training, scan the training set.
2. Convert each image to Lab.
3. Quantize each pixel's `ab` value to a bin.
4. Count bin frequencies.
5. Convert frequencies to class weights.
6. Use those weights in the classification loss.

Simple weighting:

```text
weight_c = 1 / (frequency_c + epsilon)
```

Better weighting:

```text
weight_c = 1 / ((1 - lambda) * frequency_c + lambda / K)
```

Then normalize weights so their mean is 1.

Recommended:

- Start with the simple weighting.
- Clip very large weights to avoid unstable training.
- Report the exact weighting formula in the Methods section.

Experiments should compare:

- Classification without class weights.
- Classification with class weights.

Expected outcome:

- Rebalancing should increase color vividness and rare-color prediction.
- It may reduce pixel-level accuracy because the model is less biased toward safe average colors.

## 8. Training Plan

### 8.1 Training Setup

Use PyTorch.

Recommended defaults:

- Image size: `128 x 128` for fast iteration, `256 x 256` if hardware allows.
- Batch size: 8 to 32 depending on GPU memory.
- Optimizer: Adam.
- Learning rate: `1e-4` or `2e-4`.
- Epochs: enough for convergence on the selected dataset, likely 30 to 100 for small datasets.
- Validation after every epoch.
- Save best checkpoint by validation loss.

### 8.2 Training Scripts

The codebase should have separate, simple scripts:

```text
src/
  data.py
  color_bins.py
  models.py
  losses.py
  metrics.py
  train_regression.py
  train_classification.py
  evaluate.py
  visualize.py
```

Each script should be runnable from the command line with clear arguments:

```text
python src/train_regression.py --data data/bsds --image-size 128 --epochs 50
python src/train_classification.py --data data/bsds --bins 96 --rebalance
python src/evaluate.py --checkpoint checkpoints/model.pt --model classification
```

### 8.3 Reproducibility

Set and report:

- Random seed.
- Dataset split.
- Image size.
- Batch size.
- Learning rate.
- Number of epochs.
- Number of bins.
- Rebalancing formula.

Save:

- Model checkpoints.
- Training curves.
- Metrics CSV files.
- Qualitative image grids.

## 9. Evaluation Plan

### 9.1 Quantitative Metrics

Use several metrics because no single metric fully captures colorization quality.

Recommended metrics:

- MAE in `ab` space.
- RMSE in `ab` space.
- PSNR in RGB space.
- SSIM in RGB space.
- Mean chroma:

```text
chroma = sqrt(a^2 + b^2)
```

- Optional colorfulness metric.

Interpretation:

- Regression may score well on MAE/PSNR because it predicts safe average colors.
- Classification with rebalancing may produce more vivid images but not always better pixel accuracy.
- Colorfulness/chroma helps show whether the model avoids desaturated predictions.

### 9.2 Qualitative Evaluation

For a fixed validation/test subset, create grids:

```text
grayscale input | ground truth | regression | classification | classification + rebalancing
```

Choose examples that show:

- Natural outdoor scenes.
- Indoor scenes.
- Objects with ambiguous colors.
- Highly saturated objects.
- Failure cases.

Qualitative results are important because colorization is perceptual and multimodal.

### 9.3 Ablations

Minimum ablations:

1. Regression versus classification.
2. Classification without rebalancing versus with rebalancing.

Strong ablations:

1. Number of color bins: for example 64 versus 128.
2. Argmax versus annealed-mean decoding.
3. Plain encoder-decoder versus U-Net skip connections.
4. L1 versus L2 regression.

The report does not need too many ablations. It needs a coherent set that supports the main claim.

## 10. Expected Experimental Story

The final report should aim to show the following:

1. Direct regression is simple and stable, but tends to produce dull or averaged colors.
2. Classification better matches the multimodal nature of colorization.
3. Class rebalancing increases color vividness by correcting the dominance of low-saturation bins.
4. There is a tradeoff between pixel-level reconstruction accuracy and perceptual plausibility.

This story is directly grounded in Zhang et al. and aligns with the project statement.

## 11. Suggested Milestones

### Milestone 1: Literature and Setup

Deliverables:

- Short notes on the three papers.
- Dataset selected and downloaded.
- Dataset splits fixed.
- Lab conversion verified visually.

Success check:

- A script can display RGB image, grayscale `L`, and reconstructed RGB from original `L,ab`.

### Milestone 2: Regression Baseline

Deliverables:

- PyTorch dataset returning `(L, ab)`.
- U-Net or encoder-decoder regression model.
- Training loop.
- Validation loop.
- First qualitative predictions.

Success check:

- Loss decreases.
- Reconstructed validation images have plausible but likely muted colors.

### Milestone 3: Color Quantization

Deliverables:

- Color-bin creation.
- Pixel-to-bin encoding.
- Bin-to-`ab` decoding.
- Visualization of bin distribution.

Success check:

- A ground-truth image can be quantized and decoded with acceptable visual degradation.

### Milestone 4: Classification Model

Deliverables:

- Classification output head.
- Cross-entropy training.
- Argmax decoding.
- Quantitative and qualitative evaluation.

Success check:

- Classification model trains without shape errors.
- Decoded outputs are visually coherent.

### Milestone 5: Class Rebalancing

Deliverables:

- Training-set bin-frequency estimator.
- Class-weight computation.
- Weighted classification training.
- Comparison table and image grids.

Success check:

- Rebalanced model produces higher chroma or visibly more saturated colors than unweighted classification.

### Milestone 6: Report and Presentation

Deliverables:

- Final metrics table.
- Training curves.
- Qualitative result figure.
- Failure-case figure.
- 4 to 6 page IEEE-style report.
- 15 to 20 minute group presentation.

Success check:

- The report can explain why each model was implemented and what the experiments show.

## 12. Division of Work for Three Members

### Member 1: Data and Evaluation

Responsibilities:

- Dataset loading and preprocessing.
- Lab conversion utilities.
- Color-bin frequency analysis.
- Metrics implementation.
- Visualization grids.

### Member 2: Regression and Architecture

Responsibilities:

- Regression model.
- U-Net or encoder-decoder architecture.
- Regression training loop.
- Architecture ablation if time allows.

### Member 3: Classification and Rebalancing

Responsibilities:

- Color quantization.
- Classification model head.
- Cross-entropy training.
- Class rebalancing.
- Decoding methods.

All members should understand all models before the oral defense. The oral score is individual, so no one should be isolated to only one small script.

## 13. Report Structure

Use the required IEEE-style 4 to 6 page structure.

### Abstract

State the problem, the three implemented variants, and the main quantitative and qualitative finding.

### Introduction

Explain why grayscale colorization is ill-posed and multimodal. Mention historical grayscale imagery and self-supervised learning motivation briefly.

### Related Work

Discuss:

- Cheng et al. as an early automatic deep colorization pipeline.
- Zhang et al. as the main classification and rebalancing reference.
- Nazeri et al. as the GAN-based perceptual realism reference.

### Data

Describe:

- Dataset source.
- Number of images.
- Train/validation/test split.
- Image resolution.
- Lab conversion.
- Normalization.

### Methods

Describe:

- Shared CNN/U-Net backbone.
- Regression formulation.
- Classification formulation.
- Color quantization.
- Class rebalancing formula.
- Decoding method.

### Experiments

Include:

- Metrics table.
- Training curves.
- Qualitative comparison grid.
- Ablation on rebalancing.
- Short failure-case analysis.

### Conclusion

Summarize:

- Regression is stable but dull.
- Classification handles ambiguity better.
- Rebalancing improves vividness.
- Future work could include GANs, larger datasets, or semantic conditioning.

## 14. Risks and Mitigations

### Risk: Dataset Too Small

Mitigation:

- Use data augmentation.
- Keep claims modest.
- Focus on controlled comparison rather than state-of-the-art results.

### Risk: Classification Has Too Many Classes

Mitigation:

- Use fewer bins.
- Remove unused bins.
- Start with hard labels before soft encoding.

### Risk: Rebalancing Makes Training Unstable

Mitigation:

- Clip class weights.
- Normalize weights to mean 1.
- Lower learning rate.

### Risk: Quantitative Metrics Favor Dull Regression

Mitigation:

- Report colorfulness/chroma.
- Include qualitative grids.
- Explain multimodality and the difference between exact reconstruction and plausible colorization.

### Risk: GAN Takes Too Long

Mitigation:

- Treat GAN as optional.
- Do not start GAN until regression, classification, and rebalancing experiments are complete.

## 15. Final Recommendation

The project should be built around the comparison:

```text
Regression baseline
vs.
Classification without class rebalancing
vs.
Classification with class rebalancing
```

This is the best match to the official project statement and the strongest lesson from the papers. The GAN paper should be used mainly in related work and only implemented as an optional extension. A clear, well-tested implementation of the three required variants will make for a stronger report and oral defense than an unstable GAN-centered project.

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
