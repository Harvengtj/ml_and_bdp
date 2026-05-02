# Machine Learning and Big Data Processing

Repository for the project **Grayscale Image Colorization**.

# GAN-Based Grayscale Image Colorization

## Project Goal

This project focuses on automatic grayscale image colorization using a conditional Generative Adversarial Network.

The model receives the luminance channel of an image and predicts plausible chrominance channels:

```text
input:  L channel
output: predicted ab channels
```

The final color image is reconstructed by combining:

```text
L + predicted ab -> Lab -> RGB
```

The project uses CIE Lab color space because it separates lightness from color:

- `L`: luminance / grayscale information
- `a`: green-red chrominance
- `b`: blue-yellow chrominance

## Current Group Decision

The group decided to prioritize a GAN-based implementation.

The main implementation direction is:

```text
Conditional GAN with U-Net generator
```

The generator receives `L` and predicts `ab`.

The discriminator receives either a real colorization pair or a generated colorization pair:

```text
real pair: concat(L, real ab)
fake pair: concat(L, generated ab)
```

The discriminator then predicts whether the pair is real or fake. A PatchGAN-style discriminator can also output a real/fake map instead of one single probability.

## Paper Roles

### Main Implementation Reference

**Image Colorization using Generative Adversarial Networks**

Used for:

- conditional GAN formulation
- U-Net generator idea
- encoder-decoder structure with skip connections
- adversarial training
- reconstruction plus adversarial loss
- TensorFlow reference ideas to reimplement in PyTorch

### Supporting References

**Colorful Image Colorization**

Used for:

- explaining why colorization is multimodal
- explaining why pure regression can produce dull colors
- classification and color-bin prediction as an alternative formulation
- class rebalancing as context

**Deep Colorization**

Used for:

- semantic-context discussion
- optional adaptive image clustering from section III.C

## Dataset

Current dataset choice:

```text
Berkeley Segmentation Dataset 500 (BSDS500)
```

Initial plan:

- use BSDS500 for fast controlled experiments
- preserve train, validation, and test splits when possible
- avoid using test images during training
- start with small image sizes such as `32 x 32` or `64 x 64`
- scale to `128 x 128` once the pipeline works
- try larger resolutions only if the model is stable and hardware allows it

If BSDS500 is too small, the project may later test a small ImageNet subset.

## Model Overview

### Generator

The generator is a U-Net-style model:

```text
L -> encoder -> bottleneck -> decoder with skip connections -> predicted ab
```

Expected shapes:

```text
input:  [B, 1, H, W]
output: [B, 2, H, W]
```

The final activation can be `tanh` if `ab` is normalized to `[-1, 1]`.

### Discriminator

The discriminator is conditional.

It receives:

```text
real pair: concat(L, real ab)
fake pair: concat(L, generated ab)
```

Expected input shape:

```text
[B, 3, H, W]
```

It outputs either:

- one real/fake probability
- or a PatchGAN-style real/fake map

## Training Objective

The generator is trained with:

```text
adversarial loss + reconstruction loss
```

Conceptually:

$$
L_G = L_{\text{adv}} + \lambda L_{\text{rec}}
$$

The reconstruction loss compares generated `ab` to real `ab`.

Recommended first choice:

```text
L1 loss
```

The discriminator is trained to classify:

```text
real L+ab pairs -> real
fake L+ab pairs -> fake
```

## Implementation Order

Recommended order:

1. Implement dataset loader.
2. Convert RGB images to Lab.
3. Extract `L` and `ab`.
4. Normalize channels.
5. Reconstruct RGB from original `L,ab` as a sanity check.
6. Implement U-Net generator.
7. Verify generator output shape.
8. Implement discriminator.
9. Verify discriminator input/output shape.
10. Run one-batch reconstruction-only training.
11. Run one discriminator step.
12. Run one full GAN training step.
13. Train at small resolution.
14. Save qualitative results.
15. Scale up resolution if stable.

## Evaluation

Use both qualitative and quantitative evaluation.

### Qualitative Evaluation

Create grids:

```text
grayscale input | ground truth RGB | generated RGB
```

Qualitative evaluation matters because colorization is ambiguous. Several colors can be valid for the same grayscale input.

### Quantitative Evaluation

Recommended metrics:

- `ab` MAE
- RGB PSNR
- RGB SSIM
- mean chroma / colorfulness
- generator and discriminator loss curves

These metrics should be interpreted carefully. A model can have good pixel-level metrics while still producing dull colors.

## Optional Extensions

Optional work if time allows:

- L1-only U-Net baseline without discriminator
- adaptive clustering from Deep Colorization section III.C
- small ImageNet subset if BSDS500 is insufficient
- higher-resolution training
- comparison against a simple regression baseline for report credibility

## Proposed Task Split Per Person

This split avoids having three people edit the same training code at the same time. Each person owns a coherent part of the project, but everyone should still understand the full pipeline for the oral defense.

### Person 1 - Data, Preprocessing, and Evaluation

Main responsibility:

- own the BSDS500 dataset setup
- preserve train, validation, and test splits
- implement image loading
- implement RGB-to-Lab conversion
- extract and normalize `L` and `ab`
- implement Lab-to-RGB reconstruction for visualization
- create preprocessing sanity-check figures
- implement evaluation metrics such as `ab` MAE, PSNR, SSIM, and mean chroma
- create final qualitative comparison grids

Expected deliverables:

- clean dataset loader
- preprocessing validation figures
- evaluation script
- final result grids and metric tables
- report text for the Data and Evaluation sections

### Person 2 - Generator, Discriminator, and Training Code

Main responsibility:

- implement the U-Net generator
- verify input shape `[B, 1, H, W]`
- verify output shape `[B, 2, H, W]`
- implement the conditional discriminator
- decide whether to use a single-output discriminator or PatchGAN-style discriminator
- implement reconstruction loss
- implement adversarial loss
- implement generator and discriminator optimizers
- implement one-batch debug training
- implement the full GAN training loop
- save checkpoints, losses, and generated samples

Expected deliverables:

- generator module
- discriminator module
- loss functions
- training script
- saved checkpoints and sample outputs
- report text for the Method and Training sections

### Person 3 - Literature, Experiments, Report, and Presentation

Main responsibility:

- summarize the GAN paper as the main implementation reference
- summarize Colorful Image Colorization as background for multimodality, regression, classification, and rebalancing
- summarize Deep Colorization section III.C as optional clustering context
- maintain the experiment log
- compare results across training settings
- collect failure cases
- write the Related Work, Experiments, Limitations, and Conclusion sections
- prepare the presentation structure
- prepare Q&A answers for the oral defense

Expected deliverables:

- related-work notes
- experiment log
- final report draft coordination
- presentation slides
- Q&A preparation notes

### Shared Responsibilities

- everyone reads the two main papers and Deep Colorization section III.C
- everyone understands CIE Lab color space
- everyone understands the generator-discriminator training loop
- everyone can explain why colorization is ambiguous
- everyone reviews final figures and report claims
- everyone rehearses the oral defense

## External References

- TensorFlow GAN baseline: <https://github.com/ImagingLab/Colorizing-with-GANs>
- Colorful Image Colorization reference: <https://github.com/richzhang/colorization?tab=readme-ov-file>
- IEEE final report templates: <https://www.ieee.org/conferences/publishing/templates>

## README Checklist

- [ ] Phase 1 - Project Scope and Alignment
  - [ ] 1.1 - Confirm the GAN-first direction
  - [ ] 1.2 - Define the task as conditional grayscale-to-color generation
  - [ ] 1.3 - Use CIE Lab with `L` input and `ab` output
  - [ ] 1.4 - Keep regression and classification papers as background references
  - [ ] 1.5 - Define success criteria for implementation, evaluation, and report

- [ ] Phase 2 - Literature Review
  - [ ] 2.1 - Study the GAN colorization paper
  - [ ] 2.2 - Extract the U-Net, discriminator, and loss ideas
  - [ ] 2.3 - Study Colorful Image Colorization for multimodality and classification context
  - [ ] 2.4 - Study Deep Colorization section III.C for optional clustering
  - [ ] 2.5 - Write the related-work summary

- [ ] Phase 3 - Repository and Environment Setup
  - [ ] 3.1 - Define the source-code structure
  - [ ] 3.2 - Confirm the Python environment
  - [ ] 3.3 - Install required packages
  - [ ] 3.4 - Define configs, checkpoints, figures, and result folders

- [ ] Phase 4 - Dataset and Preprocessing
  - [ ] 4.1 - Download and document BSDS500
  - [ ] 4.2 - Preserve train, validation, and test splits
  - [ ] 4.3 - Implement image loading and RGB-to-Lab conversion
  - [ ] 4.4 - Extract and normalize `L` and `ab`
  - [ ] 4.5 - Create preprocessing sanity-check figures

- [ ] Phase 5 - Generator Implementation
  - [ ] 5.1 - Implement the U-Net-style generator
  - [ ] 5.2 - Use one input channel for `L`
  - [ ] 5.3 - Use two output channels for `ab`
  - [ ] 5.4 - Add skip connections
  - [ ] 5.5 - Test generator shapes

- [ ] Phase 6 - Discriminator Implementation
  - [ ] 6.1 - Concatenate `L,ab` pairs
  - [ ] 6.2 - Implement the convolutional discriminator
  - [ ] 6.3 - Decide on single-output or PatchGAN output
  - [ ] 6.4 - Test discriminator shapes and loss compatibility

- [ ] Phase 7 - Losses and Optimizers
  - [ ] 7.1 - Implement reconstruction loss
  - [ ] 7.2 - Implement adversarial loss
  - [ ] 7.3 - Combine generator losses
  - [ ] 7.4 - Create Adam optimizers
  - [ ] 7.5 - Choose initial learning rates and loss weights

- [ ] Phase 8 - Training Pipeline
  - [ ] 8.1 - Run one-batch debug training
  - [ ] 8.2 - Implement discriminator update
  - [ ] 8.3 - Implement generator update
  - [ ] 8.4 - Implement full training loop
  - [ ] 8.5 - Save checkpoints, losses, and samples
  - [ ] 8.6 - Scale up only after the small-resolution pipeline works

- [ ] Phase 9 - Evaluation and Visualization
  - [ ] 9.1 - Generate input, ground-truth, and output grids
  - [ ] 9.2 - Compute quantitative metrics
  - [ ] 9.3 - Track loss curves
  - [ ] 9.4 - Compare results across settings
  - [ ] 9.5 - Analyze failure cases

- [ ] Phase 10 - Optional Extensions
  - [ ] 10.1 - Try an L1-only U-Net baseline
  - [ ] 10.2 - Try adaptive clustering
  - [ ] 10.3 - Try a small ImageNet subset
  - [ ] 10.4 - Try higher-resolution training

- [ ] Phase 11 - Report
  - [ ] 11.1 - Write dataset and preprocessing section
  - [ ] 11.2 - Write method section
  - [ ] 11.3 - Write experiments and evaluation section
  - [ ] 11.4 - Write related-work section
  - [ ] 11.5 - Write limitations and conclusion

- [ ] Phase 12 - Presentation
  - [ ] 12.1 - Prepare problem and Lab color space slides
  - [ ] 12.2 - Prepare GAN architecture and objective slides
  - [ ] 12.3 - Prepare result and limitation slides
  - [ ] 12.4 - Prepare Q&A answers

- [ ] Phase 13 - Packaging and Submission
  - [ ] 13.1 - Finalize README
  - [ ] 13.2 - Clean unused debug files
  - [ ] 13.3 - Verify clean setup execution
  - [ ] 13.4 - Verify final report, slides, code, and group information
  - [ ] 13.5 - Submit before the deadline
