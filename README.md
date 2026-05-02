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
- **Code:** <https://github.com/ImagingLab/Colorizing-with-GANs>

Used for:

- conditional GAN formulation
- U-Net generator idea
- encoder-decoder structure with skip connections
- adversarial training
- reconstruction plus adversarial loss
- TensorFlow reference ideas to reimplement in PyTorch

### Supporting References

**Colorful Image Colorization**
- **Code:** <http://richzhang.github.io/colorization/>

Used for:

- explaining why colorization is multimodal
- explaining why pure regression can produce dull colors
- classification and color-bin prediction as an alternative formulation
- class rebalancing as context

**Deep Colorization**
- **Code:** <http://www.cs.cityu.edu.hk/~qiyang/publications/iccv15/>

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

This split is only a coordination aid. Each person owns one area, but decisions should still come from the current code, results, and report needs.

### Person 1 - Data, Preprocessing, and Evaluation

Own the data path from raw images to usable training/evaluation batches.

Focus:

- dataset loading and Lab conversion
- sanity-check reconstruction figures
- metrics and qualitative result grids
- report material for data and evaluation

### Person 2 - Generator, Discriminator, and Training Code

Own the model and training path from tensors to trained outputs.

Focus:

- U-Net generator and conditional discriminator
- reconstruction and adversarial losses
- debug training, full training, checkpoints, and samples
- report material for method and training

### Person 3 - Literature, Experiments, Report, and Presentation

Own the explanation path from papers and runs to report and presentation.

Focus:

- related work and paper positioning
- experiment log, result comparison, and failure cases
- report coordination
- presentation and oral-defense preparation

### Shared Responsibilities

- understand the full `L -> ab -> RGB` pipeline
- understand the generator/discriminator training loop
- review final results, claims, and presentation story

## External References

- TensorFlow GAN baseline: <https://github.com/ImagingLab/Colorizing-with-GANs>
- Colorful Image Colorization reference: <https://github.com/richzhang/colorization?tab=readme-ov-file>
- IEEE final report templates: <https://www.ieee.org/conferences/publishing/templates>

## Checklist

- [ ] Agree on the minimal GAN pipeline and dataset subset.
- [ ] Get preprocessing working: `RGB -> Lab -> L, ab -> RGB`.
- [ ] Train a tiny end-to-end version: dataset, generator, discriminator, losses.
- [ ] Save result grids and basic metrics for at least one run.
- [ ] Compare what worked, what failed, and what to try next.
- [ ] Write the report around the actual implementation and results.
- [ ] Prepare presentation slides with problem, method, results, and limitations.
- [ ] Package the final code, report, slides, and submission material.
