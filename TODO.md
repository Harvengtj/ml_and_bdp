# Project TODO List

Based on the architectural and structural audits of the codebase (`ml_and_bdp/src/main.py` and `main.ipynb`) compared to the reference paper (1803.05400v5.pdf), the following issues need to be addressed. 

## 🔴 MAJOR (Breaks intended pipeline or results)

- [ ] **Fix Input Resolution Mismatch:** The code actively resizes CIFAR-10 images to `64x64` despite documentation and network design expecting `32x32`. This fundamentally alters the bottleneck size (to `4x4` instead of `2x2`) and downstream tensor math. Ensure `image_size` is set to `32` or the network architecture is properly scaled for `64x64`.
- [ ] **Fix Discriminator Output Shape (PatchGAN Bug):** Because of the 64x64 input and mismatched padding, the discriminator outputs a `[B, 1, 5, 5]` (in `main.py`) or `[B, 1, 6, 6]` (in `main.ipynb`) grid. Re-evaluate the strides and padding so it outputs the intended spatial grid size for the chosen input resolution.
- [ ] **Resolve Codebase Inconsistencies:** `main.py` and `main.ipynb` have conflicting architectures (e.g., 5 encoder layers with stride 1 start in `main.py` vs. 4 encoder layers with stride 2 start in `main.ipynb`; differing paddings resulting in different output grids). Unify the implementations.

## 🟠 MODERATE (Deviations from paper altering data flow)

- [ ] **Correct Generator Output Channels:** The paper specifies the generator should output a full 3-channel image (L*a*b*). The current code outputs `output_channels=2` (only `a` and `b`). Decide whether to strictly follow the paper or intentionally keep this optimized approach (and document the deviation).
- [ ] **Correct Discriminator Input Channels:** As a consequence of the generator output, the discriminator is currently receiving 3 channels (L + ab). The paper specifies a 4-channel input (grayscale condition + full colored image). 
- [ ] **Fix Generator First Layer Stride:** The paper specifies stride 2 for all contracting layers. The current implementation uses stride 1 for the first layer (`conv0`) to preserve resolution early on. 

## 🟡 MINOR (Small discrepancies or practical adaptations)

- [ ] **Remove First Layer Batch Normalization:** The paper explicitly notes that Batch-Norm should *not* be applied to the first layer of the generator. The code currently applies `bn0` after `conv0`.
- [ ] **Verify Discriminator Strides:** The paper states a series of stride 2 convolutions for the discriminator. The code changes to stride 1 towards the end. Verify if this was an intentional adaptation to preserve the PatchGAN spatial grid on small inputs.
- [ ] **Document Sigmoid / BCEWithLogitsLoss Choice:** The paper specifies a final `sigmoid` layer for the discriminator. The code omits this layer structurally and uses `nn.BCEWithLogitsLoss()`. While this is a PyTorch best practice for numerical stability, it should be documented as an intentional structural deviation.