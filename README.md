# Machine Learning and Big Data Processing
Repository for the project 'Grayscale Image Colorization'.

# Deep Learning Project Summary: Grayscale Image Colorization

## 1. Current Progress & Group Decisions
* **Selected Topic:** Grayscale Image Colorization exploring Generative Adversarial Networks (GANs) and class rebalancing.
* **Framework:** PyTorch (adapting from existing TensorFlow open-source code).
* **Code Status:** Preliminary draft of the Generator and Discriminator classes initialized, structured similarly to practical sessions.
* **Dataset:** ImageNet.
* **Report Status:** Overleaf document created; initial algorithm description drafted.
* **Timeline:** Active collaboration from all members to begin next week due to ongoing thesis workloads.

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