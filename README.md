# Custom CNN vs. VGG16 Transfer Learning for Pneumonia Detection in Chest X-Rays

A recall-oriented comparison of a convolutional neural network trained from scratch against a
fine-tuned **VGG16** transfer-learning model for **binary chest X-ray classification**
(`NORMAL` vs. `PNEUMONIA`).

**Author:** Ivan Logutov · M.Sc. Data Science, University of Europe for Applied Sciences
**Supervisor:** Raja Hashim Ali
**Course:** Machine Learning — Final Project (Phase 3: research-style report and presentation)

📄 **Final report (PDF):** [`Pneumonia_Detection_CNN_vs_VGG16_Final_Report.pdf`](Pneumonia_Detection_CNN_vs_VGG16_Final_Report.pdf)
📝 **LaTeX source (IEEE format):** [`overleaf-project/`](overleaf-project/)
💻 **Code (runnable notebook):** [`notebooks/pneumonia_detection.ipynb`](notebooks/pneumonia_detection.ipynb)

> **Clinical framing:** a missed pneumonia (False Negative) is more dangerous than a false
> alarm, so **Recall / Sensitivity** is the primary metric and **F1** is secondary.

---

## Abstract
Pneumonia is a leading cause of death from infection worldwide, and chest radiography is the
most common test used to confirm it. We compare two deep-learning strategies for detecting
pneumonia from chest X-rays under identical conditions: a lightweight CNN trained from scratch
and a fine-tuned VGG16 transfer-learning model. Both are trained on the Kermany benchmark with
a class-weighted loss to handle the strong class imbalance, and interpreted with Grad-CAM.
On the held-out test set, VGG16 reaches **0.896 accuracy / 0.922 F1 / 0.970 AUC**, while the
custom CNN reaches a lower **0.728 accuracy** but a higher **0.997 recall**. The two approaches
serve different clinical goals rather than one being strictly better: VGG16 for balanced,
confident decisions, and the high-recall custom CNN as an aggressive screening filter.

## Research questions
1. How effectively can a lightweight custom CNN detect pneumonia from chest X-rays when trained from scratch on a limited dataset?
2. Does VGG16 transfer learning outperform a custom CNN on the Kermany benchmark?
3. How does a class-weighted loss affect performance given the class imbalance?
4. Can Grad-CAM provide interpretable, clinically meaningful evidence of the decision regions?
5. What are the recall–accuracy trade-offs, and what do they imply for clinical screening?

## 1. Problem statement
Given a chest X-ray image, predict whether the patient has pneumonia (`PNEUMONIA = 1`) or not
(`NORMAL = 0`). This is a supervised, binary image-classification task addressed with
convolutional neural networks.

## 2. Dataset
- **Name:** Chest X-Ray Images (Pneumonia)
- **Source / link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Original paper:** Kermany et al. (2018), *Cell*, DOI: 10.1016/j.cell.2018.02.010
- **Type:** grayscale chest X-ray images (JPEG)
- **Classes (2):** `NORMAL`, `PNEUMONIA`
- **Size:** 5,856 images total

| Split | NORMAL | PNEUMONIA | Total |
|-------|-------:|----------:|------:|
| Train (original) | 1,341 | 3,875 | 5,216 |
| Test  | 234 | 390 | 624 |

**Notes handled in code:**
- The original `val/` split has only 16 images, so a **stratified 80/20 validation split is
  rebuilt from the training set** (read-only safe — no files are copied).
- The training set is imbalanced (~1:2.9), addressed with **class weights**.

## 3. Methodology
- **Preprocessing:** resize all images to **224×224**, replicate grayscale to 3 channels (RGB),
  normalize pixels to `[0, 1]` (`rescale=1/255`).
- **Augmentation (train only, conservative for medical imaging):** horizontal flip, ±10°
  rotation, small width/height shift and zoom. No vertical flip / shear / heavy jitter.
- **Custom CNN:** 4 blocks of `Conv2D → BatchNorm → ReLU → MaxPool` (32→64→128→256) →
  `GlobalAveragePooling → Dense(512) → Dropout(0.5) → Dense(1, sigmoid)`. ~522K parameters.
- **Transfer learning (VGG16):** ImageNet backbone + custom head
  (`GlobalAveragePooling → Dense(256) → Dropout(0.4) → Dense(1, sigmoid)`), trained in two
  phases — feature extraction (frozen base) then fine-tuning of the last 4 conv layers (block 5)
  at a lower learning rate.
- **Training:** Adam, binary cross-entropy, `class_weight`, callbacks `EarlyStopping` /
  `ReduceLROnPlateau` / `ModelCheckpoint` monitored on **`val_auc`** (a threshold-free metric
  that, unlike recall, cannot be gamed by a model that simply predicts PNEUMONIA for everything).
- **Evaluation:** best saved weights reloaded before testing on the untouched `test/` set;
  metrics are accuracy, precision, recall, F1, AUC-ROC, plus the confusion matrix and Grad-CAM.

## 4. Results (test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|:--------:|:---------:|:------:|:--:|:---:|
| Custom CNN | 0.728 | 0.697 | **0.997** | 0.821 | 0.930 |
| **VGG16 (transfer learning)** | **0.896** | **0.868** | 0.982 | **0.922** | **0.970** |

**Key findings**
- **VGG16 clearly wins** on every balanced metric (Accuracy, Precision, F1, AUC) — pretrained
  features generalize far better than a network trained from scratch on ~5k images.
- VGG16 misses only **7 / 390** pneumonia cases (high sensitivity) with acceptable specificity.
- The Custom CNN reaches near-perfect recall but at a fixed 0.5 threshold it **over-predicts
  PNEUMONIA** (low specificity); its AUC of 0.93 shows it still discriminates well — the issue
  is decision-threshold calibration, not lack of learning.

## 5. Figures
All figures are saved to `figures/`.

| | |
|---|---|
| Class distribution | ![](figures/class_distribution.png) |
| Sample images | ![](figures/sample_images.png) |
| Custom CNN — training curves | ![](figures/custom_cnn_training_curves.png) |
| VGG16 — training curves | ![](figures/vgg16_training_curves.png) |
| Custom CNN — confusion matrix | ![](figures/custom_cnn_confusion_matrix.png) |
| VGG16 — confusion matrix | ![](figures/vgg16_confusion_matrix.png) |
| Model comparison | ![](figures/model_comparison.png) |
| Grad-CAM | ![](figures/gradcam_visualization.png) |

## 6. Repository structure
```
.
├── notebooks/
│   └── pneumonia_detection.ipynb         # full, runnable pipeline
├── figures/                              # all generated figures (PNG)
├── overleaf-project/                     # LaTeX source of the final paper (IEEE format)
│   ├── Report.tex                        # main document
│   ├── bibliography.bib                  # 28 references
│   ├── IEEEtran.cls / IEEEtran.bst       # IEEE template files
│   ├── Figures/                          # figures used in the paper (PDF)
│   ├── Codes/  Dataset/                  # code + dataset link
│   ├── Figure and Table Source Folder/   # editable figure sources + raw-data Excel
│   ├── Literature Review/                # related-work material
│   └── Presentation/                     # presentation assets
├── Pneumonia_Detection_CNN_vs_VGG16_Final_Report.pdf   # compiled final report
├── Proposal_Pneumonia_Detection.pdf      # Phase 1/2 proposal
├── README.md
├── LICENSE
└── .gitignore                            # excludes the dataset and large model files
```
> The dataset (`chest_xray/`, ~2.4 GB) and trained models (`kaggle-output/*.keras`, incl. a
> 117 MB VGG16) are **intentionally not committed** — download the dataset from the link above;
> the models are regenerated by running the notebook.

## 7. How to run
**Kaggle (recommended):**
1. Create a new Kaggle Notebook and *Add Data* → search `chest-xray-pneumonia`
   (`paultimothymooney/chest-xray-pneumonia`).
2. Settings → **Accelerator → GPU**.
3. Open `notebooks/pneumonia_detection.ipynb` and **Run All**. Outputs (models + `figures/`)
   are written to `/kaggle/working/`. The dataset path is auto-detected.

**Local:** install `tensorflow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`,
`scikit-learn`; point the dataset path to your local `chest_xray/` (the notebook searches for
the folder containing `train/NORMAL` and `test/NORMAL`).

## 8. Reproducibility
All random operations are seeded (`SEED = 42`: Python / NumPy / TensorFlow).

## 9. Limitations & future work
Single-source dataset (possible scanner/text artifacts); validation drawn from the train
distribution; VGG16 uses a plain `1/255` rescale rather than its native `preprocess_input`;
a fixed 0.5 threshold. Future work: tune the decision threshold for a target sensitivity,
add ROC curves and a dedicated error analysis, try stronger backbones (EfficientNet/DenseNet),
focal loss, and k-fold cross-validation.

## 10. Key references
- Kermany et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* **Cell**, 172(5), 1122–1131.
- Simonyan & Zisserman (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* **ICLR**.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* **ICCV**.

The complete reference list (28 entries) is in [`overleaf-project/bibliography.bib`](overleaf-project/bibliography.bib).

## How to cite
```
Logutov, I. (2025). Custom CNN vs. VGG16 Transfer Learning for Pneumonia Detection in
Chest X-Rays: A Recall-Oriented Comparison. Machine Learning Final Project,
University of Europe for Applied Sciences.
```

## License
Released under the terms in [`LICENSE`](LICENSE).
