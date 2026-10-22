<div align="center">

# 🖼️ CIFAR-10 Image Classification with CNNs

### From a simple baseline to transfer learning, error analysis and a parameter-budget challenge

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-blueviolet)
![Best accuracy](https://img.shields.io/badge/Best%20test%20accuracy-88.4%25-brightgreen)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)

</div>

---

## 📌 Overview

This project builds and compares image classifiers on **CIFAR-10**: 60,000 colour photos of size 32×32 in 10 classes (✈️ airplane, 🚗 automobile, 🐦 bird, 🐱 cat, 🦌 deer, 🐶 dog, 🐸 frog, 🐴 horse, 🚢 ship, 🚚 truck).

Instead of just training one model, the project follows the workflow of a real machine-learning practitioner: **predict before running, change one idea at a time, justify every decision with evidence, and analyse why the model fails.**

## 🎯 Highlights

- 🏆 **88.4 % test accuracy** with a custom CNN of only **245,578 parameters** (budget: 250,000), trained from scratch in about 10 minutes.
- 🚀 **85.4 % test accuracy** with partially fine-tuned **MobileNetV2** (transfer learning).
- 🔍 **Error analysis:** confusion matrix, failure gallery, calibration check and **Grad-CAM implemented from scratch**.
- 📉 **Data-efficiency study:** transfer learning gives +56 points over training from scratch when only 1,000 images are available.
- ✅ Clean experimental discipline: stratified validation split, all decisions made on validation data, test set used only for reporting.

## 📊 Results

| # | Model | Params | Best val acc | Test acc |
|---|-------|-------:|-------------:|---------:|
| 1 | Baseline CNN | 356,810 | 0.7222 | 0.7029 |
| 2 | + GlobalAveragePooling | 111,050 | 0.7420 | 0.7274 |
| 3 | + BatchNormalization | 111,946 | 0.6582 | 0.5028 |
| 4 | Deep blocks + Dropout | 306,602 | 0.8132 | 0.7816 |
| 5 | + Data augmentation | 306,602 | 0.8138 | 0.8036 |
| 6 | MobileNetV2, frozen base | 2,270,794 | 0.8170 | 0.8085 |
| 7 | MobileNetV2, **partial fine-tuning** | 2,270,794 | **0.8660** | **0.8543** |
| 8 | MobileNetV2, full fine-tuning | 2,270,794 | 0.8610 | 0.8425 |
| 🏅 | **Bonus: custom model (≤ 250k params)** | **245,578** | 0.8932 | **0.8844** |

> 🥈 The bonus model reaches the **Silver tier** (≥ 86 %) of the parameter-budget challenge. The test set was evaluated exactly once.

## 🧪 What was done

### 🔹 Problem 0 · Data understanding
Class-balance check (5,000 images per class), stratified 45,000 / 5,000 train/validation split, `uint8` images with a `Rescaling` layer inside the model to save memory (131.8 MB vs 527.3 MB as `float32`).

### 🔹 Problem 1 · Baseline CNN
Built with the Keras Functional API, parameters and output shapes calculated **by hand**. The model starts to overfit around epoch 5 (validation loss rises while training accuracy reaches ~95 %).

### 🔹 Problem 2 · Architecture experiments
One change at a time, with 🔮 predictions of the parameter counts made before running:
- **GlobalAveragePooling:** 3.2× fewer parameters and a smaller train/validation gap.
- **BatchNorm:** faster learning at the start, but worse final result in this setup.
- **Deeper blocks + Dropout:** the largest gain (+9.1 points of validation accuracy over the baseline).

### 🔹 Problem 3 · Augmentation and training control
Label-preserving augmentations (horizontal flip, small rotation, translation, zoom, contrast), `EarlyStopping`, `ReduceLROnPlateau`, a learning-rate plot and a small **ablation study** of the augmentations.

### 🔹 Problem 4 · Transfer learning
MobileNetV2 with correct preprocessing (`[-1, 1]` range) and 96×96 upscaling. Compared **frozen**, **partial** and **full** fine-tuning, and explained why the head is trained first, why re-compiling is needed and why BatchNorm layers stay in inference mode.

### 🔹 Problem 5 · Error analysis and interpretability
- Confusion matrix and per-class precision/recall: **cat** is the hardest class (recall 65.7 %).
- Most confused pairs: cat → dog (119), cat → frog (110), dog → cat (90), deer → frog (77), truck → automobile (74).
- **Calibration:** average confidence 0.90 vs accuracy 0.85, so the model is slightly over-confident.
- **Grad-CAM from scratch:** the model mostly looks at the object, but sometimes uses background (grass, water) as a cue.

### 🔹 Problem 6 · Data-efficiency study

| Training images | Scratch (test acc) | Transfer (test acc) |
|----------------:|-------------------:|--------------------:|
| 1,000 | 0.1454 | 0.7053 |
| 5,000 | 0.6017 | 0.7790 |
| 45,000 | 0.8036 | 0.8085 |

Transfer learning matters most when labelled data is scarce, and the gap closes as data grows.

### 🔹 Problem 7 · Bonus: parameter-budget challenge
3 blocks of 2 × (Conv → BatchNorm → ReLU) with 32/64/112 filters, no conv bias, GlobalAveragePooling directly into `Dense(10)`, light augmentation, label smoothing 0.1, AdamW with weight decay and a cosine learning-rate schedule.

## 🛠️ Tech stack

`Python` · `TensorFlow / Keras` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `scikit-learn`

## 📁 Repository structure

```
.
├── CNN_Project_CIFAR10.ipynb    # full notebook with code, outputs and written analysis
├── CNN_Project_CIFAR10.pdf      # PDF export of the notebook

```

## ▶️ How to run

1. Open the notebook in **Google Colab** or **Kaggle** and enable a **GPU** runtime (T4 recommended).
2. Run **Runtime → Run all**. CIFAR-10 and the MobileNetV2 weights are downloaded automatically by Keras.
3. Training everything takes roughly 1 to 1.5 hours on a T4 GPU.

> 💡 Results can vary slightly between runs and hardware. Single-run differences of about 1 point are within random noise.

## 🧭 Key takeaways

1. 📐 **More parameters ≠ better:** the 111k-parameter GAP model beat the 357k baseline.
2. 🔧 **Regularisation matters:** Dropout and augmentation closed the train/validation gap.
3. 🧠 **Reuse knowledge:** pre-trained features are worth the most when data is limited.
4. 🔬 **Look at the errors:** most mistakes involve similar animals or misleading backgrounds, and the model can be very confident when it is wrong.

## 🔮 Future work

- Combine the small custom design with transfer learning.
- Repeat experiments with several seeds to measure run-to-run variation.
- Add calibration (temperature scaling) and collect more cat/dog images.

## 👩‍💻 Author

**Iram Shahzadi**

⭐ If you found this project useful, feel free to star the repository!
