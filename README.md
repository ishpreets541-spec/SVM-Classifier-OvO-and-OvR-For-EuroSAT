# GNR602: EuroSAT SVM Land Cover Classifier

A Tkinter-based desktop application for classifying EuroSAT satellite imagery
using Support Vector Machines (SVM). Compares **One-vs-One (OvO)** and
**One-vs-Rest (OvR)** multi-class strategies across four kernel types with
real-time parameter tuning.

---

## Features

- **4 Kernel Types:** Linear, Polynomial, RBF, Sigmoid
- **Dual Strategy Comparison:** OvO vs OvR accuracy and execution time side-by-side
- **Rich Feature Extraction:** Color statistics, HSV histograms, Sobel edge magnitude
- **Spatial Classification Maps:** Patch-level classification overlaid on test images
- **Confusion Matrix Visualization** for OvO predictions
- **Interactive GUI** — no command-line required

---

## Requirements

```
Python >= 3.8
opencv-python
numpy
scikit-learn
matplotlib
libsvm
tkinter (standard library)
```

Install dependencies:

```bash
pip install opencv-python numpy scikit-learn matplotlib libsvm-official
```

---

## Dataset

Download the [EuroSAT RGB dataset](https://github.com/phelber/EuroSAT) or via kagglehub:

```python
import kagglehub
path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
print("Path to dataset files:", path)
```

The expected folder structure is:

```
EuroSAT/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

---

## Usage

```bash
python src/GNR602_SVM.py
```

1. Set your kernel, C, gamma, and images-per-class in the GUI
2. Click **"Select Folder & Start Classification"**
3. Navigate to your EuroSAT root folder
4. Two result windows open: one for metrics and one for spatial maps

---

## Parameter Guide

See [`docs/parameter_tuning_guide.md`](docs/parameter_tuning_guide.md) for a
full kernel-by-kernel breakdown of how C, gamma, and strategy (OvO vs OvR)
affect accuracy and execution time.

### Quick Reference

| Kernel     | Best C | Best g | OvO Acc | OvR Acc |
|------------|--------|--------|---------|---------|
| RBF        | 8000   | 0.05   | ~85%    | ~84%    |
| Linear     | 1      | —      | ~81%    | ~81%    |
| Polynomial | 10     | 0.05   | ~78%    | ~77%    |
| Sigmoid    | 100    | 0.005  | ~72%    | ~70%    |

---

## Feature Extraction

Each image is represented by a **71-dimensional feature vector**:

| Feature        | Dimensions | Description                              |
|----------------|-----------|------------------------------------------|
| Color Means    | 3         | BGR channel means                        |
| Color StdDevs  | 3         | BGR channel standard deviations          |
| HSV Histogram  | 64        | 4×4×4 bin histogram in HSV space         |
| Edge Magnitude | 1         | Mean Sobel gradient magnitude            |
| **Total**      | **71**    |                                          |

Features are min-max normalised to [0, 1] using training-set statistics
before both training and inference.

---

## OvO vs OvR: Key Differences

|                      | One-vs-One (OvO)                          | One-vs-Rest (OvR)                               |
|----------------------|-------------------------------------------|-------------------------------------------------|
| **Models trained**   | K(K−1)/2 = 45                             | K = 10                                          |
| **Inference**        | 1 predict call → majority vote            | 10 predict calls → argmax decision score        |
| **Typical accuracy** | Higher (or equal)                         | ~0–1% lower                                     |
| **Typical speed**    | **Faster overall** at these dataset sizes | Slower — 10 predict passes dominate total time  |
| **Best for**         | Accuracy and speed at 500 imgs/class      | Very large training sets                        |

> **Why OvR is slower in practice:**
> The OvO timer (in the code) covers 1 training call + 1 predict call on the full
> test set. The OvR timer covers 10 training calls + 10 predict calls (one per
> class, to collect decision values for argmax). At 500 images/class (~1 000 test
> samples), those 10 predict passes dominate and make OvR's total wall-clock time
> **3–5× longer** than OvO's, despite OvR training fewer models.
>
> Measured example — Linear kernel, C=1, 500 images/class:
> ```
> OvO Time: 0.6553s
> OvR Time: 2.8971s
> ```

---

## Project Structure

```
svm-eurosat/
├── src/
│   └── GNR602_SVM.py               # Main application
├── docs/
│   ├── parameter_tuning_guide.md   # Markdown guide (GitHub-renderable)
│   └── parameter_tuning_guide.docx # Formatted Word document
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Course

**GNR602** — Digital Image Processing
EuroSAT benchmark dataset: Helber et al. (2019)
