# 🐝 **Pollen Image Classification using Bag of Visual Words (BoVW) + SIFT**

This repository contains the complete source code, configuration files, preprocessing scripts, and preliminary results of a classical computer vision pipeline for pollen grain image classification.  
The project was developed as the **final assignment of the Computer Vision course ([PPGMCS](https://ppgmcs.com.br/))**.

The approach uses the classical **SIFT → Bag of Visual Words → Linear SVM (LinearSVC)** pipeline to classify pollen from three plant species associated with honey:

- **avocado (abacate)**  
- **aroeira**  
- **eucalyptus**

---

## 📌 Project Objectives

- Extract local features (SIFT) from microscopy pollen images  
- Build a visual vocabulary using MiniBatchKMeans  
- Represent images as normalized BoVW histograms  
- Train and evaluate a Linear SVM classifier  
- Provide a reproducible and modular experimental pipeline  

---

## 🧪 Preliminary Results (Baseline K=100)

With the **baseline configuration (K=100)**, the model achieved:

- **Accuracy:** 96.67%  
- **58 correct predictions out of 60 test images**  
- Only **2 misclassified samples**  
- Per-class Precision/Recall/F1 above **0.95**

All generated outputs (reports, confusion matrix, misclassified images, etc.) are saved under:

```
results/baseline_k100/
```

---

## 🧹 Data & Preprocessing

The dataset contains ~100 images per class. The following steps were applied:

- Image resizing  
- Organized folder structure  
- 80/20 train-test split per class  
- Descriptor count estimation using `scripts/count_descriptors.py`  
- Use of resized dataset for BoVW/SIFT

Folder organization:

```
data/raw/
data/resized/
data/split_80_20/
```

---

## 🏗️ Methodology

### **1. SIFT Feature Extraction**
- Keypoint detection + 128-dimensional descriptors

### **2. Visual Vocabulary Construction**
- MiniBatchKMeans  
- K = 100 (baseline)  
- Descriptor subsampling (≈50,000 descriptors)  

### **3. BoVW Representation**
- Histogram over visual words  
- L2 normalization  
- Optional: Dense SIFT in future experiments  

### **4. Linear SVM Classification**
- Implemented with `sklearn.svm.LinearSVC`  
- `C = 1.0`, `dual=False`, `max_iter=20000`

### **5. Evaluation**
- Accuracy  
- Confusion matrix  
- Per-class Precision/Recall/F1  
- Export of misclassified images  

---

## 🧬 Project Structure

```

pollen-image-classifier/
├── configs/                 # YAML experiment configurations
│   └── baseline_k100.yaml
├── data/                    # dataset (raw, resized, splits)
├── scripts/                 # preprocessing and utility scripts
├── src/                     # main source code
│   ├── cli.py               # command-line interface
│   ├── run_bovw.py          # pipeline orchestration
│   ├── features/            # SIFT / dense SIFT extraction
│   ├── clustering/          # KMeans vocabulary construction
│   ├── models/              # training and evaluation modules
│   └── utils/               # IO and helper functions
└── results/                 # experimental results (ignored via .gitignore)

```

---

## ▶️ Reproducing Experiments

### **1. Create environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **2. Build the visual vocabulary (KMeans)**

```bash
python -m src.cli --config configs/baseline_k100.yaml --mode build_vocab
```

### **3. Train the classifier**

```bash
python -m src.cli --config configs/baseline_k100.yaml --mode train
```

### **4. Evaluate the model**

```bash
python -m src.cli --config configs/baseline_k100.yaml --mode evaluate
```

### **5. Run the entire pipeline**

```bash
python -m src.cli --config configs/baseline_k100.yaml --mode run-all
```

---

## 🧩 Configuration (YAML)

Example extract from `configs/baseline_k100.yaml`:

```yaml
descriptor:
  type: "sift"
  dense: false
kmeans:
  K: 100
  sample_descriptors: 50000
classifier:
  type: "linear_svc"
  C: 1.0
out_dir: "results/baseline_k100"
```

---

## 📈 Future Work

* Enable Dense SIFT for richer texture representation
* Apply Hellinger normalization (`sqrt(histogram)`)
* Perform full grid search for K and SVM hyperparameters
* Compare with deep feature extraction (e.g., ResNet, EfficientNet)
* Test VLAD and Fisher Vector encodings
* Explore Spatial Pyramid Matching (SPM)
* Expand dataset for robustness analysis

---

## 📜 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for more details.

---

## 👩 Author

**Angelina de Meiras-Ottoni**   
Data Science Researcher — LICA/PPGMCS/UNIMONTES
