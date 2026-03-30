# Credit Card Fraud Detection

## Overview

This project presents an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The analysis uses the ULB credit card fraud detection dataset, comprising 284,807 transactions and 492 fraud cases, to directly address the challenge of the domain by tackling the class imbalance at a ratio of approximatively 575/1. The project is structured as a single, self-contained Jupyter notebook designed to be read as a technical report where each modeling decision is accompanied by a theoretical justification.

---

## Methodology

The pipeline proceeds through five structured phases:

### 1. Exploratory data analysis
Visual and statistical characterisation of the dataset, including class distribution, transaction amount and temporal patterns by class, PCA component distributions, and the correlation structure of features.

### 2. Pre-processing
`RobustScaler` applied to `Amount` and `Time` (resistant to the heavy-tailed distribution of transaction amounts, unlike `StandardScaler`)
Stratified 80/20 train/test split to preserve the original class ratio in both partitions

### 3. SMOTE
Synthetic Minority Over-sampling Technique (Chawla et al., 2002) is applied exclusively to the training set. SMOTE interpolates between existing minority-class instances in feature space, generating synthetic fraud samples that expand the decision boundary rather than simply duplicating points.


### 4. Model training
Three classifiers of increasing complexity are trained and compared:

| Model | Rationale |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Random Forest (200 trees) | Non-linear ensemble; robust via bootstrap aggregation |
| XGBoost (300 rounds) | State-of-the-art gradient boosting; `scale_pos_weight` tuned for imbalance |

### 5. Evaluation
Models are evaluated on the held-out test set using metrics appropriate for imbalanced classification.

---

## Why accuracy is the wrong metric

With only ~0.17% fraudulent transactions, a classifier that labels every transaction as legitimate achieves 99.83% accuracy while catching zero fraud. 
The metrics used in this analysis are:

| Metric | Formula | Relevance |
|---|---|---|
| **Precision** | TP / (TP + FP) | Cost of false alarms (customer friction) |
| **Recall** | TP / (TP + FN) | Cost of missed fraud (direct financial loss) |
| **F1-Score** | Harmonic mean of P & R | Balanced single-number summary |
| **AUPRC** | Area under Precision-Recall curve | Preferred over ROC-AUC in imbalanced settings; not inflated by true negatives |

---

## Results

| Model | Precision | Recall | F1-Score | AUPRC |
|---|---|---|---|---|
| Logistic Regression | ~0.87 | ~0.62 | ~0.73 | ~0.72 |
| Random Forest | ~0.95 | ~0.82 | ~0.88 | ~0.87 |
| **XGBoost** | **~0.93** | **~0.85** | **~0.89** | **~0.89** |

> *Note: Exact figures will vary slightly depending on your environment. Run the notebook to reproduce.*

XGBoost achieves the strongest overall performance, with Random Forest competitive on Precision. For a deployment context where minimising false negatives (missed fraud) is the primary objective, Recall should be weighted more heavily in the final threshold selection.

---

## Project Structure

```
fraud-detection/
├── fraud_analysis.ipynb   
├── requirements.txt       
├── README.md              
└── creditcard.csv         ← Download from Kaggle (not included in repo)
```

---

## Setup & Running

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the dataset
1. Go to [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the root of this repository (same folder as the notebook)

### 4. Launch the notebook
```bash
jupyter notebook fraud_analysis.ipynb
# or, for JupyterLab:
jupyter lab fraud_analysis.ipynb
```

Run cells sequentially (`Shift+Enter`). Total runtime is approximately 3/6 min
---



--
