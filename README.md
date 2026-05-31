# PIMA Diabetes Prediction - Complete Analysis Pipeline

**Author:** Anjan Shrestha  
**Degree:** Master of Information Technology (Software Engineering)  
**University:** Charles Darwin University  
**Supervisor:** Dr. Asif Karim  

---

## What this project does

This project trains and compares ten artificial intelligence models for predicting
Type 2 Diabetes Mellitus using the PIMA Indian Diabetes dataset. Seven classical
machine learning algorithms and three deep learning architectures are trained under
identical conditions and evaluated using six performance metrics. The false-negative
rate is used as the primary clinical criterion because in diabetes screening, missing
a diabetic patient is the most harmful type of error.

All results tables, classification reports, and 19 publication-quality figures are
generated automatically by running a single Python script.

---

## Folder structure

```
diabetes_prediction/
│
├── README.md                        this file
├── requirements.txt                 all Python dependencies with versions
├── config.py                        all paths, constants, and settings
├── run.py                           single entry point - run this file
│
├── data/
│   └── diabetes.csv                 PIMA Indian Diabetes dataset (768 records)
│
├── src/
│   ├── preprocessing.py             load, impute, split, scale
│   ├── evaluation.py                compute all 6 metrics, save results
│   ├── utils.py                     shared helpers used across all modules
│   │
│   ├── models/
│   │   ├── classical.py             7 classical ML models + GridSearchCV tuning
│   │   └── deep_learning.py         ANN, DNN, LSTM + KerasWrapper
│   │
│   └── visualization/
│       ├── eda_figures.py           Figures 4, 5, 6
│       ├── ml_figures.py            Figures 7-16
│       └── dl_figures.py            Figures 17-22
│
└── outputs/
    ├── figures/                     all generated PNG figures saved here
    └── results/                     metrics CSV and summary text file
```

---

## Quick start

**Step 1** - Open a terminal in the project folder (VS Code integrated terminal)

**Step 2** - Install dependencies:
```
pip install -r requirements.txt
```

**Step 3** - Run the full pipeline:
```
python run.py
```

---

## What gets generated

**19 figures** saved to `outputs/figures/`:

| Figure | Description |
|--------|-------------|
| Figure 4 | Dataset structure and class balance panel |
| Figure 5 | Correlation heatmap |
| Figure 6 | Feature distributions before and after preprocessing |
| Figure 7 | Logistic Regression - base vs tuned accuracy |
| Figure 8 | Decision Tree - base vs tuned accuracy |
| Figure 9 | Random Forest - base vs tuned accuracy |
| Figure 10 | Random Forest feature importance ranking |
| Figure 11 | SVM - base vs tuned accuracy |
| Figure 12 | KNN - base vs tuned accuracy |
| Figure 13 | Naive Bayes - base vs tuned accuracy |
| Figure 14 | Gradient Boosting - base vs tuned accuracy |
| Figure 15 | All 7 classical models compared |
| Figure 16 | ROC curves for all 7 classical models |
| Figure 17 | ANN training and validation curves |
| Figure 18 | DNN training and validation curves |
| Figure 19 | LSTM training and validation curves |
| Figure 20 | All 3 DL models compared across epochs |
| Figure 21 | Clinical priority ranking by false-negative rate |
| Figure 22 | Three-metric comprehensive comparison (all 10 models) |

Figures 1, 2, and 3 (ML flowchart, DL flowchart, and approach diagram) were
created manually using a diagramming tool and are not produced by this code.

**2 result files** saved to `outputs/results/`:
- `metrics_results.csv` - full results table with all 6 metrics
- `metrics_summary.txt` - human-readable summary with best model callout

---

## Models

| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | Classical ML |
| 2 | Decision Tree | Classical ML |
| 3 | Random Forest | Classical ML |
| 4 | SVM (RBF kernel) | Classical ML |
| 5 | KNN (k=7 base) | Classical ML |
| 6 | Naive Bayes | Classical ML |
| 7 | Gradient Boosting | Classical ML |
| 8 | ANN (64-32-16) | Deep Learning |
| 9 | DNN (128-64-32-16 + BatchNorm) | Deep Learning |
| 10 | LSTM (64+32 recurrent) | Deep Learning |

---

## Dataset

PIMA Indian Diabetes Dataset  
Source: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/34/diabetes  
768 records, 8 clinical features, binary outcome (diabetic/non-diabetic)
