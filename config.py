"""
config.py
=========
Every path, constant, colour, and hyperparameter used across
this project is defined here in one place.

If you need to change a setting — a file path, a model parameter,
a colour — you only need to edit this file. Nothing else changes.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────
# The project root is the folder that contains this file.
# All other paths are built relative to it so the project
# works regardless of where it is saved on your machine.

ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(ROOT, "data", "diabetes.csv")
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")

# ── Dataset ───────────────────────────────────────────────────────
TARGET   = "Outcome"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# These five columns contain zeros that are biologically impossible.
# A blood glucose of zero or a BMI of zero cannot occur in a living
# patient. These zeros mean the measurement was not recorded, so
# they are treated as missing values and replaced during preprocessing.
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Experiment settings
TEST_SIZE    = 0.20   # 20 percent held out for testing
RANDOM_STATE = 42     # fixed seed for full reproducibility

# ── Classical model hyperparameter grids ─────────────────────────
# Each grid is passed to GridSearchCV which tries every combination
# using 5-fold stratified cross-validation.

GRID_PARAMS = {
    "Logistic Regression": {
        "C":      [0.01, 0.1, 1.0, 10],
        "solver": ["lbfgs", "newton-cg"],
    },
    "Decision Tree": {
        "criterion":        ["gini", "entropy"],
        "max_depth":        [4, 6, 8, 10],
        "min_samples_leaf": [2, 4, 6],
    },
    "Random Forest": {
        "n_estimators":      [100, 200],
        "max_depth":         [8, 10, 12],
        "min_samples_split": [2, 5],
    },
    "SVM": {
        "C":     [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    },
    "KNN": {
        "n_neighbors": [5, 7, 9, 11],
        "metric":      ["euclidean", "manhattan"],
        "weights":     ["uniform", "distance"],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
    },
    "Gradient Boosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 4, 5],
    },
}

CV_FOLDS = 5

# ── Deep learning settings ────────────────────────────────────────
DL_EPOCHS      = 120
DL_BATCH_SIZE  = 32
DL_VAL_SPLIT   = 0.15   # 15 percent of training data used for validation
ES_PATIENCE    = 15      # stop early if val_loss does not improve for 15 epochs
LR_PATIENCE    = 8       # halve the learning rate after 8 stagnant epochs
LR_FACTOR      = 0.5
LR_MIN         = 1e-6

ANN_LR  = 1e-3
DNN_LR  = 5e-4
LSTM_LR = 5e-4

# ── Colours ───────────────────────────────────────────────────────
# Dark background theme used consistently across all figures.
DARK_BG = "#0F1117"
NAVY    = "#1a1a2e"
BLUE    = "#2E74B5"
TEAL    = "#1abc9c"
RED     = "#e74c3c"
AMBER   = "#e67e22"
GREEN   = "#27ae60"
PURPLE  = "#8e44ad"
WHITE   = "#FFFFFF"
GREY    = "#AAAAAA"

# One colour per model, always in the same order:
# 7 classical models first, then ANN, DNN, LSTM.
MODEL_COLORS = [
    "#4C72B0",  # Logistic Regression
    "#DD8452",  # Decision Tree
    "#55A868",  # Random Forest
    "#C44E52",  # SVM
    "#8172B3",  # KNN
    "#937860",  # Naive Bayes
    "#DA8BC3",  # Gradient Boosting
    "#64B5CD",  # ANN
    "#E377C2",  # DNN
    "#7F7F7F",  # LSTM
]

# Each classical model gets a pair: (base bar colour, tuned bar colour)
BAR_COLORS = {
    "Logistic Regression": ("#4C72B0", "#7AA0D8"),
    "Decision Tree":       ("#DD8452", "#F0AD78"),
    "Random Forest":       ("#55A868", "#88CC95"),
    "SVM":                 ("#C44E52", "#E08080"),
    "KNN":                 ("#8172B3", "#B0A0E0"),
    "Naive Bayes":         ("#937860", "#BFA585"),
    "Gradient Boosting":   ("#DA8BC3", "#EDBBDD"),
}

# Which figure number each model comparison bar chart gets.
MODEL_FIG_NUM = {
    "Logistic Regression": 7,
    "Decision Tree":       8,
    "Random Forest":       9,
    "SVM":                 11,
    "KNN":                 12,
    "Naive Bayes":         13,
    "Gradient Boosting":   14,
}

# Figure output resolution in dots per inch.
DPI  = 200
FONT = "DejaVu Sans"
