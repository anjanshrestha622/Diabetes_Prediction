"""
src/evaluation.py
=================
Evaluates every trained model on the held-out test set and saves results.

Six metrics are computed per model:
  Accuracy    - fraction of correct predictions overall
  Precision   - fraction of diabetic predictions that are actually diabetic
  Recall      - fraction of actual diabetics that the model identified
  F1-Score    - harmonic mean of precision and recall
  ROC-AUC     - discrimination across all decision thresholds
  FN-Rate     - false negatives divided by total actual positives

Why FN-Rate is the main clinical metric
----------------------------------------
In a diabetes screening context the worst type of error is a false
negative: a patient who actually has diabetes is told they are healthy
and sent home without any treatment or follow-up. Standard metrics
like accuracy and AUC do not capture this directly. FN-Rate measures
it explicitly and is used as the primary clinical ranking criterion
in Figure 21 and the conclusions of the thesis.
"""

import os
import numpy  as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


def compute_metrics(name: str, model, X_test: np.ndarray,
                    y_test: np.ndarray) -> dict:
    """
    Compute all six metrics for one fitted model on the test set.

    Works with both scikit-learn estimators and KerasWrapper objects
    because both provide a predict() and predict_proba() interface.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    fn        = int(((y_pred == 0) & (y_test == 1)).sum())
    total_pos = int((y_test == 1).sum())

    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, y_prob), 4),
        "FN":        fn,
        "FN_Rate":   round(fn / total_pos, 3),
    }


def evaluate_all(all_models: dict, X_test: np.ndarray,
                 y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate every model and compile results into a DataFrame.

    Also prints a detailed classification report for each model
    so the per-class precision and recall can be reviewed in the
    console output.

    Returns a DataFrame sorted by ROC_AUC descending.
    """
    rows = []
    for name, model in all_models.items():
        row = compute_metrics(name, model, X_test, y_test)
        rows.append(row)
        print(f"  {name:<22}  "
              f"Acc={row['Accuracy']:.4f}  "
              f"Prec={row['Precision']:.4f}  "
              f"Rec={row['Recall']:.4f}  "
              f"F1={row['F1']:.4f}  "
              f"AUC={row['ROC_AUC']:.4f}  "
              f"FN_Rate={row['FN_Rate']:.3f}")

    df = (pd.DataFrame(rows)
            .sort_values("ROC_AUC", ascending=False)
            .reset_index(drop=True))

    print("\n" + "-" * 66)
    print("  Detailed classification reports:")
    print("-" * 66)
    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        print(f"\n  --- {name} ---")
        print(classification_report(
            y_test, y_pred,
            target_names=["Non-Diabetic", "Diabetic"],
            zero_division=0,
        ))

    return df


def save_results(df: pd.DataFrame) -> None:
    """
    Save the results table to CSV and to a plain text file.

    Two files are written to the results output folder:
      metrics_results.csv   - machine-readable table for import into Excel
      metrics_summary.txt   - human-readable summary with best model callout
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "metrics_results.csv")
    txt_path = os.path.join(RESULTS_DIR, "metrics_summary.txt")

    df.to_csv(csv_path, index_label="Rank")

    best_auc = df.iloc[0]
    best_fn  = df.sort_values("FN_Rate").iloc[0]

    with open(txt_path, "w") as f:
        f.write("PIMA DIABETES PREDICTION - MODEL COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(df.to_string())
        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"\nBest by ROC-AUC          : {best_auc['Model']}"
                f"  (AUC = {best_auc['ROC_AUC']})\n")
        f.write(f"Best for clinical screen : {best_fn['Model']}"
                f"  (FN-Rate = {best_fn['FN_Rate']})\n")

    print(f"\n  Saved  ->  {csv_path}")
    print(f"  Saved  ->  {txt_path}")
