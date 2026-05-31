"""
run.py
======
The single entry point for the complete diabetes prediction pipeline.

Run this file to execute the full analysis from start to finish:

    python run.py

What this script does, in order:
  Phase 1  Load and preprocess the PIMA dataset
  Phase 2  Train 7 classical ML models (base + GridSearchCV tuned)
  Phase 3  Train 3 deep learning models (ANN, DNN, LSTM)
  Phase 4  Evaluate all 10 models on the held-out test set
  Phase 5  Generate all 19 code-produced figures to outputs/figures/
  Phase 6  Print a final summary of the key results

The full run takes around 60-90 seconds on a standard Windows laptop.
Because the random seed is fixed at 42 throughout, the results will
be identical every time the script is run.

Note on figures
---------------
Figures 1, 2, and 3 (flowcharts and the approach diagram) were created
manually using a diagramming tool and are not produced by this script.
This script generates Figures 4 through 22.

Author      : Anjan Shrestha
Degree      : Master of Information Technology (Software Engineering)
University  : Charles Darwin University
Supervisor  : Dr. Asif Karim
"""

import os
import sys
import time
import warnings

# Suppress TensorFlow startup noise and library warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Make sure the project root is on the Python path so all imports work
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import banner, make_output_dirs

from src.preprocessing import run_pipeline

from src.models.classical    import train_all as train_classical
from src.models.deep_learning import train_all as train_deep

from src.evaluation import evaluate_all, save_results

from src.visualization.eda_figures import generate_all as gen_eda
from src.visualization.ml_figures  import generate_all as gen_ml
from src.visualization.dl_figures  import generate_all as gen_dl


def main():

    total_start = time.time()

    # Print the project header
    print()
    print("=" * 66)
    print("  PIMA DIABETES PREDICTION  -  COMPLETE ANALYSIS PIPELINE")
    print()
    print("  Author     : Anjan Shrestha")
    print("  Degree     : MIT (Software Engineering)")
    print("  University : Charles Darwin University")
    print("  Dataset    : PIMA Indians Diabetes (768 records, 8 features)")
    print("  Models     : 7 Classical ML  +  3 Deep Learning")
    print("=" * 66)

    # Create output directories if they do not exist
    make_output_dirs()


    # ── Phase 1: Data loading and preprocessing ───────────────────────────────
    banner("PHASE 1  -  DATA LOADING AND PREPROCESSING")

    (X_train, X_test, y_train, y_test,
     scaler, medians, df_raw, df_clean) = run_pipeline()

    print("\n  Preprocessing complete.")


    # ── Phase 2: Classical ML models ──────────────────────────────────────────
    banner("PHASE 2  -  CLASSICAL MACHINE LEARNING MODELS")
    print("  Training 7 models (base + GridSearchCV tuned each)...\n")

    t0 = time.time()
    (base_models, tuned_models,
     base_acc, tuned_acc,
     best_params, rf_tuned) = train_classical(X_train, y_train,
                                               X_test,  y_test)

    print(f"\n  Classical models complete  [{round(time.time()-t0, 1)}s]")


    # ── Phase 3: Deep learning models ─────────────────────────────────────────
    banner("PHASE 3  -  DEEP LEARNING MODELS  (ANN  |  DNN  |  LSTM)")

    t0 = time.time()
    dl_models = train_deep(X_train, y_train)

    print(f"\n  Deep learning models complete  [{round(time.time()-t0, 1)}s]")


    # ── Phase 4: Evaluation ───────────────────────────────────────────────────
    banner("PHASE 4  -  EVALUATING ALL 10 MODELS")

    # Merge all models in a consistent display order:
    # 7 tuned classical models first, then ANN, DNN, LSTM
    all_models = {**tuned_models, **dl_models}

    results_df = evaluate_all(all_models, X_test, y_test)

    print("\n  Final results table (sorted by ROC-AUC):\n")
    print(results_df[["Model", "Accuracy", "Precision",
                       "Recall", "F1", "ROC_AUC", "FN_Rate"]].to_string(
        index=False))

    save_results(results_df)
    print("\n  Evaluation complete.")


    # ── Phase 5: Figure generation ────────────────────────────────────────────
    banner("PHASE 5  -  GENERATING ALL FIGURES  (Figures 4 through 22)")

    t0 = time.time()

    print("\n  EDA figures (4, 5, 6):")
    gen_eda(df_raw, df_clean, medians)

    print("\n  Classical ML figures (7-16):")
    gen_ml(base_acc, tuned_acc, tuned_models, rf_tuned, X_test, y_test)

    print("\n  Deep learning and clinical figures (17-22):")
    gen_dl(dl_models, results_df)

    print(f"\n  All figures saved  [{round(time.time()-t0, 1)}s]")


    # ── Phase 6: Final summary ────────────────────────────────────────────────
    banner("PHASE 6  -  FINAL SUMMARY")

    total_time = round(time.time() - total_start, 1)
    best_auc   = results_df.iloc[0]
    best_fn    = results_df.sort_values("FN_Rate").iloc[0]

    print(f"\n  Total runtime  : {total_time} seconds\n")

    print("  Best model by ROC-AUC (discrimination):")
    print(f"    Model    : {best_auc['Model']}")
    print(f"    ROC-AUC  : {best_auc['ROC_AUC']}")
    print(f"    Accuracy : {best_auc['Accuracy']}")
    print(f"    Recall   : {best_auc['Recall']}")

    print("\n  Best model for clinical screening (lowest FN-Rate):")
    print(f"    Model    : {best_fn['Model']}")
    print(f"    FN-Rate  : {best_fn['FN_Rate']}")
    print(f"    Recall   : {best_fn['Recall']}")
    print(f"    ROC-AUC  : {best_fn['ROC_AUC']}")

    from config import FIGURES_DIR, RESULTS_DIR
    fig_count = len([f for f in os.listdir(FIGURES_DIR)
                     if f.endswith(".png")])
    print(f"\n  Output locations:")
    print(f"    Figures  ({fig_count} PNG files) : {FIGURES_DIR}")
    print(f"    Results  (CSV + TXT)         : {RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
