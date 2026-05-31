"""
src/preprocessing.py
====================
Loads the PIMA dataset, handles missing values, and prepares
the training and test sets ready for model training.

The three steps are:
  1. Load    - read the CSV and print a summary of what is in it
  2. Impute  - replace zero-coded missing values with training-fold medians
  3. Split   - stratified 80/20 split then MinMax scaling

The key design choice throughout is that nothing from the test set
is allowed to influence the training process. The imputation medians
are calculated only from the training rows, and the scaler is fitted
only on the training rows. This prevents data leakage, which is a
common source of inflated performance figures in published studies
on this dataset.
"""

import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, FEATURES, TARGET, ZERO_COLS, TEST_SIZE, RANDOM_STATE


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Read the CSV file and print a short summary to the console.

    Returns the raw DataFrame with no modifications applied.
    """
    df = pd.read_csv(path)

    print(f"\n  File   : {path}")
    print(f"  Rows   : {len(df)}")
    print(f"  Cols   : {df.columns.tolist()}")
    print(f"\n  Class distribution:")
    for val in [0, 1]:
        label = "Diabetic" if val == 1 else "Non-Diabetic"
        count = (df[TARGET] == val).sum()
        pct   = count / len(df) * 100
        print(f"    {val}  ({label:<12}) : {count}  ({pct:.1f}%)")

    return df


def impute_missing(df: pd.DataFrame,
                   train_idx: pd.Index) -> tuple:
    """
    Replace physiologically impossible zeros with the training-fold median.

    The median is calculated from training rows only. This is important
    because computing the median on the full dataset before splitting
    would let test-set information influence the training process,
    which is a form of data leakage.

    Returns the cleaned DataFrame and a dict of the medians used.
    """
    df_clean      = df.copy()
    medians_used  = {}

    print("\n  Missing value imputation:")
    for col in ZERO_COLS:
        zero_count = int((df_clean[col] == 0).sum())

        # Only use non-zero values from training rows to calculate the median
        train_vals = df_clean.loc[train_idx, col]
        median_val = float(train_vals[train_vals != 0].median())
        medians_used[col] = round(median_val, 2)

        # Replace zeros with the training median across the whole dataset
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(median_val)

        print(f"    {col:<28} {zero_count:>3} zeros  ->  median {median_val:.2f}")

    print(f"\n  Remaining NaN values: {df_clean.isnull().sum().sum()}")
    return df_clean, medians_used


def split_and_scale(df_clean: pd.DataFrame):
    """
    Split into 80 percent training and 20 percent test, then scale.

    Stratified splitting keeps the same 65/35 class ratio in both folds.
    The MinMax scaler is fitted only on the training fold, then used to
    transform both folds. The test fold never touches the scaler fitting.

    Returns X_train, X_test, y_train, y_test, and the fitted scaler.
    """
    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    print(f"\n  Train  : {X_train.shape}  "
          f"(class 0: {(y_train==0).sum()}  class 1: {(y_train==1).sum()})")
    print(f"  Test   : {X_test.shape}   "
          f"(class 0: {(y_test==0).sum()}   class 1: {(y_test==1).sum()})")

    return X_train, X_test, y_train, y_test, scaler


def run_pipeline(path: str = DATA_PATH):
    """
    Run all three preprocessing steps in order and return everything
    needed for model training and visualisation.

    Returns:
        X_train, X_test, y_train, y_test,
        scaler, medians_used, df_raw, df_clean
    """
    df_raw = load_data(path)

    # Get the training row indices before imputation to avoid leakage
    _, __, y = df_raw[FEATURES].values, None, df_raw[TARGET].values
    train_idx_arr, _ = train_test_split(
        df_raw.index,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    train_idx = pd.Index(train_idx_arr)

    df_clean, medians_used = impute_missing(df_raw, train_idx)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_clean)

    return (X_train, X_test, y_train, y_test,
            scaler, medians_used, df_raw, df_clean)
