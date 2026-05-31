"""
src/models/classical.py
=======================
Defines, trains, and tunes all seven classical machine learning models.

Each model is trained twice:
  - Base version   : sensible default hyperparameters
  - Tuned version  : best hyperparameters found by GridSearchCV

The seven models cover the main families of supervised classification:
  1. Logistic Regression   - linear, interpretable baseline
  2. Decision Tree         - non-linear, fully readable rule structure
  3. Random Forest         - bagged ensemble, best overall discriminator
  4. SVM (RBF kernel)      - margin-based, non-linear boundaries
  5. KNN                   - distance-based, instance-level reasoning
  6. Naive Bayes           - probabilistic, fast baseline
  7. Gradient Boosting     - sequential ensemble, best precision

The Random Forest model is returned separately because its feature
importance scores are needed for Figure 10 in the report.
"""

import time
import numpy as np

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier)
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import accuracy_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GRID_PARAMS, CV_FOLDS, RANDOM_STATE


def get_base_models() -> dict:
    """
    Return a dictionary of model name to unfitted estimator.

    The parameters chosen here are reasonable starting points based
    on the published literature for this dataset size and type.

    Logistic Regression  : max_iter=1000 because the default of 100
                           often fails to converge after MinMax scaling.
    Decision Tree        : max_depth=5 caps the tree to prevent overfitting
                           on 614 training records.
    Random Forest        : class_weight='balanced' compensates for the
                           65/35 class imbalance without oversampling.
    SVM                  : probability=True is needed to compute AUC scores.
    KNN                  : n_neighbors=7 is a common starting point for
                           datasets of this size.
    Naive Bayes          : var_smoothing=1e-9 is the scikit-learn default.
    Gradient Boosting    : learning_rate=0.05 with 200 trees is a slow
                           learner configuration that generalises well.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, random_state=RANDOM_STATE),

        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),

        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE),

        "KNN": KNeighborsClassifier(n_neighbors=7),

        "Naive Bayes": GaussianNB(var_smoothing=1e-9),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=RANDOM_STATE),
    }


def train_all(X_train: np.ndarray,
              y_train: np.ndarray,
              X_test:  np.ndarray,
              y_test:  np.ndarray) -> tuple:
    """
    Train every classical model at base configuration and then tune
    each one with GridSearchCV to find the best hyperparameters.

    Parameters
    ----------
    X_train, y_train  : scaled training data
    X_test,  y_test   : held-out test data (used only for accuracy reporting)

    Returns
    -------
    base_models   : dict of name to fitted base estimator
    tuned_models  : dict of name to fitted tuned estimator
    base_acc      : dict of name to test accuracy (base)
    tuned_acc     : dict of name to test accuracy (tuned)
    best_params   : dict of name to best hyperparameter dict
    rf_tuned      : the tuned Random Forest estimator (for Figure 10)
    """
    base_models  = {}
    tuned_models = {}
    base_acc     = {}
    tuned_acc    = {}
    best_params  = {}
    rf_tuned     = None

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    for name, estimator in get_base_models().items():
        t0 = time.time()

        # Train the base model
        estimator.fit(X_train, y_train)
        base_models[name] = estimator
        base_acc[name]    = round(
            accuracy_score(y_test, estimator.predict(X_test)), 4)

        # Run GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(
            estimator.__class__(**estimator.get_params()),
            GRID_PARAMS[name],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)

        tuned_models[name] = gs.best_estimator_
        tuned_acc[name]    = round(
            accuracy_score(y_test, gs.best_estimator_.predict(X_test)), 4)
        best_params[name]  = gs.best_params_

        if name == "Random Forest":
            rf_tuned = gs.best_estimator_

        elapsed = round(time.time() - t0, 1)
        change  = tuned_acc[name] - base_acc[name]
        sign    = "+" if change >= 0 else ""
        print(f"  {name:<22}  "
              f"base={base_acc[name]:.4f}  "
              f"tuned={tuned_acc[name]:.4f}  "
              f"change={sign}{change:.4f}  "
              f"[{elapsed}s]")
        print(f"    Best params: {best_params[name]}")

    return (base_models, tuned_models,
            base_acc, tuned_acc, best_params, rf_tuned)
