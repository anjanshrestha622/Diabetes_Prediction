"""
src/models/deep_learning.py
===========================
Defines and trains three deep learning architectures:
  ANN  - 3-layer feedforward network (primary DL baseline)
  DNN  - 4-layer network with BatchNorm (depth test)
  LSTM - recurrent network (empirical stress-test on static data)

The LSTM is included deliberately to test a claim made in several
published papers. LSTM was designed for sequential data where the
order of observations matters. The PIMA dataset contains eight static
clinical measurements from a single patient visit - there is no time
sequence in this data. Applying LSTM here allows the thesis to provide
direct empirical evidence that recurrent architectures do not add value
when the data has no temporal structure.

KerasWrapper
------------
A thin class that wraps any Keras model and gives it the same
predict(), predict_proba(), and score() interface as a scikit-learn
estimator. This means the evaluation code can handle all ten models
(seven classical, three deep learning) with exactly the same logic.

Training protocol (same for all three models)
----------------------------------------------
  Optimiser         : Adam
  Loss function     : binary cross-entropy
  Batch size        : 32
  Max epochs        : 120
  EarlyStopping     : patience 15, restores best weights
  ReduceLROnPlateau : halves learning rate if val_loss stalls for 8 epochs
  Validation split  : 15 percent of training data
"""

import time
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras.models     import Sequential
from tf_keras.layers     import (Dense, Dropout, BatchNormalization,
                                  Reshape, LSTM as KerasLSTM)
from tf_keras.optimizers import Adam
from tf_keras.callbacks  import EarlyStopping, ReduceLROnPlateau

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (RANDOM_STATE, DL_EPOCHS, DL_BATCH_SIZE, DL_VAL_SPLIT,
                    ES_PATIENCE, LR_PATIENCE, LR_FACTOR, LR_MIN,
                    ANN_LR, DNN_LR, LSTM_LR)

# Fix seeds so results are reproducible
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def _make_callbacks() -> list:
    """
    Return a fresh list of training callbacks.

    EarlyStopping stops training when validation loss has not improved
    for 15 consecutive epochs and then restores the weights from the
    best epoch rather than the final one.

    ReduceLROnPlateau halves the learning rate when validation loss
    has not improved for 8 consecutive epochs, which helps the
    optimiser escape plateaus without manual tuning.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LR_MIN,
            verbose=0,
        ),
    ]


def build_ann(input_dim: int) -> Sequential:
    """
    ANN - Artificial Neural Network with three hidden layers.

    Architecture: Input(8) -> Dense(64) -> Dense(32) -> Dense(16) -> Output(1)

    This is the primary deep learning baseline. Feedforward networks
    make no assumptions about temporal order or spatial structure,
    which makes them the natural choice for tabular clinical data.
    Dropout rates decrease toward the output (0.30, 0.20, 0.10) to
    regularise early layers more heavily.
    """
    model = Sequential(name="ANN")
    model.add(Dense(64, activation="relu", input_dim=input_dim,
                    kernel_initializer="he_uniform"))
    model.add(Dropout(0.30))
    model.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.20))
    model.add(Dense(16, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.10))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(Adam(ANN_LR), "binary_crossentropy", ["accuracy"])
    return model


def build_dnn(input_dim: int) -> Sequential:
    """
    DNN - Deeper network with four hidden layers and BatchNormalization.

    Architecture: Input(8) -> 128 -> 64 -> 32 -> 16 -> Output(1)
    BatchNorm is applied after each hidden layer.

    This model tests whether added depth and batch normalisation
    improve performance on this small dataset. The expectation from
    the literature is that they will not, because the PIMA dataset
    is too small and has too few features to benefit from hierarchical
    feature extraction.
    """
    model = Sequential(name="DNN")
    model.add(Dense(128, activation="relu", input_dim=input_dim,
                    kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Dropout(0.40))
    model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))
    model.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(Dense(16, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.10))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(Adam(DNN_LR), "binary_crossentropy", ["accuracy"])
    return model


def build_lstm(input_dim: int) -> Sequential:
    """
    LSTM - Long Short-Term Memory network applied as a stress-test.

    Architecture: Input(8) -> Reshape(8,1) -> LSTM(64) -> LSTM(32)
                           -> Dense(16) -> Dense(8) -> Output(1)

    To use a recurrent network on tabular data, the eight features
    are reshaped into a pseudo-sequence of 8 time steps with 1 value
    each. There is no real temporal order in these measurements, so
    the LSTM gating mechanism has no sequential signal to learn from.
    Including this model provides direct empirical evidence of the
    architectural mismatch that several published papers ignore.
    """
    model = Sequential(name="LSTM")
    model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
    model.add(KerasLSTM(64, return_sequences=True,
                        kernel_initializer="glorot_uniform"))
    model.add(Dropout(0.30))
    model.add(KerasLSTM(32, return_sequences=False,
                        kernel_initializer="glorot_uniform"))
    model.add(Dropout(0.20))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.10))
    model.add(Dense(8,  activation="relu"))
    model.add(Dense(1,  activation="sigmoid"))
    model.compile(Adam(LSTM_LR), "binary_crossentropy", ["accuracy"])
    return model


class KerasWrapper:
    """
    Wraps a Keras Sequential model with a scikit-learn compatible interface.

    After calling .fit(X, y), the wrapper provides:
      .predict(X)          - binary predictions (0 or 1)
      .predict_proba(X)    - probability array of shape (n, 2)
      .score(X, y)         - accuracy as a float

    This allows the evaluation and visualisation code to loop over
    all ten models with identical logic regardless of whether each
    model is a scikit-learn estimator or a Keras network.

    Attributes
    ----------
    name    : display name used in all figures and print statements
    history : the Keras History object, populated after .fit() is called
    """

    def __init__(self, model: Sequential, name: str):
        self.model   = model
        self.name    = name
        self.history = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KerasWrapper":
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        self.history = self.model.fit(
            X, y,
            epochs           = DL_EPOCHS,
            batch_size       = DL_BATCH_SIZE,
            validation_split = DL_VAL_SPLIT,
            callbacks        = _make_callbacks(),
            verbose          = 0,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float32)
        p = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())


def train_all(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Build and train ANN, DNN, and LSTM on the training data.

    Returns a dictionary of name to fitted KerasWrapper.
    """
    dim      = X_train.shape[1]
    builders = {"ANN": build_ann, "DNN": build_dnn, "LSTM": build_lstm}
    trained  = {}

    for name, build_fn in builders.items():
        print(f"\n  Training {name} ...", end="", flush=True)
        t0      = time.time()
        wrapper = KerasWrapper(build_fn(dim), name)
        wrapper.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 1)
        ep      = len(wrapper.history.history["loss"])
        val_acc = wrapper.history.history["val_accuracy"][-1]
        print(f"  done  "
              f"[{ep} epochs | val_acc={val_acc:.4f} | {elapsed}s]")
        trained[name] = wrapper

    return trained
