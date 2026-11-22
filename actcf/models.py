"""
models.py  –  Candan Tuna Mesut
Classifier training functions.

INTERFACE (frozen):
    train_lr(X, y, seed) -> sklearn LogisticRegression
    train_bootstrap_models(X, y, n, seed0) -> list[LogisticRegression]

Notes:
    - train_lr uses fixed seed for reproducibility.
    - train_bootstrap_models is used by Arman's robustness.py
      to simulate model retraining (bootstrap subsamples, same architecture).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

DEFAULT_SEED = 42


def train_lr(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> Pipeline:
    """
    Train a Logistic Regression classifier (with StandardScaler).

    LR is chosen as the POC model because:
    - Decision boundary is linear → CF problem becomes LP/MILP → exactly solvable
    - Closed-form LR CF can serve as a verification oracle for DiCE output
    - Convex loss guarantees unique global optimum

    Returns a sklearn Pipeline so that prediction is always on raw (unscaled) input,
    consistent with what DiCE and all other modules receive.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
            C=1.0,
        )),
    ])
    model.fit(X, y)

    train_acc = model.score(X, y)
    print(f"[models] LR trained — train acc: {train_acc:.4f}")
    return model


def train_bootstrap_models(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 10,
    seed0: int = DEFAULT_SEED,
) -> list[Pipeline]:
    """
    Train n LR models on bootstrap subsamples of (X, y).

    Used by Arman's robustness.py to estimate VaR (Value-at-Risk) score:
    how often does a CF remain valid across different model instances
    trained on subsets of the same data?

    Parameters
    ----------
    X, y   : training data (full training set)
    n      : number of bootstrap models
    seed0  : base seed; model i uses seed seed0 + i

    Returns
    -------
    list of n fitted Pipeline objects (same architecture as train_lr)
    """
    models = []
    for i in range(n):
        seed_i = seed0 + i
        X_boot, y_boot = resample(X, y, random_state=seed_i, replace=True)
        m = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=seed_i,
                C=1.0,
            )),
        ])
        m.fit(X_boot, y_boot)
        models.append(m)

    print(f"[models] {n} bootstrap LR models trained (seeds {seed0}–{seed0+n-1})")
    return models


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data import load_acsincome
    X_tr, X_te, y_tr, y_te, meta = load_acsincome(n_max=5_000)
    model = train_lr(X_tr, y_tr)
    test_acc = model.score(X_te, y_te)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Predict proba shape: {model.predict_proba(X_te[:3]).shape}")

    boot_models = train_bootstrap_models(X_tr, y_tr, n=3)
    for i, m in enumerate(boot_models):
        print(f"  Bootstrap model {i}: test acc = {m.score(X_te, y_te):.4f}")
