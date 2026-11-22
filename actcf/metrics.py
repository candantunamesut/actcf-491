"""
metrics.py stub  –  Fatma Ilgın Puhur (stub by Candan for integration)
Replace this file with Fatma's real implementation.
"""
import numpy as np

def validity(model, X_cf, desired_class=1):
    preds = model.predict(X_cf)
    return float(np.mean(preds == desired_class))

def proximity(x, X_cf, meta):
    MAD = np.array([meta["MAD"][c] for c in meta["feature_names"]])
    return float(np.mean(np.sum(np.abs(X_cf - x) / np.maximum(MAD, 1e-6), axis=1)))

def sparsity(x, X_cf):
    d = X_cf.shape[1]
    return float(np.mean(np.sum(X_cf != x, axis=1) / d))
