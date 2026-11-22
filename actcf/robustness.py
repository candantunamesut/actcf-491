"""
robustness.py stub  –  Arman İbrişim (stub by Candan for integration)
Replace this file with Arman's real implementation.
"""
import numpy as np

def ir_monte_carlo(model, X_cf, mask, scale=0.05, S=100, seed=42) -> np.ndarray:
    """Stub: return zeros (no perturbation test yet)."""
    return np.zeros(len(X_cf))

def ir_closed_form_lr(model, X_cf, mask, scale=0.05) -> np.ndarray:
    """Stub: return zeros."""
    return np.zeros(len(X_cf))

def var_score(X_cf, models) -> np.ndarray:
    """Stub: return ones (all CFs valid across all models)."""
    return np.ones(len(X_cf))

def rank(x, X_cf, cost, ir, agreement, alpha=0.2, q=0.5, k=3) -> np.ndarray:
    """Stub: return first k CFs."""
    return X_cf[:k]
