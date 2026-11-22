"""
stub files for Cem, Arman, Fatma's modules.
These stubs let everyone work and test without the others' code being done yet.
Delete / replace each stub when the real file arrives.
"""

# ── constraints.py stub (Cem's) ──────────────────────────────────────────────
import numpy as np
import pandas as pd

def load_rules(path: str) -> dict:
    """Stub: returns empty rules dict."""
    return {}

def mutable_mask(rules: dict, meta: dict) -> np.ndarray:
    """Stub: all features mutable."""
    return np.ones(len(meta["feature_names"]), dtype=bool)

def check(x: np.ndarray, X_cf: np.ndarray, rules: dict) -> pd.DataFrame:
    """Stub: no violations."""
    k = len(X_cf)
    return pd.DataFrame({"cf_id": range(k), "rule": ["none"]*k, "violated": [False]*k})

def repair(x: np.ndarray, X_cf: np.ndarray, rules: dict) -> np.ndarray:
    """Stub: return CFs unchanged."""
    return X_cf

def action_cost(x: np.ndarray, X_cf: np.ndarray, rules: dict, meta: dict) -> np.ndarray:
    """Stub: return L1 distance as cost."""
    return np.linalg.norm(X_cf - x, ord=1, axis=1)
