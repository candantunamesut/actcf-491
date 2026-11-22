"""
test_hand_calc.py  –  Candan Tuna Mesut
Verification §5.2: hand-calculated LR CF vs DiCE output.

For a Logistic Regression with StandardScaler, the decision boundary is a hyperplane.
The ℓ₁-nearest valid counterfactual (ignoring actionability) has an analytical form:
    - Find the feature j* with the largest |w̃_j| / MAD_j  (most "efficient" to change)
    - Shift x along that axis until f(x') = τ

This gives us an exact reference to compare against DiCE's random search output.

Pass criteria:
    1. The hand-calc CF has f(cf_hand) >= 0.5  (valid)
    2. DiCE's best CF has f(cf_dice) >= 0.5    (valid)
    3. Proximity of hand-calc CF ≤ proximity of DiCE CF * 1.5
       (DiCE is approximate so we allow 50% slack)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "actcf"))

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from data       import load_acsincome
from models     import train_lr
from generators import dice_generate


# ---------------------------------------------------------------------------
# Shared fixtures (loaded once for the whole test session)
# ---------------------------------------------------------------------------
_cache: dict = {}

def _get_data_and_model():
    if "model" not in _cache:
        X_tr, X_te, y_tr, y_te, meta = load_acsincome(n_max=3_000, seed=42)
        model = train_lr(X_tr, y_tr, seed=42)
        # Pick a query: LR-rejected from test set
        preds = model.predict(X_te)
        neg_idx = np.where(preds == 0)[0]
        assert len(neg_idx) > 0, "No negative predictions – increase n_max"
        x_query = X_te[neg_idx[0]]
        _cache.update(dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr,
                           meta=meta, model=model, x_query=x_query))
    return _cache


def _lr_closed_form_cf(model, x: np.ndarray, meta: dict, tau: float = 0.5) -> np.ndarray:
    """
    Closed-form ℓ₁-nearest CF for a linear classifier + StandardScaler.

    Strategy:
        1. Extract w, b from the fitted LR (in scaled space).
        2. Compute scaled version of x.
        3. Find the feature j* = argmax |w̃_j| / MAD_j where w̃ is in original space.
        4. Shift x[j*] until logit = log(tau/(1-tau)) = 0 (for tau=0.5).
        5. Return the modified point in original space.

    This is a *single-feature perturbation* CF – the simplest provably valid CF.
    Real min-ℓ₁ CF over all features requires LP; this is a constructive lower bound.
    """
    lr_step  = model.named_steps["lr"]
    sc_step  = model.named_steps["scaler"]

    w = lr_step.coef_[0]          # shape (d,) in scaled space
    b = lr_step.intercept_[0]     # scalar

    mu    = sc_step.mean_
    sigma = sc_step.scale_

    # Transform x to scaled space
    x_scaled = (x - mu) / sigma
    logit_x  = np.dot(w, x_scaled) + b   # current logit

    if logit_x >= 0:
        # Already positive class – nothing to do
        return x.copy()

    # We need logit to reach 0 (decision boundary, tau=0.5)
    # Shift feature j by delta_j (in original space):
    #   logit(x + delta_j * e_j) = logit_x + w_j * delta_j / sigma_j = 0
    #   => delta_j = -logit_x * sigma_j / w_j
    # Cost (ℓ₁ in MAD-scaled space) = |delta_j| / MAD_j

    MAD  = np.array([meta["MAD"][c] for c in meta["feature_names"]])
    types = meta["types"]
    feat  = meta["feature_names"]

    best_j     = None
    best_cost  = np.inf
    best_delta = 0.0

    for j in range(len(w)):
        if abs(w[j]) < 1e-10:
            continue
        if types[feat[j]] != "cont":   # only perturb continuous features
            continue

        delta_j = -logit_x * sigma[j] / w[j]
        cost_j  = abs(delta_j) / max(MAD[j], 1e-6)

        # Range check
        lo, hi = meta["ranges"][feat[j]]
        new_val = x[j] + delta_j
        if not (lo <= new_val <= hi):
            continue

        if cost_j < best_cost:
            best_cost  = cost_j
            best_j     = j
            best_delta = delta_j

    cf = x.copy()
    if best_j is not None:
        cf[best_j] += best_delta

    return cf


def _mad_proximity(x: np.ndarray, cf: np.ndarray, meta: dict) -> float:
    """MAD-scaled ℓ₁ proximity (lower = better)."""
    MAD   = np.array([meta["MAD"][c] for c in meta["feature_names"]])
    types = meta["types"]
    feat  = meta["feature_names"]
    total = 0.0
    for j in range(len(feat)):
        if types[feat[j]] in ("cont", "ord"):
            total += abs(x[j] - cf[j]) / max(MAD[j], 1e-6)
        else:
            total += float(x[j] != cf[j])
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hand_calc_cf_is_valid():
    """Hand-calculated CF must cross the decision boundary."""
    d = _get_data_and_model()
    cf = _lr_closed_form_cf(d["model"], d["x_query"], d["meta"])
    prob = d["model"].predict_proba([cf])[0][1]
    assert prob >= 0.5, (
        f"Hand-calc CF has P(y=1) = {prob:.4f}, expected >= 0.5.\n"
        f"This means the closed-form formula has a bug OR no continuous "
        f"feature can reach the boundary within its valid range."
    )


def test_dice_cf_is_valid():
    """At least one DiCE CF must cross the decision boundary."""
    d = _get_data_and_model()
    X_cf = dice_generate(
        model=d["model"], x=d["x_query"], k=5, meta=d["meta"],
        X_train=d["X_tr"], y_train=d["y_tr"], method="random", seed=42,
    )
    probs = d["model"].predict_proba(X_cf)[:, 1]
    assert any(probs >= 0.5), (
        f"No DiCE CF is valid. Probs: {probs}.\n"
        f"Likely cause: too few iterations or mutable features too constrained."
    )


def test_hand_calc_vs_dice_proximity():
    """
    The hand-calc (single-feature) CF's proximity should be ≤ DiCE's best CF * 1.5.

    We allow 50% slack because:
    - Hand-calc perturbs only 1 feature (minimum possible intervention)
    - DiCE is a heuristic over all features; it may spread changes across features
    - The important thing is that both are in a comparable ballpark
    """
    d = _get_data_and_model()
    cf_hand  = _lr_closed_form_cf(d["model"], d["x_query"], d["meta"])
    X_cf_dice = dice_generate(
        model=d["model"], x=d["x_query"], k=5, meta=d["meta"],
        X_train=d["X_tr"], y_train=d["y_tr"], method="random", seed=42,
    )

    prox_hand = _mad_proximity(d["x_query"], cf_hand, d["meta"])
    # Best (smallest) proximity among valid DiCE CFs
    prox_dice_list = [
        _mad_proximity(d["x_query"], X_cf_dice[i], d["meta"])
        for i in range(len(X_cf_dice))
        if d["model"].predict([X_cf_dice[i]])[0] == 1
    ]

    if not prox_dice_list:
        pytest.skip("No valid DiCE CF found – skip proximity comparison.")

    prox_dice_best = min(prox_dice_list)
    print(f"\n  Hand-calc proximity : {prox_hand:.4f}")
    print(f"  DiCE best proximity : {prox_dice_best:.4f}")

    # Hand-calc should be tighter or within 50% of DiCE
    assert prox_hand <= prox_dice_best * 1.5, (
        f"Hand-calc CF ({prox_hand:.4f}) is much farther than DiCE's best "
        f"({prox_dice_best:.4f}). Check the closed-form formula."
    )


def test_immutable_features_unchanged():
    """
    If we mark AGEP (age) and RAC1P (race) as immutable,
    DiCE must not change them (mutable_mask enforces this).
    """
    d = _get_data_and_model()
    feat = d["meta"]["feature_names"]
    mask = np.ones(len(feat), dtype=bool)
    for col in ["AGEP", "RAC1P", "SEX"]:
        mask[feat.index(col)] = False

    X_cf = dice_generate(
        model=d["model"], x=d["x_query"], k=3, meta=d["meta"],
        X_train=d["X_tr"], y_train=d["y_tr"], method="random",
        mutable_mask=mask, seed=42,
    )

    for col in ["AGEP", "RAC1P", "SEX"]:
        j = feat.index(col)
        for i in range(len(X_cf)):
            assert X_cf[i, j] == d["x_query"][j], (
                f"Immutable feature {col} changed in CF {i}: "
                f"{d['x_query'][j]} → {X_cf[i, j]}"
            )


if __name__ == "__main__":
    # Run as script for quick manual check
    print("=== Hand-calc test (manual run) ===")
    d = _get_data_and_model()
    cf = _lr_closed_form_cf(d["model"], d["x_query"], d["meta"])
    prob = d["model"].predict_proba([cf])[0][1]
    prox = _mad_proximity(d["x_query"], cf, d["meta"])
    print(f"Query prediction:   {d['model'].predict_proba([d['x_query']])[0]}")
    print(f"Hand-calc CF prob:  {prob:.4f}")
    print(f"Hand-calc proximity: {prox:.4f}")

    print("\nRunning test_hand_calc_cf_is_valid()...")
    test_hand_calc_cf_is_valid()
    print("✓ PASSED")

    print("\nRunning test_dice_cf_is_valid()...")
    test_dice_cf_is_valid()
    print("✓ PASSED")

    print("\nRunning test_hand_calc_vs_dice_proximity()...")
    test_hand_calc_vs_dice_proximity()
    print("✓ PASSED")

    print("\nRunning test_immutable_features_unchanged()...")
    test_immutable_features_unchanged()
    print("✓ PASSED")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! (Section 5.2 verified)")
