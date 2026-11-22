"""
generators.py  –  Candan Tuna Mesut
Counterfactual generation via DiCE-ml.

INTERFACE (frozen):
    dice_generate(model, x, k, method, meta, mutable_mask) -> np.ndarray  shape (k, d)

IMPORTANT: This is the ONLY module that uses DiCE DataFrames internally.
All outputs are numpy arrays so other modules stay DiCE-agnostic.

Runtime estimate:
    method="random"   : ~0.5–2 s per query  (model-agnostic, our default for POC)
    method="gradient" : ~5–15 s per query    (LR only, more precise)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _build_dice_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta: dict,
) -> dice_ml.Data:
    """
    Build a dice_ml.Data object from raw numpy arrays.

    Key gotcha: Folktables categoricals are stored as integers in our arrays,
    but DiCE must know which columns are categorical so it treats them correctly
    (indicator distance, no interpolation).  We pass them as strings to DiCE
    so it recognises them as nominal.
    """
    feature_names = meta["feature_names"]
    types         = meta["types"]

    # Reconstruct DataFrame with correct dtypes for DiCE
    df_train = pd.DataFrame(X_train, columns=feature_names)

    # DiCE expects categorical features as strings (or objects), not ints
    cat_cols = [c for c in feature_names if types[c] == "cat"]
    for col in cat_cols:
        df_train[col] = df_train[col].astype(str)

    df_train["target"] = y_train.astype(int)

    # Build continuous feature ranges
    continuous_features = [c for c in feature_names if types[c] in ("cont", "ord")]

    dice_data = dice_ml.Data(
        dataframe=df_train,
        continuous_features=continuous_features,
        outcome_name="target",
    )
    return dice_data


def dice_generate(
    model,                      # fitted sklearn Pipeline (from models.py)
    x: np.ndarray,              # 1-D array, shape (d,)  – the query instance
    k: int,
    meta: dict,
    X_train: np.ndarray,        # needed to initialise DiCE Data object
    y_train: np.ndarray,
    method: str = "random",     # "random" (model-agnostic) or "gradient" (LR only)
    mutable_mask: np.ndarray | None = None,  # bool array, True = feature may change
    desired_class: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate k counterfactuals for a single query x using DiCE.

    Parameters
    ----------
    model         : fitted sklearn Pipeline
    x             : query point, shape (d,)
    k             : number of counterfactuals to generate
    meta          : metadata dict from data.py
    X_train       : training features (needed to build DiCE Data object)
    y_train       : training labels
    method        : "random" or "gradient"
    mutable_mask  : bool array length d; False = feature must not change (immutable)
                    If None, all features are mutable.
    desired_class : target class for the CF (1 = income > 50k)
    seed          : random seed for reproducibility

    Returns
    -------
    X_cf : np.ndarray, shape (k, d)
           Each row is one counterfactual, columns match meta["feature_names"].
           Rows are padded by repeating last CF if DiCE returns fewer than k.
    """
    feature_names = meta["feature_names"]
    types         = meta["types"]
    cat_cols      = [c for c in feature_names if types[c] == "cat"]
    d             = len(feature_names)

    if mutable_mask is None:
        mutable_mask = np.ones(d, dtype=bool)

    # Features that must not change
    immutable_features = [feature_names[i] for i in range(d) if not mutable_mask[i]]

    # ------------------------------------------------------------------
    # Build DiCE objects
    # ------------------------------------------------------------------
    dice_data  = _build_dice_data(X_train, y_train, meta)
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    exp        = Dice(dice_data, dice_model, method=method)

    # ------------------------------------------------------------------
    # Prepare query as DataFrame (DiCE expects DataFrame input)
    # ------------------------------------------------------------------
    query_df = pd.DataFrame([x], columns=feature_names)
    for col in cat_cols:
        query_df[col] = query_df[col].astype(str)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    try:
        cf_result = exp.generate_counterfactuals(
            query_df,
            total_CFs=k,
            desired_class=desired_class,
            features_to_vary=[feature_names[i] for i in range(d) if mutable_mask[i]],
            random_seed=seed,
            verbose=False,
        )
        cf_df = cf_result.cf_examples_list[0].final_cfs_df

        if cf_df is None or len(cf_df) == 0:
            # DiCE found nothing – return query repeated k times as fallback
            print(f"[generators] WARNING: DiCE returned no CFs. Returning query copies.")
            return np.tile(x, (k, 1))

        # Drop the target column DiCE appends
        cf_df = cf_df[feature_names].copy()

        # Decode categorical columns back to float codes to match our array format
        for col in cat_cols:
            cf_df[col] = pd.to_numeric(cf_df[col], errors="coerce").fillna(0)

        X_cf = cf_df.values.astype(float)

    except Exception as e:
        print(f"[generators] WARNING: DiCE failed ({e}). Returning query copies.")
        return np.tile(x, (k, 1))

    # ------------------------------------------------------------------
    # Pad to exactly k rows if DiCE returned fewer
    # ------------------------------------------------------------------
    if len(X_cf) < k:
        pad = np.tile(X_cf[-1], (k - len(X_cf), 1))
        X_cf = np.vstack([X_cf, pad])

    return X_cf[:k]


def build_queries_csv(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    n_queries: int = 50,
    seed: int = 42,
    out_path: str = "experiments/queries.csv",
) -> np.ndarray:
    """
    Select n_queries instances that the model predicts as negative (income <= 50k).
    These are the people who *need* a counterfactual.

    Saves indices + seed to out_path and returns the query array.

    Returns
    -------
    queries : np.ndarray, shape (n_queries, d)
    """
    preds  = model.predict(X_test)
    neg_idx = np.where(preds == 0)[0]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(neg_idx, size=min(n_queries, len(neg_idx)), replace=False)
    chosen.sort()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df_out = pd.DataFrame({"index": chosen, "seed": seed})
    df_out.to_csv(out_path, index=False)
    print(f"[generators] queries.csv saved → {out_path}  ({len(chosen)} queries)")

    return X_test[chosen]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data   import load_acsincome
    from models import train_lr

    print("=== Generator smoke test (small subsample) ===")
    X_tr, X_te, y_tr, y_te, meta = load_acsincome(n_max=3_000)
    model = train_lr(X_tr, y_tr)

    # Pick one negative instance
    preds  = model.predict(X_te)
    neg    = X_te[preds == 0]
    if len(neg) == 0:
        print("No negatives found – try larger dataset")
        exit()

    x_query = neg[0]
    print(f"Query: {x_query}")

    X_cf = dice_generate(
        model=model, x=x_query, k=3, meta=meta,
        X_train=X_tr, y_train=y_tr, method="random", seed=42,
    )
    print(f"Generated CFs shape: {X_cf.shape}")
    for i, cf in enumerate(X_cf):
        print(f"  CF {i}: {cf}  → pred={model.predict([cf])[0]}")

    # Save queries.csv
    queries = build_queries_csv(X_te, y_te, model, n_queries=50, seed=42,
                                out_path="../experiments/queries.csv")
    print(f"Queries shape: {queries.shape}")
