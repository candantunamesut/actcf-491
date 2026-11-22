"""
data.py  –  Candan Tuna Mesut
ACSIncome dataset loading, encoding, splitting, and metadata.

INTERFACE (frozen – do not change signatures after teammates start using this):
    load_acsincome(state, year, n_max, seed) -> X_train, X_test, y_train, y_test, meta

meta dict keys:
    feature_names : list[str]        – column order used everywhere
    types         : dict[str, str]   – "cont" | "ord" | "cat"
    MAD           : dict[str, float] – median absolute deviation (for proximity scaling)
    ranges        : dict[str, tuple] – (min, max) of each feature in training set
    label_col     : str              – target column name ("PINCP_binary")
    pos_label     : int              – 1 = income > threshold (the "good" outcome)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSIncome

# ---------------------------------------------------------------------------
# Constants (agreed by group – frozen in poc.yaml)
# ---------------------------------------------------------------------------
DEFAULT_STATE  = "CA"
DEFAULT_YEAR   = "2018"
DEFAULT_N_MAX  = 30_000   # subsample cap; keeps runtime manageable
DEFAULT_SEED   = 42
INCOME_THRESH  = 50_000   # Folktables default

# ACSIncome features (order fixed by Folktables)
# AGEP=age, COW=class-of-worker, SCHL=education, MAR=marital,
# OCCP=occupation, POBP=place-of-birth, RELP=relationship,
# WKHP=hrs-worked, SEX=sex, RAC1P=race
_FEATURE_COLS = ["AGEP", "COW", "SCHL", "MAR", "OCCP",
                 "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
_LABEL_COL    = "PINCP_binary"

# Feature type annotation (used by DiCE and Cem's constraints)
# "cont"  – continuous numeric
# "ord"   – ordinal integer (can increment/decrement)
# "cat"   – nominal categorical (no ordering)
_TYPES = {
    "AGEP": "cont",   # age in years – continuous
    "COW":  "cat",    # class of worker – nominal
    "SCHL": "ord",    # educational attainment – ordinal
    "MAR":  "cat",    # marital status – nominal
    "OCCP": "cat",    # occupation code – nominal (500+ codes, grouped below)
    "POBP": "cat",    # place of birth – nominal
    "RELP": "cat",    # relationship – nominal
    "WKHP": "cont",   # usual hours worked per week – continuous
    "SEX":  "cat",    # sex – nominal binary
    "RAC1P":"cat",    # race – nominal
}

# OCCP grouping: 500+ raw codes → ~25 broad SOC groups
# Source: BLS Standard Occupational Classification major groups
_OCCP_GROUP_BREAKS = [
    (0,    99,   "MGT"),      # Management
    (100,  199,  "BUS"),      # Business & Financial
    (200,  299,  "CMM"),      # Computer & Mathematical
    (300,  399,  "ENG"),      # Architecture & Engineering
    (400,  499,  "SCI"),      # Life/Physical/Social Science
    (500,  599,  "CMS"),      # Community & Social Service
    (600,  699,  "LGL"),      # Legal
    (700,  799,  "EDU"),      # Education
    (800,  899,  "ENT"),      # Arts & Entertainment
    (900,  999,  "MED"),      # Healthcare Practitioners
    (1000, 1099, "HLS"),      # Healthcare Support
    (1100, 1199, "PRT"),      # Protective Service
    (1200, 1299, "EAT"),      # Food Preparation
    (1300, 1399, "CLN"),      # Building & Grounds
    (1400, 1499, "PRS"),      # Personal Care
    (1500, 1599, "SAL"),      # Sales
    (1600, 1699, "OFF"),      # Office & Administrative
    (1700, 1799, "FFF"),      # Farming/Fishing/Forestry
    (1800, 1899, "CON"),      # Construction
    (1900, 1999, "EXT"),      # Extraction
    (2000, 2099, "RPR"),      # Installation & Repair
    (2100, 2199, "PRD"),      # Production
    (2200, 2299, "TRN"),      # Transportation
    (2300, 2399, "MIL"),      # Military
    (9000, 9999, "OTH"),      # Other / unknown
]


def _group_occp(series: pd.Series) -> pd.Series:
    """Map raw OCCP codes to broad SOC group strings."""
    def _map(code):
        for lo, hi, label in _OCCP_GROUP_BREAKS:
            if lo <= code <= hi:
                return label
        return "OTH"
    return series.apply(_map)


def _compute_mad(df: pd.DataFrame, features: list[str], types: dict) -> dict[str, float]:
    """Median Absolute Deviation for continuous/ordinal features (used in DiCE proximity)."""
    mad = {}
    for col in features:
        if types[col] in ("cont", "ord"):
            m = np.median(df[col].values)
            mad_val = np.median(np.abs(df[col].values - m))
            mad[col] = float(mad_val) if mad_val > 0 else 1.0  # guard against 0
        else:
            mad[col] = 1.0  # categorical: placeholder (DiCE uses indicator distance)
    return mad


def load_acsincome(
    state:  str = DEFAULT_STATE,
    year:   str = DEFAULT_YEAR,
    n_max:  int = DEFAULT_N_MAX,
    seed:   int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load ACSIncome, subsample, encode, split 80/20, and return arrays + metadata.

    Returns
    -------
    X_train, X_test  : np.ndarray, shape (n_train, d) and (n_test, d)
    y_train, y_test  : np.ndarray, shape (n_train,) and (n_test,), dtype int {0,1}
    meta             : dict  (see module docstring)
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Download / cache ACSIncome features & label
    # ------------------------------------------------------------------
    print(f"[data] Loading ACSIncome {state}/{year} …")
    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
    acs_data    = data_source.get_data(states=[state], download=True)

    features, label, _ = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    # Guard: newer folktables may already return 0/1 binary labels
    # (i.e. PINCP already thresholded). Detect and handle both cases.
    if label.max() <= 1:
        # Already binary
        df[_LABEL_COL] = label.astype(int)
    else:
        df[_LABEL_COL] = (label > INCOME_THRESH).astype(int)

    # Keep only the columns we need
    df = df[_FEATURE_COLS + [_LABEL_COL]].dropna()

    # ------------------------------------------------------------------
    # 2. Group OCCP codes → broad categories
    # ------------------------------------------------------------------
    df["OCCP"] = _group_occp(df["OCCP"].astype(int))

    # ------------------------------------------------------------------
    # 3. Subsample (stratified by label to preserve class balance)
    # Note: pandas 3.x groupby().apply() can drop the groupby key column,
    #       so we subsample by index instead.
    # ------------------------------------------------------------------
    if len(df) > n_max:
        rng_pd = np.random.default_rng(seed)
        # Stratified: sample proportionally from pos and neg separately
        idx_pos = df.index[df[_LABEL_COL] == 1].tolist()
        idx_neg = df.index[df[_LABEL_COL] == 0].tolist()
        n_pos   = int(n_max * len(idx_pos) / len(df))
        n_neg   = n_max - n_pos
        chosen_pos = rng_pd.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
        chosen_neg = rng_pd.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
        chosen     = np.concatenate([chosen_pos, chosen_neg])
        df = df.loc[chosen].reset_index(drop=True)
    print(f"[data] Subsample size: {len(df)}  "
          f"(pos rate: {df[_LABEL_COL].mean():.2%})")

    # ------------------------------------------------------------------
    # 4. Train/test split  (80/20, stratified)
    # ------------------------------------------------------------------
    X_df = df[_FEATURE_COLS]
    y    = df[_LABEL_COL].values.astype(int)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=seed, stratify=y
    )

    # ------------------------------------------------------------------
    # 5. Encode: ordinal/continuous → keep numeric; categorical → int codes
    #    (DiCE needs consistent integer codes for categorical columns)
    # ------------------------------------------------------------------
    # Build category maps from TRAINING set only (avoid leakage)
    cat_cols  = [c for c in _FEATURE_COLS if _TYPES[c] == "cat"]
    cat_maps  = {}   # col -> {string_val: int_code}

    for col in cat_cols:
        cats = sorted(X_train_df[col].unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}

    def _encode(df_part: pd.DataFrame) -> np.ndarray:
        out = df_part.copy()
        for col in cat_cols:
            # unseen categories in test → map to 0 (rare edge case)
            out[col] = out[col].map(cat_maps[col]).fillna(0).astype(int)
        return out.values.astype(float)

    X_train = _encode(X_train_df)
    X_test  = _encode(X_test_df)

    # ------------------------------------------------------------------
    # 6. Compute metadata
    # ------------------------------------------------------------------
    train_df_encoded = pd.DataFrame(X_train, columns=_FEATURE_COLS)

    ranges = {
        col: (float(train_df_encoded[col].min()),
              float(train_df_encoded[col].max()))
        for col in _FEATURE_COLS
    }
    mad = _compute_mad(train_df_encoded, _FEATURE_COLS, _TYPES)

    meta = {
        "feature_names": _FEATURE_COLS,
        "types":         _TYPES,
        "MAD":           mad,
        "ranges":        ranges,
        "label_col":     _LABEL_COL,
        "pos_label":     1,
        "cat_maps":      cat_maps,     # col -> {original_val: int_code}
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "seed":          seed,
    }

    print(f"[data] X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"[data] Features: {_FEATURE_COLS}")
    return X_train, X_test, y_train, y_test, meta


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, meta = load_acsincome()
    print("\nMeta keys:", list(meta.keys()))
    print("Types:", meta["types"])
    print("MAD sample:", {k: round(v, 2) for k, v in list(meta["MAD"].items())[:4]})
    print("Sample X_train[0]:", X_tr[0])
