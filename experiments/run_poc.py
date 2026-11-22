"""
run_poc.py  –  Candan Tuna Mesut (integration entry point)
Proof-of-concept: end-to-end pipeline for 50 queries.

Steps (matches assignment §5.1):
  1. load_acsincome → train_lr
  2. dice_generate for 50 queries
  3. check → violation rates (Cem)
  4. repair + filter (Cem)
  5. validity, proximity, action_cost, sparsity (Fatma/Cem stubs for now)
  6. ir_monte_carlo, var_score (Arman stub for now)
  7. Comparison: raw DiCE vs DiCE + constraint filter

Usage:
    cd actcf/
    python experiments/run_poc.py

Results written to results/poc_results.csv
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "actcf"))

import numpy as np
import pandas as pd

from data        import load_acsincome
from models      import train_lr, train_bootstrap_models
from generators  import dice_generate, build_queries_csv
from constraints import load_rules, mutable_mask, check, repair, action_cost
from robustness  import ir_monte_carlo, var_score, rank
from metrics     import validity, proximity, sparsity

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = dict(
    state="CA", year="2018", n_max=30_000, seed=42,
    k=5, n_queries=50, tau=0.5,
    rules_path=os.path.join(os.path.dirname(__file__), "..", "actcf", "constraints.yaml"),
)


def main():
    t0 = time.time()
    print("=" * 60)
    print("ActCF Proof-of-Concept Run")
    print("=" * 60)

    # ── Step 1: Data + Model ─────────────────────────────────────────
    X_tr, X_te, y_tr, y_te, meta = load_acsincome(
        state=CONFIG["state"], year=CONFIG["year"],
        n_max=CONFIG["n_max"], seed=CONFIG["seed"],
    )
    model = train_lr(X_tr, y_tr, seed=CONFIG["seed"])
    print(f"  Test acc: {model.score(X_te, y_te):.4f}")

    boot_models = train_bootstrap_models(X_tr, y_tr, n=10, seed0=CONFIG["seed"])

    # ── Constraints ──────────────────────────────────────────────────
    rules = load_rules(CONFIG["rules_path"])
    mask  = mutable_mask(rules, meta)
    print(f"  Mutable features: {mask.sum()} / {len(mask)}")

    # ── Step 2: Select queries ────────────────────────────────────────
    queries_path = os.path.join(RESULTS_DIR, "..", "experiments", "queries.csv")
    queries = build_queries_csv(
        X_te, y_te, model,
        n_queries=CONFIG["n_queries"], seed=CONFIG["seed"],
        out_path=queries_path,
    )
    print(f"  Queries: {queries.shape}")

    # ── Steps 3–7: Per-query loop ─────────────────────────────────────
    rows = []
    for qi, x in enumerate(queries):
        # Generate raw DiCE CFs
        X_cf_raw = dice_generate(
            model=model, x=x, k=CONFIG["k"], meta=meta,
            X_train=X_tr, y_train=y_tr, method="random",
            mutable_mask=mask, seed=CONFIG["seed"] + qi,
        )

        # Step 3: Check violations
        viol_df   = check(x, X_cf_raw, rules)
        n_violated = viol_df["violated"].sum()

        # Step 4: Repair + filter
        X_cf_fixed = repair(x, X_cf_raw, rules)

        # Step 5: Metrics on raw
        val_raw  = validity(model, X_cf_raw)
        prox_raw = proximity(x, X_cf_raw, meta)
        spar_raw = sparsity(x, X_cf_raw)
        cost_raw = action_cost(x, X_cf_raw, rules, meta)

        # Metrics on repaired
        val_fix  = validity(model, X_cf_fixed)
        prox_fix = proximity(x, X_cf_fixed, meta)
        spar_fix = sparsity(x, X_cf_fixed)

        # Step 6: Robustness (Arman stubs for now)
        ir_scores  = ir_monte_carlo(model, X_cf_fixed, mask, seed=CONFIG["seed"] + qi)
        var_scores = var_score(X_cf_fixed, boot_models)

        # Step 7: Rank + select top CFs
        X_top = rank(x, X_cf_fixed, cost=cost_raw, ir=ir_scores,
                     agreement=var_scores, k=3)

        rows.append({
            "query_id":        qi,
            "val_raw":         val_raw,
            "val_fixed":       val_fix,
            "prox_raw":        prox_raw,
            "prox_fixed":      prox_fix,
            "sparsity_raw":    spar_raw,
            "sparsity_fixed":  spar_fix,
            "n_violations":    int(n_violated),
            "ir_mean":         float(ir_scores.mean()),
            "var_mean":        float(var_scores.mean()),
        })

        if qi % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Query {qi:3d}/{CONFIG['n_queries']} — "
                  f"val_raw={val_raw:.2f}  val_fixed={val_fix:.2f}  "
                  f"violations={int(n_violated)}  ({elapsed:.0f}s elapsed)")

    # ── Save results ──────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "poc_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[run_poc] Results saved → {out_path}")
    print("\n── Summary ─────────────────────────────────────────────────")
    print(df[["val_raw","val_fixed","prox_raw","prox_fixed",
              "sparsity_raw","n_violations"]].describe().round(3).to_string())
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
