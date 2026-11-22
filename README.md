# ActCF: Actionable and Robust Counterfactual Explanations
### ENS 491/492 — Proof, Implementation and Research Design Proposal

This repository contains the proof-of-concept (POC) implementation for **ActCF** (Actionable and Robust Counterfactual Explanations).

---

## Repository Structure

```
actcf/
├── actcf/                  # Core library modules
│   ├── data.py             # ACSIncome data loader, OCCP grouping, MAD scaling
│   ├── models.py           # Logistic Regression baseline & bootstrap models
│   ├── generators.py       # DiCE counterfactual generator & query selector
│   ├── constraints.py      # Actionability & feasibility constraints (Furkan)
│   ├── robustness.py       # Input & model retraining robustness (Arman)
│   └── metrics.py          # Recourse evaluation metrics (Ilgın)
├── configs/
│   └── poc.yaml            # Shared frozen experiment configuration
├── experiments/
│   ├── queries.csv         # 50 frozen test queries (negative instances)
│   └── run_poc.py          # End-to-end integration runner
├── results/
│   └── poc_results.csv     # Preliminary experiment output metrics
├── tests/
│   └── test_hand_calc.py   # Analytical closed-form verification tests (§5.2)
└── requirements.txt
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Verification Tests (§5.2)
```bash
python tests/test_hand_calc.py
```

### 3. Run End-to-End POC Experiment
```bash
python experiments/run_poc.py
```
Outputs are written to `results/poc_results.csv`.
