# Candan Tuna Mesut — Presentation Slides (2–3 Slides)
## ENS 491/492 ActCF Phase 2

---

### Slide 1: Mathematical Foundations & DiCE Derivation
**Title:** DiCE Foundations & Why Classical Counterfactuals Fail
**Key Talking Points:**
- **The Optimization Formulation:**
  $$\min_{\mathbf{c}_1, \dots, \mathbf{c}_k} \frac{1}{k}\sum_{i=1}^k \mathcal{L}_v(f(\mathbf{c}_i), y^*) + \frac{\lambda_1}{k}\sum_{i=1}^k \text{dist}(\mathbf{c}_i, \mathbf{x}) - \lambda_2 \det(\mathbf{K})$$
- **Hinge Loss on Logits:** Pushes candidate points just across the margin ($z=0$ for $\tau=0.5$) and ceases gradient pressure.
  - *Consequence:* Candidates cluster immediately along the boundary $\implies$ highly fragile when models undergo retraining drift.
- **Penalty Relaxation:** Converting the hard feasibility condition $f(\mathbf{c}) \ge \tau$ into a soft Lagrange penalty makes optimization smooth, but **surrenders theoretical validity guarantees**.
- **The Trade-Off Paradox ($\lambda_1$ vs $\lambda_2$):**
  - $\lambda_1 \uparrow$ (proximity) $\implies$ validity drops.
  - $\lambda_2 \uparrow$ (DPP diversity) $\implies$ counterfactuals stray far from input, incurring unrealistic user effort.

---

### Slide 2: Optimization Solvability & The Verification Oracle
**Title:** Model Solvability & The Hand-Calculated Verification Oracle
**Key Talking Points:**
- **Model Geometry & Solvability:**
  - **Logistic Regression (LR):** Linear boundary $\implies$ LP / MILP with global certificate of optimality $\implies$ **Verification Oracle**.
  - **Tree Ensembles (RF):** $\nabla_{\mathbf{x}} f = \mathbf{0}$ everywhere $\implies$ NP-hard combinatorial leaf search; requires stochastic/random sampling.
  - **Neural Networks (MLP):** Differentiable but non-convex $\implies$ local optima only.
- **Verification Experiment (§5.2):**
  - Analytically derived the exact single-feature closed-form minimum perturbation:
    $$\delta_{j^*} = -\frac{z(\mathbf{x})\sigma_{j^*}}{w_{j^*}}, \quad j^* = \arg\max \frac{|w_j|\text{MAD}_j}{\sigma_j}$$
  - **Result:** Exact closed-form reaches $P(y=1) = \mathbf{0.5000}$ with proximity **14.05**, while DiCE heuristic search achieves $P=0.58$ with proximity **18.18**.
  - Confirms data pipeline precision and proves heuristic search remains within a bounded factor ($1.29\times$) of true optimality.

---

### Slide 3: Architectural Decision & Shared Infrastructure
**Title:** Architectural Decision 1 & Integration Baseline
**Key Talking Points:**
- **Decision 1: DiCE-based Generator vs. New Engine:**
  - *Choice:* Hybrid architecture — leverage DiCE's open-source model-agnostic random sampler for rapid exploration, and construct downstream filtering for causal constraints and model robustness.
  - *Defense:* The literature bottleneck is not sampling candidate points, but verifying feasibility and retraining stability. Falsifiable via candidate yield rate.
- **Shared Delivery for the Team:**
  - `data.py`: ACSIncome CA/2018 (30k subsample, 10 features, OCCP grouped into SOC clusters, MAD scaling).
  - `models.py`: Standard LR ($77.6\%$ acc) + 10 bootstrap models for Arman's retraining tests.
  - `queries.csv`: 50 frozen negative test queries for reproducible evaluation across all team members.
  - Runtime: Full 50-query POC executes end-to-end in **48 seconds**.
