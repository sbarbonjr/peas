"""
PEAS Quickstart — 5-minute walkthrough.

Run as a script or open as a Jupyter notebook (jupytext compatible).
"""

# %% [markdown]
# # PEAS Quickstart
#
# This notebook walks through a minimal PEAS audit:
# 1. Train a Random Forest on Breast Cancer
# 2. Generate a feature ranking with permutation importance
# 3. Estimate stability across 10 model re-initialisations
# 4. Audit with PEASScorer and read the diagnostic state
# 5. Plot the result on the diagnostic plane

# %% Imports
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from peas import PEASScorer
from peas.evidence import FaithfulnessEvidence
from peas.evidence.stability import StabilityEvidence, ranking_stability
from peas.logic import FixedThreshold

SEED = 42
K = 5

# %% Load data
bunch = load_breast_cancer()
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
y = pd.Series(bunch.target)
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape}  Eval: {X_eval.shape}")

# %% Train model and generate ranking via permutation importance
model = RandomForestClassifier(n_estimators=100, random_state=SEED)
model.fit(X_train, y_train)

perm = permutation_importance(model, X_eval, y_eval, n_repeats=10, random_state=SEED)
ranking = pd.DataFrame({
    "feature": X_eval.columns,
    "importance_score": perm.importances_mean,
}).sort_values("importance_score", ascending=False).reset_index(drop=True)
ranking["rank"] = range(1, len(ranking) + 1)

print("\nTop 5 features:")
print(ranking.head(K)[["rank", "feature", "importance_score"]].to_string(index=False))

# %% Estimate stability across 10 model re-initialisations
rankings = []
for s in range(SEED, SEED + 10):
    m = RandomForestClassifier(n_estimators=100, random_state=s).fit(X_train, y_train)
    p = permutation_importance(m, X_eval, y_eval, n_repeats=5, random_state=s)
    r = pd.DataFrame({"feature": X_eval.columns, "importance_score": p.importances_mean})
    r = r.sort_values("importance_score", ascending=False).reset_index(drop=True)
    r["rank"] = range(1, len(r) + 1)
    rankings.append(r)

stability = ranking_stability(rankings)
random_similarity = 1.0 - stability   # simplified proxy
lam = StabilityEvidence.from_precomputed(stability, random_similarity)
print(f"\nStability (mean Spearman rho): {stability:.4f}")
print(f"Lambda (contrary evidence):    {lam:.4f}")

# %% Compute mu and run the audit
scorer = PEASScorer(
    mu_source=FaithfulnessEvidence(random_repetitions=20),
    lambda_source=StabilityEvidence(),
    threshold=FixedThreshold(0.5),
    k=K,
)
mu = scorer.mu_source.score(model, X_train, X_eval, y_eval, ranking, K, SEED)
result = scorer.audit_precomputed(mu=mu, lam=lam)

print(f"\n{'='*50}")
print(f"  PEAS Audit Result")
print(f"{'='*50}")
print(f"  {result}")

# %% Plot the diagnostic plane
try:
    from peas.utils.plot import plot_diagnostic_plane
    import matplotlib.pyplot as plt

    ax = plot_diagnostic_plane(
        results=[{"method": "RandomForest + Permutation",
                  "mu": result.mu, "lam": result.lam,
                  "diagnostic": result.diagnostic}],
        theta=0.5,
        title="PEAS Diagnostic Plane — Breast Cancer",
    )
    plt.tight_layout()
    plt.savefig("diagnostic_plane_quickstart.png", dpi=150)
    print("\nPlot saved to diagnostic_plane_quickstart.png")
    plt.show()
except ImportError:
    print("\nInstall matplotlib for plotting: pip install matplotlib")
