"""
Custom Evidence Sources — plug in your own mu and lambda.

Demonstrates how to extend EvidenceSource with:
  1. CompactnessEvidence  — favorable evidence from explanation sparsity
  2. InterMethodAgreement — favorable evidence from SHAP vs permutation agreement
  3. SubgroupRobustness   — contrary evidence from performance gap across subgroups

Run as a script or open as a Jupyter notebook (jupytext compatible).
"""

# %% [markdown]
# # Custom Evidence Sources
#
# PEAS is built around two ABCs:
# - `EvidenceSource.score()` → float in [0, 1]
#
# This notebook shows three real extension examples.

# %% Imports
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from peas import PEASScorer
from peas.evidence.base import EvidenceSource
from peas.logic import FixedThreshold

SEED = 42
K = 5

bunch = load_breast_cancer()
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
y = pd.Series(bunch.target)
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train, y_train)

# %% 1. Compactness Evidence (favorable mu)
class CompactnessEvidence(EvidenceSource):
    """
    Favorable evidence: explanations that concentrate predictive power in
    fewer features are more compact, hence more trustworthy.

    Score = 1 - (k / n_features), capped to [0, 1].
    """
    def __init__(self):
        super().__init__(name="compactness")

    def score(self, model, X_train, X_eval, y_eval, ranking, k, seed=0):
        n = len(ranking)
        return self._clip(1.0 - k / n)


# %% 2. Inter-method Agreement (favorable mu)
class InterMethodAgreement(EvidenceSource):
    """
    Favorable evidence: high Spearman rank correlation between two independently
    derived rankings (e.g. SHAP and permutation importance).

    Requires a second ranking to be passed via extra_ranking at construction time.
    """
    def __init__(self, extra_ranking: pd.DataFrame):
        super().__init__(name="inter_method_agreement")
        self.extra_ranking = extra_ranking

    def score(self, model, X_train, X_eval, y_eval, ranking, k, seed=0):
        from scipy.stats import spearmanr
        merged = ranking.merge(self.extra_ranking, on="feature", suffixes=("_a", "_b"))
        if len(merged) < 2:
            return 0.5
        rho, _ = spearmanr(merged["rank_a"], merged["rank_b"])
        return self._clip((float(rho) + 1.0) / 2.0)  # map [-1,1] → [0,1]


# %% 3. Subgroup Robustness (contrary lambda)
class SubgroupRobustness(EvidenceSource):
    """
    Contrary evidence: if the model's balanced accuracy differs substantially
    across two feature-defined subgroups, the explanation is less trustworthy.

    Score = performance gap between subgroups, normalised to [0, 1].
    """
    def __init__(self, subgroup_feature: str, threshold: float):
        super().__init__(name="subgroup_robustness")
        self.subgroup_feature = subgroup_feature
        self.threshold = threshold

    def score(self, model, X_train, X_eval, y_eval, ranking, k, seed=0):
        if self.subgroup_feature not in X_eval.columns:
            return 0.0
        mask = X_eval[self.subgroup_feature] >= self.threshold
        if mask.sum() == 0 or (~mask).sum() == 0:
            return 0.0
        acc_a = balanced_accuracy_score(y_eval[mask],  model.predict(X_eval[mask]))
        acc_b = balanced_accuracy_score(y_eval[~mask], model.predict(X_eval[~mask]))
        return self._clip(abs(acc_a - acc_b))


# %% Build rankings for demonstration
perm = permutation_importance(model, X_eval, y_eval, n_repeats=10, random_state=SEED)
ranking_perm = pd.DataFrame({
    "feature": X_eval.columns,
    "importance_score": perm.importances_mean,
}).sort_values("importance_score", ascending=False).reset_index(drop=True)
ranking_perm["rank"] = range(1, len(ranking_perm) + 1)

# Simulate a second ranking (e.g. from SHAP or RF built-in)
rng = np.random.default_rng(SEED + 1)
ranking_alt = ranking_perm.copy()
ranking_alt["rank"] = rng.permutation(len(ranking_alt)) + 1
ranking_alt = ranking_alt.sort_values("rank").reset_index(drop=True)

# %% Audit with custom sources
scorer = PEASScorer(
    mu_source=InterMethodAgreement(extra_ranking=ranking_alt),
    lambda_source=SubgroupRobustness(
        subgroup_feature="mean radius",
        threshold=float(X_eval["mean radius"].median()),
    ),
    threshold=FixedThreshold(0.5),
    k=K,
)

result = scorer.audit(model, X_train, X_eval, y_eval, ranking_perm, k=K, seed=SEED)
print(f"\n{'='*55}")
print("  Custom Evidence Audit (agreement + subgroup robustness)")
print(f"{'='*55}")
print(f"  {result}")

# %% Compactness as mu alongside subgroup robustness as lambda
scorer2 = PEASScorer(
    mu_source=CompactnessEvidence(),
    lambda_source=SubgroupRobustness(
        subgroup_feature="mean radius",
        threshold=float(X_eval["mean radius"].median()),
    ),
    threshold=FixedThreshold(0.5),
    k=K,
)
result2 = scorer2.audit(model, X_train, X_eval, y_eval, ranking_perm, k=K, seed=SEED)
print(f"\n  Compactness + Subgroup Robustness:")
print(f"  {result2}")
