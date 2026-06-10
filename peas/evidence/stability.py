"""
Stability-based contrary evidence source (lambda).

Contrary evidence lambda is derived from two complementary instability signals:

  low_Stability     = 1 - rank_stability
                      where rank_stability is the mean Spearman rho across
                      n_runs independent model re-initialisations
                      (Alvarez-Melis & Jaakkola, 2018).

  RandomSimilarity  = 1 - BaselineGain
                      where BaselineGain is the empirical percentile of the
                      method's composite faithfulness score relative to
                      randomly permuted rankings; high similarity to random
                      indicates low information content.

Both signals are causally independent from the faithfulness metrics that
form mu, so mu + lambda != 1 in general, preserving the full paraconsistent
geometry across all evaluation profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from peas.evidence.base import EvidenceSource


class StabilityEvidence(EvidenceSource):
    """
    Contrary evidence lambda derived from ranking instability.

    Parameters
    ----------
    n_runs : int
        Number of independent model re-initialisations for stability estimation.
    weights : dict[str, float] | None
        Relative weights for {'low_Stability', 'RandomSimilarity'}.
        Defaults to equal weighting.
    """

    def __init__(
        self,
        n_runs: int = 10,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(name="stability")
        self.n_runs = n_runs
        self.weights = weights or {"low_Stability": 1.0, "RandomSimilarity": 1.0}

    def score(self, model, X_train, X_eval, y_eval, ranking, k, seed=0):
        raise NotImplementedError(
            "StabilityEvidence.score() requires pre-computed stability values. "
            "Use from_precomputed() or supply stability and random_similarity directly."
        )

    @classmethod
    def from_precomputed(
        cls,
        stability: float,
        random_similarity: float,
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Compute lambda from pre-computed stability and random similarity values.

        Parameters
        ----------
        stability : float
            Mean Spearman rho across model re-initialisations in [0, 1].
        random_similarity : float
            Similarity of the ranking to random permutations in [0, 1].
        weights : dict | None
            Optional custom weights.

        Returns
        -------
        float
            Contrary evidence lambda in [0, 1].
        """
        w = weights or {"low_Stability": 1.0, "RandomSimilarity": 1.0}
        signals = {
            "low_Stability":    float(np.clip(1.0 - stability, 0.0, 1.0)),
            "RandomSimilarity": float(np.clip(random_similarity, 0.0, 1.0)),
        }
        total_w = sum(w.values())
        return float(np.clip(sum(signals[k] * v for k, v in w.items()) / total_w, 0.0, 1.0))


def ranking_stability(rankings: list[pd.DataFrame]) -> float:
    """
    Compute mean pairwise Spearman rank correlation across a list of rankings.

    Parameters
    ----------
    rankings : list of pd.DataFrame
        Each DataFrame must have columns ['feature', 'rank'].

    Returns
    -------
    float
        Mean absolute Spearman rho across all pairs, in [0, 1].
    """
    if len(rankings) < 2:
        return 1.0
    rhos = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            merged = rankings[i].merge(rankings[j], on="feature", suffixes=("_i", "_j"))
            if len(merged) < 2:
                continue
            rho, _ = spearmanr(merged["rank_i"], merged["rank_j"])
            rhos.append(abs(rho))
    return float(np.mean(rhos)) if rhos else 1.0
