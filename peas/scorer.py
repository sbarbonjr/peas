"""
PEASScorer — top-level API.

Combines a favorable evidence source (mu), a contrary evidence source
(lambda), and a PAL2v threshold policy into a single audit object.

Quickstart
----------
    from peas import PEASScorer
    from peas.evidence import FaithfulnessEvidence, StabilityEvidence

    scorer = PEASScorer(
        mu_source=FaithfulnessEvidence(),
        lambda_source=StabilityEvidence(),
    )
    result = scorer.audit(model, X_train, X_eval, y_eval, ranking, k=5)
    print(result)
    # [RELIABLE] mu=0.847  lambda=0.193  certainty=+0.654  contradiction=-0.040
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from peas.evidence.base import EvidenceSource
from peas.logic.pal2v import PAL2vDiagnostic, AuditResult
from peas.logic.threshold import ThresholdPolicy, FixedThreshold


class PEASScorer:
    """
    Paraconsistent Explanation Audit System scorer.

    Parameters
    ----------
    mu_source : EvidenceSource
        Computes favorable evidence mu in [0, 1].
    lambda_source : EvidenceSource
        Computes contrary evidence lambda in [0, 1].
    threshold : ThresholdPolicy | None
        Threshold policy for the PAL2v diagnostic layer.
        Defaults to FixedThreshold(0.5).
    k : int
        Default number of top features to evaluate.

    Notes
    -----
    mu and lambda are computed from structurally independent evidence families,
    so mu + lambda != 1 in general. This is the key property that allows all
    four PAL2v diagnostic states to be reachable.
    """

    def __init__(
        self,
        mu_source: EvidenceSource,
        lambda_source: EvidenceSource,
        threshold: ThresholdPolicy | None = None,
        k: int = 5,
    ) -> None:
        self.mu_source = mu_source
        self.lambda_source = lambda_source
        self.diagnostic = PAL2vDiagnostic(threshold or FixedThreshold(0.5))
        self.k = k

    def audit(
        self,
        model,
        X_train: pd.DataFrame,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        ranking: pd.DataFrame,
        k: int | None = None,
        seed: int = 0,
    ) -> AuditResult:
        """
        Audit an explanation ranking and return the PAL2v diagnostic state.

        Parameters
        ----------
        model :
            Fitted scikit-learn compatible estimator.
        X_train : pd.DataFrame
        X_eval : pd.DataFrame
        y_eval : pd.Series
        ranking : pd.DataFrame
            Must contain columns ['feature', 'rank'].
        k : int | None
            Number of top features. Falls back to self.k.
        seed : int

        Returns
        -------
        AuditResult
        """
        k = k or self.k
        mu  = self.mu_source.score(model, X_train, X_eval, y_eval, ranking, k, seed)
        lam = self.lambda_source.score(model, X_train, X_eval, y_eval, ranking, k, seed)
        return self.diagnostic.evaluate(mu, lam)

    def audit_precomputed(self, mu: float, lam: float) -> AuditResult:
        """
        Audit from pre-computed (mu, lambda) values.

        Useful when evidence scores are computed externally or in bulk.
        """
        return self.diagnostic.evaluate(
            float(np.clip(mu, 0.0, 1.0)),
            float(np.clip(lam, 0.0, 1.0)),
        )

    def __repr__(self) -> str:
        return (
            f"PEASScorer(mu={self.mu_source.name!r}, "
            f"lambda={self.lambda_source.name!r}, "
            f"threshold={self.diagnostic.threshold!r})"
        )
