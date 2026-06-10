"""
Abstract base class for PEAS evidence sources.

Any measurable property of an explanation that can be normalised to [0, 1]
can serve as a favorable (mu) or contrary (lambda) evidence source.
Subclass EvidenceSource and implement score() to plug in custom evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class EvidenceSource(ABC):
    """
    Base class for evidence sources used in PEAS.

    An evidence source takes a trained model, the training data, an evaluation
    split, and a feature ranking produced by an XAI method, and returns a
    scalar in [0, 1] representing the strength of the evidence signal.

    Parameters
    ----------
    name : str
        Human-readable identifier shown in audit reports.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def score(
        self,
        model,
        X_train: pd.DataFrame,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        ranking: pd.DataFrame,
        k: int,
        seed: int = 0,
    ) -> float:
        """
        Compute the evidence score for a given explanation ranking.

        Parameters
        ----------
        model :
            A fitted scikit-learn compatible estimator.
        X_train : pd.DataFrame
            Training features (used for baseline construction).
        X_eval : pd.DataFrame
            Evaluation features.
        y_eval : pd.Series
            Evaluation labels.
        ranking : pd.DataFrame
            Feature ranking with columns ['feature', 'rank', 'importance_score'].
        k : int
            Number of top features to evaluate.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        float
            Evidence score in [0, 1].
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))
