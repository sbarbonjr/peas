"""
Threshold policies for the PAL2v diagnostic layer.

The threshold theta governs which (mu, lambda) pairs fall into each of the
four diagnostic quadrants. Three policies are provided:

  FixedThreshold        — conventional symmetric theta = 0.5
  AsymmetricThreshold   — independent thresholds for mu and lambda
  CalibratedThreshold   — theta set to the median of observed scores

The threshold is not a mathematical constant of PAL2v; it is a design
parameter that should be calibrated to the evaluation context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ThresholdPolicy(ABC):
    """Abstract base for threshold policies."""

    @abstractmethod
    def thresholds(self, mu: float, lam: float) -> tuple[float, float]:
        """Return (theta_mu, theta_lambda) for the given evidence pair."""


class FixedThreshold(ThresholdPolicy):
    """
    Symmetric fixed threshold (conventional PAL2v default).

    Parameters
    ----------
    theta : float
        Threshold applied to both mu and lambda. Default 0.5.
    """

    def __init__(self, theta: float = 0.5) -> None:
        if not 0.0 < theta < 1.0:
            raise ValueError(f"theta must be in (0, 1), got {theta}")
        self.theta = theta

    def thresholds(self, mu: float, lam: float) -> tuple[float, float]:
        return self.theta, self.theta

    def __repr__(self) -> str:
        return f"FixedThreshold(theta={self.theta})"


class AsymmetricThreshold(ThresholdPolicy):
    """
    Independent thresholds for favorable and contrary evidence.

    Useful when the evaluation context tolerates instability differently
    from low faithfulness (e.g. theta_mu=0.6, theta_lam=0.4 tightens the
    faithfulness requirement while relaxing the instability one).

    Parameters
    ----------
    theta_mu : float
        Threshold for favorable evidence mu.
    theta_lam : float
        Threshold for contrary evidence lambda.
    """

    def __init__(self, theta_mu: float = 0.5, theta_lam: float = 0.5) -> None:
        self.theta_mu = theta_mu
        self.theta_lam = theta_lam

    def thresholds(self, mu: float, lam: float) -> tuple[float, float]:
        return self.theta_mu, self.theta_lam

    def __repr__(self) -> str:
        return f"AsymmetricThreshold(theta_mu={self.theta_mu}, theta_lam={self.theta_lam})"


class CalibratedThreshold(ThresholdPolicy):
    """
    Data-driven threshold set to the median of observed (mu, lambda) scores.

    After calling fit() on a collection of (mu, lambda) pairs, thresholds()
    returns the per-axis medians. This grounds the diagnostic boundary in the
    empirical distribution of the evaluation context rather than a fixed value.

    Parameters
    ----------
    default : float
        Fallback threshold used before fit() is called.
    """

    def __init__(self, default: float = 0.5) -> None:
        self.default = default
        self._theta_mu: float = default
        self._theta_lam: float = default
        self._fitted: bool = False

    def fit(self, mus: list[float], lams: list[float]) -> "CalibratedThreshold":
        """
        Calibrate thresholds to the median of observed evidence scores.

        Parameters
        ----------
        mus : list[float]
            Observed favorable evidence scores.
        lams : list[float]
            Observed contrary evidence scores.
        """
        self._theta_mu  = float(np.median(mus))
        self._theta_lam = float(np.median(lams))
        self._fitted = True
        return self

    def thresholds(self, mu: float, lam: float) -> tuple[float, float]:
        return self._theta_mu, self._theta_lam

    def __repr__(self) -> str:
        status = f"theta_mu={self._theta_mu:.3f}, theta_lam={self._theta_lam:.3f}" if self._fitted else "unfitted"
        return f"CalibratedThreshold({status})"
