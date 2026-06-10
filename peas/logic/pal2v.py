"""
PAL2v diagnostic layer.

Maps a pair (mu, lambda) in [0,1]^2 to one of four diagnostic states
defined by Paraconsistent Annotated Logic with two values (PAL2v):

    State           mu        lambda    Meaning
    ─────────────── ───────── ───────── ──────────────────────────────────
    reliable        >= theta  < theta   Faithful and stable
    contradictory   >= theta  >= theta  Faithful but unstable
    rejected        < theta   >= theta  Unfaithful (instability dominates)
    inconclusive    < theta   < theta   Insufficient evidence

References
----------
  Da Costa & Marconi (1989); Abe et al. (1992)
  Pena et al. (2017) — PAL2v applied to anomaly detection
"""

from __future__ import annotations

from dataclasses import dataclass

from peas.logic.threshold import ThresholdPolicy, FixedThreshold


@dataclass
class AuditResult:
    """Full PEAS audit result for a single explanation."""

    mu: float
    lam: float
    certainty: float
    contradiction: float
    diagnostic: str

    def __str__(self) -> str:
        return (
            f"[{self.diagnostic.upper()}] "
            f"mu={self.mu:.3f}  lambda={self.lam:.3f}  "
            f"certainty={self.certainty:+.3f}  contradiction={self.contradiction:+.3f}"
        )


class PAL2vDiagnostic:
    """
    PAL2v diagnostic layer.

    Parameters
    ----------
    threshold : ThresholdPolicy
        Policy that determines theta for a given (mu, lambda) pair.
        Defaults to FixedThreshold(0.5).
    """

    STATES = ("reliable", "contradictory", "rejected", "inconclusive")

    def __init__(self, threshold: ThresholdPolicy | None = None) -> None:
        self.threshold = threshold or FixedThreshold(0.5)

    def evaluate(self, mu: float, lam: float) -> AuditResult:
        """
        Assign a PAL2v diagnostic state to the (mu, lambda) pair.

        Parameters
        ----------
        mu : float
            Favorable evidence in [0, 1].
        lam : float
            Contrary evidence in [0, 1].

        Returns
        -------
        AuditResult
        """
        theta_mu, theta_lam = self.threshold.thresholds(mu, lam)

        if mu >= theta_mu and lam < theta_lam:
            state = "reliable"
        elif mu >= theta_mu and lam >= theta_lam:
            state = "contradictory"
        elif mu < theta_mu and lam >= theta_lam:
            state = "rejected"
        else:
            state = "inconclusive"

        return AuditResult(
            mu=mu,
            lam=lam,
            certainty=mu - lam,
            contradiction=mu + lam - 1.0,
            diagnostic=state,
        )

    def __repr__(self) -> str:
        return f"PAL2vDiagnostic(threshold={self.threshold!r})"
