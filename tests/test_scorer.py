"""Basic sanity tests for PEASScorer and PAL2vDiagnostic."""

import pytest
from peas.logic.pal2v import PAL2vDiagnostic
from peas.logic.threshold import FixedThreshold, AsymmetricThreshold, CalibratedThreshold
from peas import PEASScorer
from peas.evidence.base import EvidenceSource


class ConstantEvidence(EvidenceSource):
    """Stub evidence source that always returns a fixed value."""
    def __init__(self, value: float):
        super().__init__(name=f"constant_{value}")
        self.value = value

    def score(self, model, X_train, X_eval, y_eval, ranking, k, seed=0):
        return self.value


@pytest.mark.parametrize("mu,lam,expected", [
    (0.8, 0.2, "reliable"),
    (0.8, 0.8, "contradictory"),
    (0.2, 0.8, "rejected"),
    (0.2, 0.2, "inconclusive"),
    (0.5, 0.4, "reliable"),     # boundary: mu==theta → reliable
    (0.4, 0.5, "rejected"),     # boundary: lam==theta → rejected
])
def test_diagnostic_states(mu, lam, expected):
    diag = PAL2vDiagnostic(FixedThreshold(0.5))
    result = diag.evaluate(mu, lam)
    assert result.diagnostic == expected


def test_certainty_and_contradiction():
    diag = PAL2vDiagnostic()
    result = diag.evaluate(0.8, 0.3)
    assert abs(result.certainty - (0.8 - 0.3)) < 1e-9
    assert abs(result.contradiction - (0.8 + 0.3 - 1.0)) < 1e-9


def test_asymmetric_threshold():
    diag = PAL2vDiagnostic(AsymmetricThreshold(theta_mu=0.7, theta_lam=0.3))
    assert diag.evaluate(0.65, 0.2).diagnostic == "inconclusive"  # mu < theta_mu
    assert diag.evaluate(0.75, 0.2).diagnostic == "reliable"


def test_calibrated_threshold():
    threshold = CalibratedThreshold()
    threshold.fit(mus=[0.8, 0.9, 0.7, 0.85], lams=[0.2, 0.3, 0.1, 0.25])
    diag = PAL2vDiagnostic(threshold)
    result = diag.evaluate(0.9, 0.1)
    assert result.diagnostic == "reliable"


def test_precomputed_audit():
    scorer = PEASScorer(
        mu_source=ConstantEvidence(0.8),
        lambda_source=ConstantEvidence(0.2),
    )
    result = scorer.audit_precomputed(mu=0.8, lam=0.2)
    assert result.diagnostic == "reliable"
    assert abs(result.certainty - 0.6) < 1e-9


def test_clipping():
    scorer = PEASScorer(
        mu_source=ConstantEvidence(0.8),
        lambda_source=ConstantEvidence(0.2),
    )
    result = scorer.audit_precomputed(mu=1.5, lam=-0.1)
    assert result.mu == 1.0
    assert result.lam == 0.0
