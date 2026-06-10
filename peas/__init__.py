"""
PEAS — Paraconsistent Explanation Audit System
"""

from peas.logic.pal2v import PAL2vDiagnostic
from peas.logic.threshold import FixedThreshold, CalibratedThreshold, AsymmetricThreshold
from peas.scorer import PEASScorer
from peas.evidence.base import EvidenceSource

__all__ = [
    "PEASScorer",
    "EvidenceSource",
    "PAL2vDiagnostic",
    "FixedThreshold",
    "CalibratedThreshold",
    "AsymmetricThreshold",
]

__version__ = "0.1.0"
