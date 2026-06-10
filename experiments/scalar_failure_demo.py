"""
Demonstrates the scalar aggregation failure case from the XKDD 2026 paper
(Table 3 — PEAS versus a single score).

Two configurations have identical single scores s = (1 + certainty) / 2,
yet PEAS assigns different diagnostic states because it preserves the
individual magnitudes of mu and lambda.

Usage
-----
    python experiments/scalar_failure_demo.py
"""

from __future__ import annotations

import pandas as pd
from peas import PEASScorer
from peas.evidence import FaithfulnessEvidence, StabilityEvidence


def scalar_score(mu: float, lam: float) -> float:
    """Single-score aggregation s = (1 + certainty) / 2 = (1 + mu - lambda) / 2."""
    return (1 + mu - lam) / 2


def run_demo():
    scorer = PEASScorer(
        mu_source=FaithfulnessEvidence(),
        lambda_source=StabilityEvidence(),
    )

    cases = [
        # Block 1: same certainty c ≈ 0.42
        {"label": "Block 1 — Reliable",      "mu": 0.71, "lam": 0.29},
        {"label": "Block 1 — Contradictory", "mu": 0.91, "lam": 0.49},
        # Block 2: same certainty c ≈ 0.54
        {"label": "Block 2 — Reliable",      "mu": 0.77, "lam": 0.23},
        {"label": "Block 2 — Contradictory", "mu": 0.94, "lam": 0.40},
    ]

    rows = []
    for c in cases:
        result = scorer.audit_precomputed(mu=c["mu"], lam=c["lam"])
        s = scalar_score(c["mu"], c["lam"])
        rows.append({
            "Configuration": c["label"],
            "mu":    f"{result.mu:.3f}",
            "lambda": f"{result.lam:.3f}",
            "certainty c": f"{result.certainty:+.3f}",
            "single score s": f"{s:.3f}",
            "PEAS state": result.diagnostic.upper(),
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*75)
    print("  Scalar Aggregation Failure Demo")
    print("="*75)
    print(df.to_string(index=False))
    print()
    print("Observation: within each block, both configurations share the same")
    print("single score s (and certainty c), yet PEAS assigns different states.")
    print("A score-based ranking would prefer the CONTRADICTORY configuration")
    print("(higher s) over the RELIABLE one in both blocks.")


if __name__ == "__main__":
    run_demo()
