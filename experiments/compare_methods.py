"""
Multi-method PEAS comparison across datasets and models.

Reproduces the method comparison table from the XKDD 2026 paper
(Table 2 — PEAS results per method).

Usage
-----
    python experiments/compare_methods.py
    python experiments/compare_methods.py --latex
    python experiments/compare_methods.py --run-dir artifacts/runs/full_v1
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               ExtraTreesClassifier)
from sklearn.model_selection import train_test_split

from peas import PEASScorer
from peas.evidence import FaithfulnessEvidence
from peas.evidence.stability import StabilityEvidence, ranking_stability
from peas.logic import FixedThreshold, PAL2vDiagnostic


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

MODELS = {
    "random_forest":     lambda s: RandomForestClassifier(n_estimators=100, random_state=s),
    "gradient_boosting": lambda s: GradientBoostingClassifier(n_estimators=100, random_state=s),
    "extra_trees":       lambda s: ExtraTreesClassifier(n_estimators=100, random_state=s),
}

SEEDS = [23, 42, 77]
K = 5
N_STABILITY_RUNS = 10


def permutation_ranking(model, X_eval, y_eval, seed) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    res = permutation_importance(model, X_eval, y_eval, n_repeats=8, random_state=seed)
    df = pd.DataFrame({"feature": X_eval.columns, "importance_score": res.importances_mean})
    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def random_ranking(X_eval, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = list(X_eval.columns)
    rng.shuffle(features)
    df = pd.DataFrame({"feature": features, "importance_score": np.random.rand(len(features))})
    df["rank"] = range(1, len(df) + 1)
    return df


def run_comparison(latex: bool = False):
    records = []
    scorer = PEASScorer(
        mu_source=FaithfulnessEvidence(random_repetitions=20),
        lambda_source=StabilityEvidence(),
        threshold=FixedThreshold(0.5),
        k=K,
    )

    for ds_name, loader in DATASETS.items():
        bunch = loader()
        X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        y = pd.Series(bunch.target)

        for model_name, model_fn in MODELS.items():
            for seed in SEEDS:
                X_tr, X_ev, y_tr, y_ev = train_test_split(
                    X, y, test_size=0.3, random_state=seed, stratify=y)

                model = model_fn(seed)
                model.fit(X_tr, y_tr)

                for xai_name, ranking_fn in [
                    ("permutation", lambda m, s: permutation_ranking(m, X_ev, y_ev, s)),
                    ("random_ranking", lambda m, s: random_ranking(X_ev, s)),
                ]:
                    rankings = [ranking_fn(model_fn(seed + i).fit(X_tr, y_tr), seed + i)
                                for i in range(N_STABILITY_RUNS)]
                    stab = ranking_stability(rankings)
                    rand_sim = 1.0 - stab

                    lam = StabilityEvidence.from_precomputed(stab, rand_sim)
                    mu  = scorer.mu_source.score(model, X_tr, X_ev, y_ev, rankings[0], K, seed)
                    result = scorer.audit_precomputed(mu=mu, lam=lam)

                    records.append({
                        "dataset": ds_name, "model": model_name,
                        "method": xai_name, "seed": seed,
                        "mu": result.mu, "lambda": result.lam,
                        "certainty": result.certainty,
                        "contradiction": result.contradiction,
                        "state": result.diagnostic,
                    })
                    print(f"  {ds_name:20s} {model_name:20s} {xai_name:18s} "
                          f"seed={seed}  {result}")

    df = pd.DataFrame(records)
    summary = (df.groupby("method")[["mu", "lambda", "certainty", "contradiction"]]
               .mean().round(4).sort_values("certainty", ascending=False))
    print("\n=== Method summary (mean over datasets, models, seeds) ===")
    print(summary.to_string())

    if latex:
        print("\n=== LaTeX table ===")
        print(summary.to_latex(float_format="%.4f"))

    out = Path("artifacts") / "comparison_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="PEAS multi-method comparison")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--run-dir", default=None,
                        help="Load pre-computed results from a run directory instead of re-running")
    args = parser.parse_args()
    run_comparison(latex=args.latex)


if __name__ == "__main__":
    main()
