"""
Minimal end-to-end PEAS audit on a single dataset and XAI method.

Usage
-----
    python experiments/audit_single_model.py
    python experiments/audit_single_model.py --dataset breast_cancer --xai shap --k 5
"""

from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from peas import PEASScorer
from peas.evidence import FaithfulnessEvidence
from peas.evidence.stability import StabilityEvidence, ranking_stability
from peas.logic import FixedThreshold


DATASETS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
}


def load_dataset(name: str):
    bunch = DATASETS[name]()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target)
    return X, y


def get_ranking_shap(model, X_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required: pip install shap")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list):
        importance = np.abs(np.array(shap_values)).mean(axis=(0, 1))
    else:
        importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": X_train.columns, "importance_score": importance})
    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def get_ranking_permutation(model, X_eval, y_eval, seed: int) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_eval, y_eval, n_repeats=10, random_state=seed)
    df = pd.DataFrame({
        "feature": X_eval.columns,
        "importance_score": result.importances_mean,
    }).sort_values("importance_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def run_audit(dataset: str, xai: str, k: int, n_runs: int, seed: int):
    print(f"\n{'='*60}")
    print(f"  PEAS Audit  |  dataset={dataset}  xai={xai}  k={k}")
    print(f"{'='*60}")

    X, y = load_dataset(dataset)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    # Train model and n_runs re-initialisations for stability
    rankings = []
    for run_seed in range(seed, seed + n_runs):
        model = RandomForestClassifier(n_estimators=100, random_state=run_seed)
        model.fit(X_train, y_train)
        if xai == "shap":
            r = get_ranking_shap(model, X_train, run_seed)
        else:
            r = get_ranking_permutation(model, X_eval, y_eval, run_seed)
        rankings.append(r)

    # Use the first run as the primary ranking and model
    main_model = RandomForestClassifier(n_estimators=100, random_state=seed).fit(X_train, y_train)
    main_ranking = rankings[0]

    # Compute stability from the n_runs rankings
    stability = ranking_stability(rankings)
    random_similarity = 1.0 - stability  # simplified proxy; full pipeline uses BaselineGain

    lam = StabilityEvidence.from_precomputed(stability, random_similarity)

    scorer = PEASScorer(
        mu_source=FaithfulnessEvidence(random_repetitions=20),
        lambda_source=FaithfulnessEvidence(),   # placeholder; lambda injected below
        threshold=FixedThreshold(0.5),
        k=k,
    )

    # Compute mu directly
    mu = scorer.mu_source.score(
        main_model, X_train, X_eval, y_eval, main_ranking, k, seed)

    result = scorer.audit_precomputed(mu=mu, lam=lam)

    print(f"\n  mu  (faithfulness) : {result.mu:.4f}")
    print(f"  lam (instability)  : {result.lam:.4f}")
    print(f"  certainty          : {result.certainty:+.4f}")
    print(f"  contradiction      : {result.contradiction:+.4f}")
    print(f"\n  >> {result}")
    print()


def main():
    parser = argparse.ArgumentParser(description="PEAS single-model audit")
    parser.add_argument("--dataset", default="breast_cancer", choices=list(DATASETS))
    parser.add_argument("--xai", default="permutation", choices=["shap", "permutation"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Model re-initialisations for stability estimation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_audit(args.dataset, args.xai, args.k, args.n_runs, args.seed)


if __name__ == "__main__":
    main()
