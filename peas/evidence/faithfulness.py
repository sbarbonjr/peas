"""
Perturbation-based faithfulness evidence source (favorable evidence, mu).

Implements the deletion / insertion / completeness triad from:
  Samek et al. (2017) — Evaluating the Visualization of What a DNN Learns
  Hooker et al. (2019) — A Benchmark for Interpretability Methods in DNNs

The three raw scores are converted to percentile ranks against randomly
selected feature subsets, so the final score is calibrated relative to
what a uniformly random explanation would achieve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from peas.evidence.base import EvidenceSource


class FaithfulnessEvidence(EvidenceSource):
    """
    Favorable evidence mu derived from deletion, insertion, and completeness.

    Parameters
    ----------
    random_repetitions : int
        Number of random feature subsets used for percentile calibration.
    weights : dict[str, float] | None
        Relative weights for {'F_deletion', 'F_insertion', 'C_topk'}.
        Defaults to equal weighting.
    """

    def __init__(
        self,
        random_repetitions: int = 20,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(name="faithfulness")
        self.random_repetitions = random_repetitions
        self.weights = weights or {"F_deletion": 1.0, "F_insertion": 1.0, "C_topk": 1.0}

    def score(self, model, X_train, X_eval, y_eval, ranking, k, seed=0):
        baseline = X_train.median(numeric_only=True)
        feature_names = list(X_eval.columns)
        top = ranking.sort_values("rank").head(k)["feature"].tolist()
        original_pred = model.predict(X_eval)
        base_acc = balanced_accuracy_score(y_eval, original_pred)

        def _deletion_drop(features):
            X_mod = X_eval.copy()
            for f in features:
                X_mod[f] = baseline[f]
            return max(0.0, base_acc - balanced_accuracy_score(y_eval, model.predict(X_mod)))

        def _insertion_completeness(features):
            X_base = pd.DataFrame(
                np.tile(baseline.to_numpy(), (len(X_eval), 1)),
                columns=feature_names, index=X_eval.index,
            )
            for f in features:
                X_base[f] = X_eval[f]
            ins_pred = model.predict(X_base)
            if hasattr(model, "predict_proba"):
                orig_conf = model.predict_proba(X_eval)[np.arange(len(X_eval)), original_pred]
                ins_conf  = model.predict_proba(X_base)[np.arange(len(X_eval)), original_pred]
                insertion = self._clip(float(np.mean(ins_conf / (orig_conf + 1e-9))))
            else:
                insertion = float(np.mean(ins_pred == original_pred))
            completeness = float(np.mean(ins_pred == original_pred))
            return insertion, completeness

        rng = np.random.default_rng(seed)
        random_sets = [
            rng.choice(feature_names, size=min(k, len(feature_names)), replace=False).tolist()
            for _ in range(self.random_repetitions)
        ]

        def _percentile(val, randoms):
            a = np.asarray(randoms)
            return self._clip((np.sum(a < val) + 0.5 * np.sum(a == val)) / len(a))

        method_drop = _deletion_drop(top)
        method_ins, method_comp = _insertion_completeness(top)

        rand_drops = [_deletion_drop(s) for s in random_sets]
        rand_ins_comp = [_insertion_completeness(s) for s in random_sets]
        rand_ins   = [x[0] for x in rand_ins_comp]
        rand_comps = [x[1] for x in rand_ins_comp]

        scores = {
            "F_deletion": _percentile(method_drop, rand_drops),
            "F_insertion": _percentile(method_ins,  rand_ins),
            "C_topk":      _percentile(method_comp, rand_comps),
        }
        total_w = sum(self.weights.values())
        return self._clip(sum(scores[k_] * v for k_, v in self.weights.items()) / total_w)
