"""
Threshold Sensitivity Analysis.

Shows how the choice of theta affects diagnostic state assignments
across a grid of (mu, lambda) pairs, and demonstrates the three
ThresholdPolicy options: Fixed, Asymmetric, Calibrated.

Run as a script or open as a Jupyter notebook (jupytext compatible).
"""

# %% [markdown]
# # Threshold Sensitivity
#
# The threshold theta is not fixed by PAL2v theory — it is a hyperparameter.
# This notebook shows its effect and how to calibrate it.

# %% Imports
import numpy as np
import pandas as pd
from peas.logic import PAL2vDiagnostic, FixedThreshold, AsymmetricThreshold, CalibratedThreshold

# %% 1. Fixed theta grid sweep
print("=== Sensitivity to theta (Fixed) ===")
test_pairs = [
    ("reliable_clear",      0.85, 0.15),
    ("reliable_border",     0.55, 0.45),
    ("contradictory_clear", 0.85, 0.75),
    ("rejected_clear",      0.25, 0.80),
    ("inconclusive_clear",  0.25, 0.20),
]

rows = []
for label, mu, lam in test_pairs:
    for theta in [0.3, 0.4, 0.5, 0.6, 0.7]:
        diag = PAL2vDiagnostic(FixedThreshold(theta))
        result = diag.evaluate(mu, lam)
        rows.append({"pair": label, "mu": mu, "lambda": lam,
                     "theta": theta, "state": result.diagnostic})

df = pd.DataFrame(rows)
print(df.pivot_table(index="pair", columns="theta", values="state", aggfunc="first").to_string())

# %% 2. Asymmetric theta
print("\n=== Asymmetric threshold (theta_mu=0.65, theta_lam=0.40) ===")
diag_asym = PAL2vDiagnostic(AsymmetricThreshold(theta_mu=0.65, theta_lam=0.40))
for label, mu, lam in test_pairs:
    r = diag_asym.evaluate(mu, lam)
    print(f"  {label:25s}  mu={mu}  lam={lam}  →  {r.diagnostic}")

# %% 3. Calibrated threshold
print("\n=== Calibrated threshold (median of observed scores) ===")
# Simulate a batch of audit results (e.g. from running the benchmark)
rng = np.random.default_rng(42)
observed_mus  = list(rng.beta(8, 2, size=50))   # right-skewed (methods tend to be faithful)
observed_lams = list(rng.beta(2, 5, size=50))   # left-skewed (methods tend to be stable)

threshold = CalibratedThreshold()
threshold.fit(observed_mus, observed_lams)
print(f"  Calibrated: {threshold}")

diag_cal = PAL2vDiagnostic(threshold)
for label, mu, lam in test_pairs:
    r = diag_cal.evaluate(mu, lam)
    print(f"  {label:25s}  mu={mu}  lam={lam}  →  {r.diagnostic}")

# %% 4. Visualise state regions for different thetas
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    configs = [
        ("Fixed theta=0.5", FixedThreshold(0.5)),
        ("Asymmetric mu=0.65 lam=0.40", AsymmetricThreshold(0.65, 0.40)),
        ("Calibrated (data-driven)", threshold),
    ]
    colors = {"reliable": "#d4edda", "contradictory": "#fff3cd",
              "rejected": "#f8d7da", "inconclusive": "#d1ecf1"}
    state_colors = {"reliable": "#28a745", "contradictory": "#fd7e14",
                    "rejected": "#dc3545", "inconclusive": "#17a2b8"}

    grid = np.linspace(0, 1, 80)
    MU, LAM = np.meshgrid(grid, grid)

    for ax, (title, policy) in zip(axes, configs):
        diag = PAL2vDiagnostic(policy)
        color_map = np.zeros((*MU.shape, 4))
        state_to_rgba = {
            "reliable":      [0.83, 0.93, 0.85, 0.7],
            "contradictory": [1.00, 0.95, 0.80, 0.7],
            "rejected":      [0.97, 0.84, 0.84, 0.7],
            "inconclusive":  [0.82, 0.93, 0.95, 0.7],
        }
        for i in range(MU.shape[0]):
            for j in range(MU.shape[1]):
                state = diag.evaluate(float(MU[i, j]), float(LAM[i, j])).diagnostic
                color_map[i, j] = state_to_rgba[state]
        ax.imshow(color_map, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mu$")
        ax.set_title(title, fontsize=9)
        for label, mu, lam in test_pairs:
            r = diag.evaluate(mu, lam)
            ax.scatter(lam, mu, color=state_colors[r.diagnostic], s=60,
                       edgecolors="white", linewidths=0.5, zorder=5)

    plt.tight_layout()
    plt.savefig("threshold_sensitivity.png", dpi=150)
    print("\nPlot saved to threshold_sensitivity.png")
    plt.show()
except ImportError:
    print("\nInstall matplotlib for visualisation: pip install matplotlib")
