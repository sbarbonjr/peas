"""
Diagnostic plane visualisation utilities.
"""

from __future__ import annotations

import numpy as np


def plot_diagnostic_plane(
    results: list[dict],
    theta: float = 0.5,
    ax=None,
    label_key: str = "method",
    title: str = "PEAS Diagnostic Plane",
):
    """
    Plot (mu, lambda) points on the PAL2v diagnostic plane.

    Parameters
    ----------
    results : list of dict
        Each dict must have keys 'mu', 'lambda' (or 'lam'), and optionally
        the key specified in label_key.
    theta : float
        Threshold for the quadrant boundary lines.
    ax : matplotlib.axes.Axes | None
        Existing axes to plot onto. Creates a new figure if None.
    label_key : str
        Key in each result dict to use as the point label.
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    colors = {
        "reliable":      "#d4edda",
        "contradictory": "#fff3cd",
        "rejected":      "#f8d7da",
        "inconclusive":  "#d1ecf1",
    }
    regions = [
        (0, theta, theta, 1,    "reliable"),
        (theta, 1, theta, 1,    "contradictory"),
        (theta, 1, 0, theta,    "rejected"),
        (0, theta, 0, theta,    "inconclusive"),
    ]
    for lam0, lam1, mu0, mu1, state in regions:
        ax.fill_between([lam0, lam1], mu0, mu1, color=colors[state], alpha=0.6, zorder=0)

    ax.axvline(theta, color="gray", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(theta, color="gray", linestyle="--", linewidth=1.2, zorder=1)

    point_colors = {
        "reliable": "#28a745", "contradictory": "#fd7e14",
        "rejected": "#dc3545", "inconclusive":  "#17a2b8",
    }
    for r in results:
        mu  = r["mu"]
        lam = r.get("lambda", r.get("lam", 0.0))
        state = r.get("diagnostic", _state(mu, lam, theta))
        label = r.get(label_key, "")
        ax.scatter(lam, mu, color=point_colors[state], s=80, zorder=3, edgecolors="white", linewidths=0.5)
        if label:
            ax.annotate(label, (lam, mu), textcoords="offset points", xytext=(6, 3), fontsize=8)

    for state, color in point_colors.items():
        ax.scatter([], [], color=color, label=state.capitalize(), s=60)
    ax.legend(fontsize=8, loc="lower right")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\lambda$ (contrary evidence)", fontsize=11)
    ax.set_ylabel(r"$\mu$ (favorable evidence)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    return ax


def _state(mu, lam, theta):
    if mu >= theta and lam < theta:  return "reliable"
    if mu >= theta and lam >= theta: return "contradictory"
    if mu < theta  and lam >= theta: return "rejected"
    return "inconclusive"
