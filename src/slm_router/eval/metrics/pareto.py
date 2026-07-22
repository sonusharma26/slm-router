"""Pareto frontier analysis for cost-quality trade-offs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CQPoint:
    """A cost-quality operating point for a model or routing policy.

    Attributes:
        label: Human-readable name (e.g. model ID or policy name).
        cost: Cost per query (lower is better, e.g. USD).
        quality: Quality score in [0, 1] (higher is better).
    """
    label: str
    cost: float
    quality: float


def pareto_frontier(points: list[CQPoint]) -> list[CQPoint]:
    """Compute the Pareto frontier: minimise cost, maximise quality.

    A point P dominates Q if P.cost <= Q.cost AND P.quality >= Q.quality
    (with at least one strict inequality).

    Args:
        points: List of CQPoint objects to analyse.

    Returns:
        Subset of *points* that are not dominated by any other point,
        sorted by cost ascending.
    """
    if not points:
        return []

    # Sort by cost ascending, then quality descending as tie-breaker
    sorted_pts = sorted(points, key=lambda p: (p.cost, -p.quality))

    frontier: list[CQPoint] = []
    best_quality = float("-inf")

    for pt in sorted_pts:
        if pt.quality > best_quality:
            frontier.append(pt)
            best_quality = pt.quality

    return frontier


def plot_pareto(
    points: list[CQPoint],
    frontier: list[CQPoint],
    out_png: str,
) -> None:
    """Scatter-plot all cost-quality points and highlight the Pareto frontier.

    Uses the Agg matplotlib backend (no display required).

    Args:
        points: All operating points to plot.
        frontier: Pareto-optimal subset (highlighted in red).
        out_png: Output path for the PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend; safe on servers/Windows
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_pareto. "
            "Install it with: pip install matplotlib"
        ) from exc

    import os

    # Ensure output directory exists
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Build sets for quick lookup
    frontier_labels = {p.label for p in frontier}

    # Plot all points in blue/grey
    non_frontier = [p for p in points if p.label not in frontier_labels]
    if non_frontier:
        ax.scatter(
            [p.cost for p in non_frontier],
            [p.quality for p in non_frontier],
            color="steelblue",
            alpha=0.7,
            label="Dominated",
            zorder=3,
        )
        for p in non_frontier:
            ax.annotate(
                p.label,
                (p.cost, p.quality),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color="steelblue",
            )

    # Plot frontier points in red
    if frontier:
        ax.scatter(
            [p.cost for p in frontier],
            [p.quality for p in frontier],
            color="crimson",
            zorder=4,
            label="Pareto frontier",
        )
        # Draw connecting line along frontier
        sorted_frontier = sorted(frontier, key=lambda p: p.cost)
        ax.plot(
            [p.cost for p in sorted_frontier],
            [p.quality for p in sorted_frontier],
            color="crimson",
            linewidth=1.5,
            linestyle="--",
            zorder=3,
        )
        for p in frontier:
            ax.annotate(
                p.label,
                (p.cost, p.quality),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color="crimson",
                fontweight="bold",
            )

    ax.set_xlabel("Cost (USD / query)")
    ax.set_ylabel("Quality score")
    ax.set_title("Cost–Quality Pareto Frontier")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
