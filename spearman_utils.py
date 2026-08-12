"""Statistical summaries and CSV exports for loss-expressibility studies.

The primary statistic is Spearman's rank correlation, which tests whether loss
and expressibility KL divergence have a monotonic relationship without assuming
linearity or matching numeric scales.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from scipy.stats import spearmanr


class SpearmanSummary(TypedDict):
    """Summary returned for one exported loss-KL relationship."""

    correlation: float
    p_value: float
    n_samples: int
    csv_path: str


EXPRESSIBILITY_METRICS = {
    "ideal_kl": "loss_kl_ideal.csv",
    "hilbert_schmidt_kl": "loss_kl_hilbert_schmidt.csv",
    "uhlmann_kl": "loss_kl_uhlmann.csv",
}


def extract_metric(
    results: Sequence[Mapping[str, object]],
    key: str,
) -> np.ndarray:
    """Extract one finite numeric metric from every result record."""
    if not results:
        raise ValueError("results cannot be empty")

    values = []
    for index, result in enumerate(results):
        if key not in result:
            raise KeyError(f"Result {index} is missing metric {key!r}")
        try:
            value = float(result[key])
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Result {index} metric {key!r} must be numeric"
            ) from error
        if not np.isfinite(value):
            raise ValueError(
                f"Result {index} metric {key!r} must be finite"
            )
        values.append(value)
    return np.asarray(values, dtype=float)


def compute_spearman_correlation(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Return Spearman correlation and its two-sided p-value."""
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if first_values.ndim != 1 or second_values.ndim != 1:
        raise ValueError("Spearman inputs must be one-dimensional")
    if first_values.size != second_values.size:
        raise ValueError("Spearman inputs must have equal lengths")
    if first_values.size < 3:
        raise ValueError("At least three observations are required")
    if not np.isfinite(first_values).all() or not np.isfinite(
        second_values
    ).all():
        raise ValueError("Spearman inputs must contain only finite values")
    if np.ptp(first_values) == 0 or np.ptp(second_values) == 0:
        raise ValueError("Spearman correlation requires non-constant inputs")

    statistic = spearmanr(first_values, second_values)
    return float(statistic.statistic), float(statistic.pvalue)


def export_loss_expressibility_csv(
    results: Sequence[Mapping[str, object]],
    metric_key: str,
    csv_path: str | Path,
) -> SpearmanSummary:
    """Export one loss-KL dataset and return its rank correlation summary."""
    losses = extract_metric(results, "loss")
    divergences = extract_metric(results, metric_key)
    correlation, p_value = compute_spearman_correlation(losses, divergences)

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["loss", metric_key])
        writer.writerows(zip(losses, divergences, strict=True))

    return {
        "correlation": correlation,
        "p_value": p_value,
        "n_samples": int(losses.size),
        "csv_path": str(output_path),
    }


def export_all_loss_expressibility_csvs(
    results: Sequence[Mapping[str, object]],
    output_dir: str | Path = "files",
) -> dict[str, SpearmanSummary]:
    """Export loss relationships for all supported expressibility metrics."""
    output_directory = Path(output_dir)
    return {
        metric_key: export_loss_expressibility_csv(
            results,
            metric_key,
            output_directory / filename,
        )
        for metric_key, filename in EXPRESSIBILITY_METRICS.items()
    }
