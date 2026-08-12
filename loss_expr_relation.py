"""Study loss versus quantum-circuit expressibility on four-class MNIST.

The script samples subcircuits from a pretrained supercircuit, evaluates their
classification loss, estimates ideal and noisy expressibility, and exports the
relationships as CSV files with Spearman rank-correlation summaries.

Use validation data for exploratory analysis (the default). Reserve the test set
for a final, pre-specified analysis::

    python loss_expr_relation.py --population-size 100

Run the lightweight integration check first::

    python loss_expr_relation.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import random
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from qiskit_ibm_runtime.fake_provider import FakeYorktownV2
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from expr_search_mnist import (
    SuperQFCModel,
    build_dataflow,
    build_model_for_gene,
    configure_runtime,
    evaluate_gene_model,
)
from quantum_expressibility import (
    estimate_ideal_expressibility,
    estimate_noisy_expressibility,
)
from quantumnas_style_search import load_supercircuit_checkpoint
from spearman_utils import export_all_loss_expressibility_csvs


Gene = list[int]

DEFAULT_CONFIG = "configs_mnist.yaml"
DEFAULT_CHECKPOINT = "max-acc-valid.pt"
DEFAULT_OUTPUT_DIR = "files"
DEFAULT_POPULATION_SIZE = 100
DEFAULT_N_PAIRS = 200
DEFAULT_N_BINS = 75
SMOKE_TEST_SHOTS = 32


class ExperimentResult(TypedDict):
    """Metrics collected for one sampled quantum architecture."""

    gene: Gene
    accuracy: float
    loss: float
    ideal_kl: float
    hilbert_schmidt_kl: float
    uhlmann_kl: float


def parse_args() -> argparse.Namespace:
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
    )
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="valid",
        help="Dataset split used to measure classification loss.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum classification samples per architecture.",
    )
    parser.add_argument("--expressibility-pairs", type=int, default=DEFAULT_N_PAIRS)
    parser.add_argument("--expressibility-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument(
        "--ideal-classification",
        action="store_true",
        help="Evaluate classification without backend noise.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic-data integration check.",
    )
    return parser.parse_args()


def sample_unique_population(
    gene_choices: Sequence[Sequence[int]],
    population_size: int,
    seed: int,
) -> list[Gene]:
    """Sample a reproducible population without duplicate architectures."""
    if population_size < 1:
        raise ValueError("population_size must be at least 1")
    if not gene_choices or any(not choices for choices in gene_choices):
        raise ValueError("Every gene position must have at least one choice")

    search_space_size = int(np.prod([len(choices) for choices in gene_choices]))
    if population_size > search_space_size:
        raise ValueError(
            f"population_size={population_size} exceeds the search space size "
            f"of {search_space_size}"
        )

    rng = random.Random(seed)
    population: list[Gene] = []
    observed_genes: set[tuple[int, ...]] = set()
    while len(population) < population_size:
        gene = [rng.choice(choices) for choices in gene_choices]
        key = tuple(gene)
        if key not in observed_genes:
            observed_genes.add(key)
            population.append(gene)
    return population


def evaluate_population(
    cfg: Any,
    model: SuperQFCModel,
    population: Sequence[Gene],
    dataflow: dict[str, Any],
    backend: Any,
    device: torch.device,
    *,
    split: str = "valid",
    use_qiskit: bool = True,
    max_samples: int | None = None,
    n_pairs: int = DEFAULT_N_PAIRS,
    n_bins: int = DEFAULT_N_BINS,
    seed: int = 42,
) -> list[ExperimentResult]:
    """Collect classification and expressibility metrics for each gene."""
    results = []
    for gene in population:
        accuracy, loss = evaluate_gene_model(
            cfg,
            model,
            gene,
            dataflow,
            split=split,
            device=device,
            use_qiskit=use_qiskit,
            max_samples=max_samples,
        )
        ideal_kl = estimate_ideal_expressibility(
            gene,
            n_qubits=model.n_wires,
            n_pairs=n_pairs,
            n_bins=n_bins,
            seed=seed,
        )
        hilbert_schmidt_kl, uhlmann_kl = estimate_noisy_expressibility(
            architecture=gene,
            n_qubits=model.n_wires,
            backend=backend,
            n_pairs=n_pairs,
            n_bins=n_bins,
            seed=seed,
        )
        results.append(
            {
                "gene": gene.copy(),
                "accuracy": accuracy,
                "loss": loss,
                "ideal_kl": ideal_kl,
                "hilbert_schmidt_kl": hilbert_schmidt_kl,
                "uhlmann_kl": uhlmann_kl,
            }
        )
    return results


def log_summaries(summaries: Mapping[str, Mapping[str, object]]) -> None:
    """Log correlation summaries in a compact, readable format."""
    for metric_name, summary in summaries.items():
        logger.info(
            f"{metric_name}: Spearman correlation="
            f"{float(summary['correlation']):.6f}, "
            f"p-value={float(summary['p_value']):.3e}, "
            f"n={int(summary['n_samples'])}, "
            f"CSV={summary['csv_path']}"
        )


def _synthetic_dataflow(cfg: Any, device: torch.device) -> dict[str, Any]:
    """Create small in-memory batches for the smoke test."""
    inputs = torch.randn(3, 1, 28, 28, device=device)
    targets = torch.tensor([0, 1, 2], device=device)
    batch = {
        cfg.dataset.input_name: inputs,
        cfg.dataset.target_name: targets,
    }
    return {"train": [batch], "valid": [batch], "test": [batch]}


def run_smoke_test(
    cfg: Any,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    """Exercise checkpoint, classification, expressibility, and CSV exports."""
    backend = FakeYorktownV2()
    architecture_probe = SuperQFCModel(copy.deepcopy(cfg.model.arch))
    gene_choices = architecture_probe.arch_space
    population = sample_unique_population(
        gene_choices,
        population_size=3,
        seed=int(cfg.debug.seed),
    )
    model = build_model_for_gene(
        cfg,
        population[0],
        device,
        noisy_backend=backend,
        n_shots=SMOKE_TEST_SHOTS,
    )
    load_supercircuit_checkpoint(model, checkpoint_path)
    results = evaluate_population(
        cfg,
        model,
        population,
        _synthetic_dataflow(cfg, device),
        backend,
        device,
        split="valid",
        use_qiskit=True,
        max_samples=3,
        n_pairs=2,
        n_bins=8,
        seed=int(cfg.debug.seed),
    )

    # Three quantum results can legitimately tie on a tiny, shot-based sample.
    # Use deterministic non-constant values to test the statistical exporter.
    export_probe = [dict(result) for result in results]
    for index, result in enumerate(export_probe):
        result["loss"] = float(index + 1)
        result["ideal_kl"] = float(3 - index)
        result["hilbert_schmidt_kl"] = float(index + 2)
        result["uhlmann_kl"] = float(2 * index + 1)

    smoke_output_dir = Path(".codex_review") / "loss_expressibility"
    try:
        summaries = export_all_loss_expressibility_csvs(
            export_probe,
            smoke_output_dir,
        )
        if len(summaries) != 3:
            raise RuntimeError("Smoke test did not export all metric summaries")
    finally:
        for filename in (
            "loss_kl_ideal.csv",
            "loss_kl_hilbert_schmidt.csv",
            "loss_kl_uhlmann.csv",
        ):
            (smoke_output_dir / filename).unlink(missing_ok=True)
        if smoke_output_dir.exists():
            smoke_output_dir.rmdir()

    if len(results) != 3 or not all(
        np.isfinite(result["loss"]) for result in results
    ):
        raise RuntimeError("Smoke test produced invalid quantum results")
    logger.info("Loss-expressibility smoke test passed.")


def run_experiment(
    cfg: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Run the full loss-expressibility analysis and export its results."""
    run_name = f"loss_expr_relation_{datetime.now():%Y%m%d_%H%M%S}"
    set_run_dir(str(Path("runs") / run_name))
    backend = FakeYorktownV2()
    dataflow = build_dataflow(cfg, device)
    architecture_probe = SuperQFCModel(copy.deepcopy(cfg.model.arch))
    population = sample_unique_population(
        architecture_probe.arch_space,
        args.population_size,
        seed=int(cfg.debug.seed),
    )
    model = build_model_for_gene(
        cfg,
        population[0],
        device,
        noisy_backend=None if args.ideal_classification else backend,
    )
    load_supercircuit_checkpoint(model, Path(args.checkpoint).resolve())

    results = evaluate_population(
        cfg,
        model,
        population,
        dataflow,
        backend,
        device,
        split=args.split,
        use_qiskit=not args.ideal_classification,
        max_samples=args.max_samples,
        n_pairs=args.expressibility_pairs,
        n_bins=args.expressibility_bins,
        seed=int(cfg.debug.seed),
    )
    summaries = export_all_loss_expressibility_csvs(
        results,
        args.output_dir,
    )
    log_summaries(summaries)


def main() -> None:
    """Load configuration and run the smoke test or full analysis."""
    args = parse_args()
    if args.population_size < 1:
        raise ValueError("--population-size must be at least 1")
    if args.expressibility_pairs < 1:
        raise ValueError("--expressibility-pairs must be at least 1")
    if args.expressibility_bins < 2:
        raise ValueError("--expressibility-bins must be at least 2")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")

    configs.load(args.config)
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = float(configs.optimizer.lr)
    device = configure_runtime(configs)
    checkpoint_path = Path(args.checkpoint).resolve()

    if args.smoke_test:
        run_smoke_test(configs, device, checkpoint_path)
    else:
        run_experiment(configs, device, args)


if __name__ == "__main__":
    main()
