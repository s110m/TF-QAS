"""QuantumNAS-style architecture search for a four-class MNIST QNN.

The search follows a weight-sharing workflow:

1. Load a pretrained supercircuit checkpoint.
2. Search candidate subcircuits on a small noisy validation subset.
3. Retrain each selected subcircuit on an ideal simulator.
4. Select weights on clean validation accuracy.
5. Report noisy test accuracy once for each target backend.

Run the lightweight integration check before starting the full experiment::

    python quantumnas_style_search.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchquantum.algorithm.quantumnas.super_layers
import torchquantum.device
import torchquantum.measurement
import torchquantum.operator
from qiskit_ibm_runtime.fake_provider import (
    FakeManilaV2,
    FakeTorino,
    FakeYorktownV2,
)
from torchpack.environ import set_run_dir
from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from expr_search_mnist import (
    SuperQFCModel,
    build_dataflow,
    build_model_for_gene,
    configure_runtime,
    evaluate_gene_model,
    portable_state_dict,
    train_gene,
)


Gene = list[int]
StateDict = dict[str, torch.Tensor]

DEFAULT_CHECKPOINT = "max-acc-valid.pt"
DEFAULT_N_EXPERIMENTS = 5
SMOKE_TEST_SHOTS = 32


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_mnist.yaml")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Trusted pretrained supercircuit checkpoint.",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=DEFAULT_N_EXPERIMENTS,
        help="Independent evolutionary searches per backend.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic-data search without loading MNIST.",
    )
    return parser.parse_args()


def _register_legacy_checkpoint_aliases() -> None:
    """Register names required by checkpoints created by the original script."""
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "SuperQFCModel0"):
        main_module.SuperQFCModel0 = SuperQFCModel

    legacy_modules = {
        "torchquantum.devices": torchquantum.device,
        "torchquantum.super_layers": (
            torchquantum.algorithm.quantumnas.super_layers
        ),
        "torchquantum.operators": torchquantum.operator,
        "torchquantum.measure": torchquantum.measurement,
    }
    for legacy_name, current_module in legacy_modules.items():
        sys.modules.setdefault(legacy_name, current_module)


def load_supercircuit_checkpoint(
    model: SuperQFCModel,
    checkpoint_path: Path,
) -> None:
    """Load a trusted legacy checkpoint into the current model definition."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Supercircuit checkpoint not found: {checkpoint_path}"
        )

    _register_legacy_checkpoint_aliases()
    checkpoint = io.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dictionary")
    if not isinstance(checkpoint["model"], Mapping):
        raise ValueError("Checkpoint 'model' entry must be a state dictionary")

    legacy_parameter_suffixes = (".U3_params", ".CU3_params")
    model_state = {
        name: value
        for name, value in checkpoint["model"].items()
        if "q_device." not in name
        and not name.endswith(legacy_parameter_suffixes)
    }
    incompatible = model.load_state_dict(model_state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing model parameters: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint contains unsupported parameters: "
            f"{incompatible.unexpected_keys}"
        )


class AccuracyEvolutionarySearcher:
    """Elitist evolutionary search using noisy validation accuracy."""

    def __init__(
        self,
        gene_choices: Sequence[Sequence[int]],
        cfg: Any,
        dataflow: dict[str, Any],
        device: torch.device,
        *,
        n_iterations: int | None = None,
        parent_size: int | None = None,
        mutation_size: int | None = None,
        crossover_size: int | None = None,
        mutation_probability: float | None = None,
        max_validation_samples: int | None = None,
    ) -> None:
        self.gene_choices = [list(choices) for choices in gene_choices]
        self.cfg = cfg
        self.dataflow = dataflow
        self.device = device
        self.n_iterations = int(
            cfg.es.n_iterations if n_iterations is None else n_iterations
        )
        self.parent_size = int(
            cfg.es.parent_size if parent_size is None else parent_size
        )
        self.mutation_size = int(
            cfg.es.mutation_size if mutation_size is None else mutation_size
        )
        self.crossover_size = int(
            cfg.es.crossover_size if crossover_size is None else crossover_size
        )
        self.mutation_probability = float(
            mutation_probability
            if mutation_probability is not None
            else cfg.es.mutation_prob
        )
        self.max_validation_samples = int(
            cfg.es.eval.n_test_samples
            if max_validation_samples is None
            else max_validation_samples
        )
        self.population: list[Gene] = []
        self.score_cache: dict[tuple[int, ...], float] = {}

        if not self.gene_choices or any(
            not choices for choices in self.gene_choices
        ):
            raise ValueError("Every gene position must have at least one choice")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if self.parent_size < 1:
            raise ValueError("parent_size must be at least 1")
        if self.mutation_size < 0 or self.crossover_size < 0:
            raise ValueError("mutation_size and crossover_size cannot be negative")
        if self.crossover_size and self.parent_size < 2:
            raise ValueError("At least two parents are required for crossover")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0 and 1")
        if self.max_validation_samples < 1:
            raise ValueError("max_validation_samples must be at least 1")

    def random_sample(self, sample_count: int) -> list[Gene]:
        """Sample genes independently from the discrete search space."""
        return [
            [random.choice(choices) for choices in self.gene_choices]
            for _ in range(sample_count)
        ]

    def mutate(self, gene: Sequence[int]) -> Gene:
        """Mutate each gene position independently."""
        return [
            random.choice(self.gene_choices[index])
            if random.random() < self.mutation_probability
            else value
            for index, value in enumerate(gene)
        ]

    def crossover(self, parents: Sequence[Sequence[int]]) -> Gene:
        """Create a child by uniformly mixing two parents."""
        return [
            parents[0][index] if random.random() < 0.5 else parents[1][index]
            for index in range(len(self.gene_choices))
        ]

    def score_gene(
        self,
        model: SuperQFCModel,
        gene: Sequence[int],
    ) -> float:
        """Evaluate and cache noisy validation accuracy for one gene."""
        key = tuple(gene)
        if key not in self.score_cache:
            accuracy, _ = evaluate_gene_model(
                self.cfg,
                model,
                gene,
                self.dataflow,
                split="valid",
                device=self.device,
                use_qiskit=True,
                max_samples=self.max_validation_samples,
            )
            self.score_cache[key] = accuracy
        return self.score_cache[key]

    def select_and_transform(
        self,
        scores: Sequence[float],
    ) -> tuple[Gene, float]:
        """Keep elite parents, then generate mutations and crossovers."""
        if len(scores) != len(self.population):
            raise ValueError("Every population member must have one score")
        if self.parent_size > len(self.population):
            raise ValueError("parent_size cannot exceed population size")

        parent_indices = np.argsort(-np.asarray(scores))[: self.parent_size]
        parents = [self.population[index] for index in parent_indices]
        best_gene = parents[0].copy()
        best_score = float(scores[parent_indices[0]])

        mutations = [
            self.mutate(random.choice(parents))
            for _ in range(self.mutation_size)
        ]
        crossovers = [
            self.crossover(random.sample(parents, 2))
            for _ in range(self.crossover_size)
        ]
        self.population = parents + mutations + crossovers
        return best_gene, best_score

    def search(
        self,
        model: SuperQFCModel,
        n_experiments: int,
    ) -> list[Gene]:
        """Run independent searches for a single noisy backend model."""
        if n_experiments < 1:
            raise ValueError("n_experiments must be at least 1")

        population_size = (
            self.parent_size + self.mutation_size + self.crossover_size
        )
        winners = []
        for experiment_index in range(n_experiments):
            self.population = self.random_sample(population_size)
            best_gene: Gene | None = None
            best_score = -1.0

            for _ in range(self.n_iterations):
                scores = [
                    self.score_gene(model, gene) for gene in self.population
                ]
                generation_gene, generation_score = self.select_and_transform(
                    scores
                )
                if generation_score > best_score:
                    best_gene = generation_gene
                    best_score = generation_score

            if best_gene is None:
                raise RuntimeError("Evolutionary search produced no solution")
            winners.append(best_gene.copy())
            logger.info(
                f"Experiment {experiment_index + 1}: best gene={best_gene}, "
                f"validation accuracy={best_score:.6f}"
            )
        return winners


def search_backends(
    cfg: Any,
    dataflow: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    noisy_backends: Sequence[Any],
    n_experiments: int,
) -> list[list[Gene]]:
    """Search the best genes independently for every noisy backend."""
    architecture_probe = SuperQFCModel(copy.deepcopy(cfg.model.arch))
    gene_choices = architecture_probe.arch_space
    initial_gene = [choices[0] for choices in gene_choices]
    winners_by_backend = []

    for backend in noisy_backends:
        model = build_model_for_gene(
            cfg,
            initial_gene,
            device,
            noisy_backend=backend,
        )
        load_supercircuit_checkpoint(model, checkpoint_path)
        searcher = AccuracyEvolutionarySearcher(
            gene_choices,
            cfg,
            dataflow,
            device,
        )
        winners = searcher.search(model, n_experiments)
        winners_by_backend.append(winners)
        logger.info(f"{backend.name} selected genes: {winners}")

    return winners_by_backend


def train_and_evaluate_genes(
    cfg: Any,
    dataflow: dict[str, Any],
    device: torch.device,
    noisy_backends: Sequence[Any],
    genes_by_backend: Sequence[Sequence[Gene]],
) -> list[np.ndarray]:
    """Retrain selected genes once and report noisy test accuracy."""
    trained_states: dict[tuple[int, ...], StateDict] = {}
    results = []

    for backend, genes in zip(noisy_backends, genes_by_backend, strict=True):
        backend_accuracies = []
        for gene in genes:
            key = tuple(gene)
            if key not in trained_states:
                trained_states[key] = train_gene(cfg, gene, dataflow, device)

            model = build_model_for_gene(
                cfg,
                gene,
                device,
                noisy_backend=backend,
            )
            model.load_state_dict(trained_states[key])
            accuracy, _ = evaluate_gene_model(
                cfg,
                model,
                gene,
                dataflow,
                split="test",
                device=device,
                use_qiskit=True,
            )
            backend_accuracies.append(accuracy)

        accuracy_array = np.asarray(backend_accuracies)
        results.append(accuracy_array)
        logger.info(
            f"{backend.name} test accuracies: {accuracy_array} "
            f"(mean {accuracy_array.mean():.6f})"
        )
    return results


def _synthetic_dataflow(cfg: Any, device: torch.device) -> dict[str, Any]:
    """Create tiny in-memory batches for the integration test."""
    inputs = torch.randn(2, 1, 28, 28, device=device)
    targets = torch.tensor([0, 1], device=device)
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
    """Exercise checkpoint loading, gradients, and noisy evolutionary search."""
    dataflow = _synthetic_dataflow(cfg, device)
    architecture_probe = SuperQFCModel(copy.deepcopy(cfg.model.arch))
    gene_choices = architecture_probe.arch_space
    initial_gene = [choices[0] for choices in gene_choices]

    clean_model = build_model_for_gene(cfg, initial_gene, device)
    load_supercircuit_checkpoint(clean_model, checkpoint_path)
    batch = dataflow["train"][0]
    clean_outputs = clean_model(batch[cfg.dataset.input_name])
    F.nll_loss(clean_outputs, batch[cfg.dataset.target_name]).backward()

    noisy_model = build_model_for_gene(
        cfg,
        initial_gene,
        device,
        noisy_backend=FakeYorktownV2(),
        n_shots=SMOKE_TEST_SHOTS,
    )
    noisy_model.load_state_dict(portable_state_dict(clean_model))
    searcher = AccuracyEvolutionarySearcher(
        gene_choices,
        cfg,
        dataflow,
        device,
        n_iterations=1,
        parent_size=2,
        mutation_size=1,
        crossover_size=1,
        mutation_probability=0.5,
        max_validation_samples=2,
    )
    winners = searcher.search(noisy_model, n_experiments=1)
    if len(winners) != 1 or len(winners[0]) != len(gene_choices):
        raise RuntimeError("Smoke-test search returned an invalid gene")

    logger.info(
        f"QuantumNAS smoke test passed on {device} with gene {winners[0]}."
    )


def run_experiment(
    cfg: Any,
    device: torch.device,
    checkpoint_path: Path,
    n_experiments: int,
) -> None:
    """Run noisy architecture search, retraining, and final evaluation."""
    run_name = f"quantumnas_style_search_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("runs") / run_name
    set_run_dir(str(run_dir))

    dataflow = build_dataflow(cfg, device)
    noisy_backends = [FakeYorktownV2(), FakeTorino(), FakeManilaV2()]
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Seed: {cfg.debug.seed}")
    logger.info(f"Search experiments per backend: {n_experiments}")
    logger.info(f"Noisy backends: {[backend.name for backend in noisy_backends]}")

    selected_genes = search_backends(
        cfg,
        dataflow,
        device,
        checkpoint_path,
        noisy_backends,
        n_experiments,
    )
    test_accuracies = train_and_evaluate_genes(
        cfg,
        dataflow,
        device,
        noisy_backends,
        selected_genes,
    )
    logger.info(f"Selected genes by backend: {selected_genes}")
    logger.info(f"Final noisy test accuracies: {test_accuracies}")


def main() -> None:
    """Load configuration and run the smoke test or full experiment."""
    args = parse_args()
    if args.experiments < 1:
        raise ValueError("--experiments must be at least 1")

    configs.load(args.config)
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = float(configs.optimizer.lr)
    device = configure_runtime(configs)
    checkpoint_path = Path(args.checkpoint).resolve()

    if args.smoke_test:
        run_smoke_test(configs, device, checkpoint_path)
    else:
        run_experiment(
            configs,
            device,
            checkpoint_path,
            args.experiments,
        )


if __name__ == "__main__":
    main()
