"""Expressibility-guided architecture search for a four-class MNIST QNN.

The experiment compares randomly sampled quantum circuits with circuits selected
by an evolutionary search that minimizes expressibility KL divergence. Candidate
circuits are trained once on an ideal simulator, selected on validation accuracy,
and then evaluated with several IBM fake-backend noise models.

Run the inexpensive integration check before starting the full experiment::

    python expr_search_mnist.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchquantum as tq
import tqdm
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import (
    FakeManilaV2,
    FakeTorino,
    FakeYorktownV2,
)
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from quantum_expressibility import estimate_ideal_expressibility
from torchquantum.algorithm.quantumnas.super_layers import super_layer_name_dict
from torchquantum.dataset import MNIST
from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.plugin import (
    op_history2qiskit,
    op_history2qiskit_expand_params,
    qiskit_assemble_circs,
    tq2qiskit_measurement,
)
from torchquantum.plugin.qiskit.qiskit_processor import QiskitProcessor


Gene = list[int]
StateDict = dict[str, torch.Tensor]
AccuracyPredictor = Callable[..., float]
ExpressibilityPredictor = Callable[..., float]

DIGITS_OF_INTEREST = (0, 1, 2, 3)
N_TRAIN_SAMPLES = 5_000
N_VALID_SAMPLES = 3_000
N_TEST_SAMPLES = 300
EVALUATION_INTERVAL = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs_mnist.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny clean/noisy integration check without loading MNIST.",
    )
    return parser.parse_args()


def configure_runtime(cfg: Any) -> torch.device:
    """Seed random generators and select the requested compute device."""
    if cfg.debug.set_seed:
        seed = int(cfg.debug.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

    if cfg.run.device == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")

    if cfg.run.device == "gpu":
        logger.warning("GPU was requested but CUDA is unavailable; using CPU.")
    return torch.device("cpu")


def build_dataflow(cfg: Any, device: torch.device) -> dict[str, Any]:
    """Create reproducible MNIST train, validation, and test loaders."""
    return build_mnist_dataflow(cfg, device)


def build_mnist_dataflow(
    cfg: Any,
    device: torch.device,
    *,
    fashion: bool = False,
    data_root: str = "./mnist_data",
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Create reproducible MNIST-family data loaders."""
    dataset = MNIST(
        root=data_root,
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=DIGITS_OF_INTEREST,
        n_train_samples=N_TRAIN_SAMPLES,
        n_valid_samples=N_VALID_SAMPLES,
        n_test_samples=N_TEST_SAMPLES,
        fashion=fashion,
    )

    train_generator = torch.Generator().manual_seed(int(cfg.debug.seed))
    loader_workers = (
        int(cfg.run.workers_per_gpu)
        if num_workers is None
        else int(num_workers)
    )
    if loader_workers < 0:
        raise ValueError("num_workers cannot be negative")
    loaders = {}
    for split in dataset:
        is_training = split == "train"
        loaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=int(cfg.run.bsz),
            shuffle=is_training,
            generator=train_generator if is_training else None,
            num_workers=loader_workers,
            pin_memory=device.type == "cuda",
        )
    return loaders


class SuperQFCModel(tq.QuantumModule):
    """A searchable quantum fully connected classifier."""

    def __init__(self, arch: Any) -> None:
        super().__init__()
        self.arch = arch
        self.n_wires = int(arch["n_wires"])
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict[arch["encoder_op_list_name"]]
        )
        self.q_layer = super_layer_name_dict[arch["q_layer_name"]](arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.sample_arch: Sequence[int] | None = None

    def set_sample_arch(self, sample_arch: Sequence[int]) -> None:
        self.sample_arch = sample_arch
        self.q_layer.set_sample_arch(sample_arch)

    def count_sample_params(self) -> int:
        return self.q_layer.count_sample_params()

    def forward(
        self,
        x: torch.Tensor,
        verbose: bool = False,
        use_qiskit: bool = False,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        q_device = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=batch_size,
            record_op=True,
            device=x.device,
        )

        kernel_size = getattr(self.arch, "down_sample_kernel_size", None)
        if kernel_size is not None:
            x = F.avg_pool2d(x, kernel_size)
        x = x.reshape(batch_size, -1)

        if use_qiskit:
            x = self._forward_qiskit(q_device, x)
        else:
            self.encoder(q_device, x)
            self.q_layer(q_device)
            x = self.measure(q_device)

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n{x.data}")

        output_len = getattr(self.arch, "output_len", None)
        if output_len is not None:
            x = x.reshape(batch_size, -1, output_len).sum(-1)
        elif x.ndim > 2:
            x = x.reshape(batch_size, -1)

        return F.log_softmax(x, dim=1)

    def _forward_qiskit(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the encoded circuit through the configured Qiskit processor."""
        if self.qiskit_processor is None:
            raise RuntimeError("A Qiskit processor must be set before noisy inference.")

        batch_size = x.shape[0]
        self.encoder(q_device, x)
        encoder_history = q_device.op_history
        q_device.reset_op_history()
        encoder_circuits = op_history2qiskit_expand_params(
            self.n_wires,
            encoder_history,
            bsz=batch_size,
        )

        self.q_layer(q_device)
        layer_history = q_device.op_history
        q_device.reset_op_history()
        layer_circuit = op_history2qiskit(self.n_wires, layer_history)
        measurement_circuit = tq2qiskit_measurement(q_device, self.measure)
        circuits = qiskit_assemble_circs(
            encoder_circuits,
            layer_circuit,
            measurement_circuit,
        )
        return self.qiskit_processor.process_ready_circs(
            q_device,
            circuits,
            parallel=False,
        ).to(x.device)

    @property
    def arch_space(self) -> list[list[int]]:
        """Return the discrete search choices for each gene position."""
        space = [layer.arch_space for layer in self.q_layer.super_layers_all]
        space.append(
            list(
                range(
                    self.q_layer.n_front_share_blocks,
                    self.q_layer.n_blocks + 1,
                )
            )
        )
        return space


def build_model_for_gene(
    cfg: Any,
    gene: Sequence[int],
    device: torch.device,
    noisy_backend: Any | None = None,
    n_shots: int | None = None,
) -> SuperQFCModel:
    """Build a fixed candidate model, optionally with a noisy Qiskit processor."""
    model = SuperQFCModel(copy.deepcopy(cfg.model.arch))
    model.set_sample_arch(gene)

    if noisy_backend is not None:
        noise_model = NoiseModel.from_backend(noisy_backend)
        processor = QiskitProcessor(
            use_real_qc=False,
            noise_model=noise_model,
            n_shots=int(n_shots or cfg.qiskit.n_shots),
            seed_transpiler=int(cfg.qiskit.seed_transpiler),
            seed_simulator=int(cfg.qiskit.seed_simulator),
            optimization_level=int(cfg.qiskit.optimization_level),
        )
        # Use the fake backend as the transpilation target so its topology and
        # native gates are respected without passing conflicting options.
        processor.backend = AerSimulator.from_backend(noisy_backend)
        model.set_qiskit_processor(processor)

    return model.to(device)


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Compute and log top-1 accuracy and negative log-likelihood."""
    predictions = outputs.argmax(dim=1)
    accuracy = predictions.eq(targets).float().mean().item()
    loss = F.nll_loss(outputs, targets).item()
    logger.info(f"Accuracy: {accuracy:.6f}")
    logger.info(f"Loss: {loss:.6f}")
    return accuracy, loss


def portable_state_dict(model: SuperQFCModel) -> StateDict:
    """Copy model weights while excluding TorchQuantum's runtime buffers."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if ".q_device." not in name
    }


def evaluate_gene_model(
    cfg: Any,
    model: SuperQFCModel,
    gene: Sequence[int],
    dataflow: dict[str, Any],
    split: str,
    device: torch.device,
    use_qiskit: bool = False,
    max_samples: int | None = None,
) -> tuple[float, float]:
    """Evaluate a candidate while preserving its previous train/eval mode."""
    if split not in dataflow:
        raise ValueError(f"Unknown data split: {split!r}")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")

    model.set_sample_arch(gene)
    was_training = model.training
    model.eval()
    try:
        all_targets = []
        all_outputs = []
        samples_seen = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(dataflow[split], desc=f"Evaluating {split}"):
                inputs = batch[cfg.dataset.input_name].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                targets = batch[cfg.dataset.target_name].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                if max_samples is not None:
                    remaining = max_samples - samples_seen
                    if remaining <= 0:
                        break
                    inputs = inputs[:remaining]
                    targets = targets[:remaining]

                all_outputs.append(model(inputs, use_qiskit=use_qiskit))
                all_targets.append(targets)
                samples_seen += targets.shape[0]

        return compute_metrics(
            torch.cat(all_outputs, dim=0),
            torch.cat(all_targets, dim=0),
        )
    finally:
        model.train(was_training)


def train_gene(
    cfg: Any,
    gene: Sequence[int],
    dataflow: dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> StateDict:
    """Train one gene and return the weights with best validation accuracy."""
    n_epochs = int(cfg.run.n_epochs)
    if n_epochs < 1:
        raise ValueError("configs.run.n_epochs must be at least 1")

    model = build_model_for_gene(cfg, gene, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
    )
    criterion = torch.nn.NLLLoss()

    best_accuracy = -1.0
    best_state: StateDict | None = None
    last_validation_accuracy = float("nan")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataflow["train"]:
            inputs = batch[cfg.dataset.input_name].to(
                device,
                non_blocking=device.type == "cuda",
            )
            targets = batch[cfg.dataset.target_name].to(
                device,
                non_blocking=device.type == "cuda",
            )

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        should_evaluate = (
            epoch % EVALUATION_INTERVAL == 0 or epoch == n_epochs - 1
        )
        if should_evaluate:
            last_validation_accuracy, _ = evaluate_gene_model(
                cfg,
                model,
                gene,
                dataflow,
                split="valid",
                device=device,
            )
            if last_validation_accuracy > best_accuracy:
                best_accuracy = last_validation_accuracy
                best_state = portable_state_dict(model)

        scheduler.step()
        if verbose:
            mean_loss = total_loss / len(dataflow["train"])
            logger.info(
                f"Epoch {epoch + 1:03d} | loss {mean_loss:.4f} | "
                f"validation {last_validation_accuracy:.4f} | "
                f"best {best_accuracy:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state.")
    return best_state


def make_accuracy_predictor(
    cfg: Any,
    dataflow: dict[str, Any],
    device: torch.device,
) -> AccuracyPredictor:
    """Create an evaluator that trains each unique gene exactly once."""
    trained_states: dict[tuple[int, ...], StateDict] = {}

    def predict(
        gene: Sequence[int],
        noisy_backend: Any | None = None,
        use_qiskit: bool = True,
    ) -> float:
        key = tuple(gene)
        if key not in trained_states:
            trained_states[key] = train_gene(cfg, gene, dataflow, device)

        model = build_model_for_gene(
            cfg,
            gene,
            device,
            noisy_backend=noisy_backend if use_qiskit else None,
        )
        model.load_state_dict(trained_states[key])
        accuracy, _ = evaluate_gene_model(
            cfg,
            model,
            gene,
            dataflow,
            split="test",
            device=device,
            use_qiskit=use_qiskit,
        )
        return accuracy

    return predict


class EvolutionarySearcher:
    """Simple elitist evolutionary search over discrete circuit genes."""

    def __init__(
        self,
        gene_choices: Sequence[Sequence[int]],
        accuracy_predictor: AccuracyPredictor,
        expressibility_predictor: ExpressibilityPredictor,
        cfg: Any,
    ) -> None:
        self.gene_choices = [list(choices) for choices in gene_choices]
        self.accuracy_predictor = accuracy_predictor
        self.expressibility_predictor = expressibility_predictor
        self.n_qubits = int(cfg.model.arch.n_wires)
        self.n_iterations = int(cfg.es.n_iterations)
        self.parent_size = int(cfg.es.parent_size)
        self.mutation_size = int(cfg.es.mutation_size)
        self.mutation_probability = float(cfg.es.mutation_prob)
        self.crossover_size = int(cfg.es.crossover_size)
        self.expressibility_cache: dict[tuple[int, ...], float] = {}
        self.population: list[Gene] = []
        self.best_solution: Gene | None = None

        if self.crossover_size and self.parent_size < 2:
            raise ValueError("At least two parents are required for crossover.")

    def random_sample(self, sample_count: int) -> list[Gene]:
        return [
            [random.choice(choices) for choices in self.gene_choices]
            for _ in range(sample_count)
        ]

    def expressibility_score(self, gene: Sequence[int]) -> float:
        """Return negative KL divergence so that larger scores are better."""
        key = tuple(gene)
        if key not in self.expressibility_cache:
            kl_divergence = self.expressibility_predictor(
                gene,
                n_qubits=self.n_qubits,
            )
            self.expressibility_cache[key] = -float(kl_divergence)
        return self.expressibility_cache[key]

    def select_and_transform(self, scores: Sequence[float]) -> None:
        """Keep elite parents, then generate mutations and crossovers."""
        if len(scores) != len(self.population):
            raise ValueError("Every population member must have one score.")

        parent_indices = np.argsort(-np.asarray(scores))[: self.parent_size]
        parents = [self.population[index] for index in parent_indices]
        self.best_solution = parents[0].copy()

        mutations = [
            self.mutate(random.choice(parents))
            for _ in range(self.mutation_size)
        ]
        crossovers = [
            self.crossover(random.sample(parents, 2))
            for _ in range(self.crossover_size)
        ]
        self.population = parents + mutations + crossovers

    def mutate(self, gene: Sequence[int]) -> Gene:
        return [
            random.choice(self.gene_choices[index])
            if random.random() < self.mutation_probability
            else value
            for index, value in enumerate(gene)
        ]

    def crossover(self, parents: Sequence[Sequence[int]]) -> Gene:
        return [
            parents[0][index] if random.random() < 0.5 else parents[1][index]
            for index in range(len(self.gene_choices))
        ]

    def search_best_expressibility(self, n_experiments: int) -> list[Gene]:
        """Run independent searches and return the winner from each one."""
        population_size = (
            self.parent_size + self.mutation_size + self.crossover_size
        )
        winners = []
        for _ in range(n_experiments):
            self.population = self.random_sample(population_size)
            for _ in range(self.n_iterations):
                scores = [
                    self.expressibility_score(gene) for gene in self.population
                ]
                self.select_and_transform(scores)

            if self.best_solution is None:
                raise RuntimeError("Evolutionary search produced no solution.")
            winners.append(self.best_solution.copy())
        return winners

    def run(
        self,
        noisy_backends: Sequence[Any],
        n_experiments: int,
    ) -> dict[str, list[np.ndarray]]:
        """Compare random and expressibility-guided genes on each backend."""
        random_genes = self.random_sample(n_experiments)
        evolved_genes = self.search_best_expressibility(n_experiments)
        logger.info(f"Random genes: {random_genes}")
        logger.info(f"Expressibility-guided genes: {evolved_genes}")

        random_results = []
        evolved_results = []
        for backend in noisy_backends:
            random_accuracies = np.asarray(
                [
                    self.accuracy_predictor(
                        gene=gene,
                        noisy_backend=backend,
                        use_qiskit=True,
                    )
                    for gene in random_genes
                ]
            )
            evolved_accuracies = np.asarray(
                [
                    self.accuracy_predictor(
                        gene=gene,
                        noisy_backend=backend,
                        use_qiskit=True,
                    )
                    for gene in evolved_genes
                ]
            )
            random_results.append(random_accuracies)
            evolved_results.append(evolved_accuracies)
            logger.info(
                f"{backend.name} random accuracies: {random_accuracies} "
                f"(mean {random_accuracies.mean():.4f})"
            )
            logger.info(
                f"{backend.name} guided accuracies: {evolved_accuracies} "
                f"(mean {evolved_accuracies.mean():.4f})"
            )

        return {"random": random_results, "evolutionary": evolved_results}


def run_smoke_test(
    cfg: Any,
    device: torch.device,
    dataset_label: str = "MNIST",
) -> None:
    """Exercise model construction, gradients, expressibility, and noisy inference."""
    architecture_probe = SuperQFCModel(copy.deepcopy(cfg.model.arch))
    gene = [choices[0] for choices in architecture_probe.arch_space]
    inputs = torch.randn(2, 1, 28, 28, device=device)
    targets = torch.tensor([0, 1], device=device)

    model = build_model_for_gene(cfg, gene, device)
    outputs = model(inputs)
    if outputs.shape != (2, len(DIGITS_OF_INTEREST)):
        raise RuntimeError(f"Unexpected clean output shape: {outputs.shape}")
    F.nll_loss(outputs, targets).backward()

    kl_divergence = estimate_ideal_expressibility(
        gene,
        n_qubits=int(cfg.model.arch.n_wires),
        n_pairs=4,
        n_bins=8,
        seed=int(cfg.debug.seed),
    )
    if not np.isfinite(kl_divergence):
        raise RuntimeError("Smoke-test expressibility is not finite.")

    noisy_model = build_model_for_gene(
        cfg,
        gene,
        device,
        noisy_backend=FakeYorktownV2(),
        n_shots=32,
    )
    noisy_model.load_state_dict(portable_state_dict(model))
    noisy_model.eval()
    with torch.no_grad():
        noisy_outputs = noisy_model(inputs, use_qiskit=True)
    if noisy_outputs.shape != outputs.shape or not torch.isfinite(noisy_outputs).all():
        raise RuntimeError("Noisy inference returned invalid output.")

    logger.info(
        f"{dataset_label} smoke test passed on {device} "
        f"(gene={gene}, KL={kl_divergence:.6f})."
    )


def run_experiment(
    cfg: Any,
    device: torch.device,
    *,
    fashion: bool = False,
    data_root: str = "./mnist_data",
    run_name_prefix: str = "expr_search_mnist",
    dataset_label: str = "MNIST",
) -> None:
    """Build data and execute the complete architecture-search experiment."""
    run_name = f"{run_name_prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("runs") / run_name
    set_run_dir(str(run_dir))

    dataflow = build_mnist_dataflow(
        cfg,
        device,
        fashion=fashion,
        data_root=data_root,
    )
    architecture_probe = SuperQFCModel(copy.deepcopy(cfg.model.arch))
    gene_choices = architecture_probe.arch_space
    noisy_backends = [FakeYorktownV2(), FakeTorino(), FakeManilaV2()]

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Dataset: {dataset_label}")
    logger.info(f"TorchQuantum source: {Path(tq.__file__).resolve().parent}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {cfg.debug.seed}")
    logger.info(f"Epochs: {cfg.run.n_epochs}")
    logger.info(f"Batch size: {cfg.run.bsz}")
    logger.info(f"Gene choices: {gene_choices}")
    logger.info(f"Noisy backends: {[backend.name for backend in noisy_backends]}")

    searcher = EvolutionarySearcher(
        gene_choices=gene_choices,
        accuracy_predictor=make_accuracy_predictor(cfg, dataflow, device),
        expressibility_predictor=estimate_ideal_expressibility,
        cfg=cfg,
    )
    results = searcher.run(noisy_backends, n_experiments=5)
    logger.info(f"Random results: {results['random']}")
    logger.info(f"Evolutionary results: {results['evolutionary']}")


def main() -> None:
    """Load configuration and run either the smoke test or full experiment."""
    args = parse_args()
    configs.load(args.config)
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = float(configs.optimizer.lr)
    device = configure_runtime(configs)

    if args.smoke_test:
        run_smoke_test(configs, device)
    else:
        run_experiment(configs, device)


if __name__ == "__main__":
    main()
