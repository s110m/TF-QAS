"""Train a weight-sharing quantum supercircuit.

The resulting validation-selected checkpoint is consumed by
``quantumnas_style_search.py`` and ``loss_expr_relation.py``. The default
configuration trains a four-class MNIST supercircuit.

Run the lightweight compatibility check before full training::

    python train_supercircuit.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.algorithm.quantumnas.super_layers
import torchquantum.device
import torchquantum.measurement
import torchquantum.operator
from torchpack.environ import set_run_dir
from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from expr_search_mnist import SuperQFCModel, portable_state_dict
from torchquantum.algorithm.quantumnas.super_utils import get_named_sample_arch
from torchquantum.plugin import qiskit2tq, tq2qiskit
from torchquantum.plugin.qiskit.qiskit_processor import QiskitProcessor
from torchquantum.util import (
    build_module_from_op_list,
    build_module_op_list,
    get_cared_configs,
    get_p_c_reg_mapping,
    get_p_v_reg_mapping,
    get_v_c_reg_mapping,
)


DEFAULT_CONFIG = "configs_train_supercircuit_mnist.yml"
DEFAULT_RUNS_DIR = Path("runs")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse script arguments and preserve TorchPack configuration overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint-dir",
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Base directory for the configured resume checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to runs/<configuration-name>.",
    )
    parser.add_argument("--gpu", help="CUDA device IDs, for example '0' or '0,1'.")
    parser.add_argument("--pdb", action="store_true")
    parser.add_argument("--print-configs", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers (use 0 in restricted Windows shells).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one synthetic optimization step and checkpoint check.",
    )
    return parser.parse_known_args()


def configure_runtime(cfg: Any, gpu_ids: str | None) -> torch.device:
    """Seed random generators and select the configured device."""
    if gpu_ids is not None:
        # CUDA reads this value lazily when its runtime is initialized.
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

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
        cfg.run.device = "cpu"
        return torch.device("cpu")
    if cfg.run.device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported run device: {cfg.run.device!r}")


def default_run_dir(config_path: str) -> Path:
    """Derive a stable run directory from a configuration filename."""
    config_name = Path(config_path).stem
    if config_name.startswith("configs_"):
        config_name = config_name.removeprefix("configs_")
    return DEFAULT_RUNS_DIR / config_name


def build_dataflow(
    cfg: Any,
    num_workers: int | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build reproducible train, validation, and test loaders."""
    workers = (
        int(cfg.run.workers_per_gpu)
        if num_workers is None
        else int(num_workers)
    )
    if workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = builder.make_dataset()
    train_generator = torch.Generator().manual_seed(int(cfg.debug.seed))
    dataflow = {}
    for split in dataset:
        is_training = split == "train"
        batch_size = (
            int(cfg.run.bsz)
            if is_training
            else int(getattr(cfg.run, "eval_bsz", cfg.run.bsz))
        )
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=is_training,
            generator=train_generator if is_training else None,
            num_workers=workers,
            pin_memory=cfg.run.device == "gpu",
        )
    return dataflow


def _register_legacy_checkpoint_aliases(model: torch.nn.Module) -> None:
    """Register module aliases needed by checkpoints from older scripts."""
    sys.modules["__main__"].SuperQFCModel0 = model.__class__
    aliases = {
        "torchquantum.devices": torchquantum.device,
        "torchquantum.super_layers": (
            torchquantum.algorithm.quantumnas.super_layers
        ),
        "torchquantum.operators": torchquantum.operator,
        "torchquantum.measure": torchquantum.measurement,
    }
    for legacy_name, current_module in aliases.items():
        sys.modules.setdefault(legacy_name, current_module)


def resolve_checkpoint_path(
    checkpoint_dir: Path | None,
    configured_name: str,
) -> Path:
    """Resolve a configured checkpoint name against an optional base directory."""
    checkpoint_path = Path(configured_name)
    if checkpoint_dir is not None and not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_dir / checkpoint_path
    return checkpoint_path.resolve()


def load_training_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    cfg: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """Load model state and optional architecture-search metadata."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _register_legacy_checkpoint_aliases(model)
    checkpoint = io.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dictionary")

    if not bool(getattr(cfg.ckpt, "weight_from_scratch", False)):
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        logger.warning("Checkpoint weights ignored; training from scratch.")

    solution = checkpoint.get("solution")
    score = checkpoint.get("score")
    if solution is not None:
        model.set_sample_arch(solution["arch"])
        logger.info(f"Loaded architecture solution: {solution}")

    register_mapping = checkpoint.get("v_c_reg_mapping")
    if register_mapping is not None and hasattr(model, "measure"):
        model.measure.set_v_c_reg_mapping(register_mapping)

    if cfg.model.load_op_list:
        operation_list = checkpoint.get("q_layer_op_list")
        if operation_list is None:
            raise ValueError("Checkpoint has no q_layer_op_list")
        model.q_layer = build_module_from_op_list(operation_list)
    return dict(checkpoint), solution, score


def make_qiskit_processor(cfg: Any) -> QiskitProcessor:
    """Build a processor using arguments supported by the current API."""
    return QiskitProcessor(
        use_real_qc=bool(cfg.qiskit.use_real_qc),
        backend_name=cfg.qiskit.backend_name,
        noise_model_name=cfg.qiskit.noise_model_name,
        n_shots=int(cfg.qiskit.n_shots),
        initial_layout=cfg.qiskit.initial_layout,
        seed_transpiler=int(cfg.qiskit.seed_transpiler),
        seed_simulator=int(cfg.qiskit.seed_simulator),
        optimization_level=int(cfg.qiskit.optimization_level),
        max_jobs=int(cfg.qiskit.max_jobs),
    )


def transpile_model_if_requested(
    model: torch.nn.Module,
    cfg: Any,
    solution: Any,
) -> None:
    """Transpile quantum layers and update logical/physical register mappings."""
    if not cfg.model.transpile_before_run:
        return

    processor = make_qiskit_processor(cfg)
    modules = [model] if hasattr(model, "q_layer") else list(model.nodes)
    for module in modules:
        circuit = tq2qiskit(module.q_device, module.q_layer)
        wires = list(range(module.q_device.n_wires))
        circuit.measure(wires, wires)
        if solution is not None:
            processor.set_layout(solution["layout"])
        transpiled_circuit = processor.transpile(circuit)
        module.q_layer = qiskit2tq(circ=transpiled_circuit)
        register_mapping = get_v_c_reg_mapping(transpiled_circuit)
        module.measure.set_v_c_reg_mapping(register_mapping)

        if cfg.trainer.add_noise:
            noise_model = builder.make_noise_model_tq()
            noise_model.is_add_noise = True
            noise_model.v_c_reg_mapping = register_mapping
            noise_model.p_c_reg_mapping = get_p_c_reg_mapping(
                transpiled_circuit
            )
            noise_model.p_v_reg_mapping = get_p_v_reg_mapping(
                transpiled_circuit
            )
            module.set_noise_model_tq(noise_model)


def configure_sample_architecture(model: torch.nn.Module, cfg: Any) -> None:
    """Apply an optional fixed architecture from configuration."""
    sample_architecture = getattr(cfg.model.arch, "sample_arch", None)
    if sample_architecture is None or cfg.model.transpile_before_run:
        return
    if isinstance(sample_architecture, str):
        sample_architecture = get_named_sample_arch(
            model.arch_space,
            sample_architecture,
        )
    model.set_sample_arch(sample_architecture)
    logger.info(f"Using fixed sample architecture: {sample_architecture}")


def configure_pruning_model(model: torch.nn.Module, cfg: Any) -> None:
    """Convert a sampled super layer for pruning-aware training when requested."""
    if cfg.trainer.name == "pruning_trainer":
        model.q_layer = build_module_from_op_list(
            build_module_op_list(model.q_layer)
        )


def build_training_components(
    model: torch.nn.Module,
) -> tuple[Any, Any, Any, Any]:
    """Construct the configured criterion, optimizer, scheduler, and trainer."""
    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    trainer = builder.make_trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return criterion, optimizer, scheduler, trainer


def run_smoke_test(cfg: Any, device: torch.device) -> None:
    """Run one sampled optimization step and verify checkpoint compatibility."""
    model = builder.make_model().to(device)
    gene = [choices[0] for choices in model.arch_space]
    model.set_sample_arch(gene)
    optimizer = builder.make_optimizer(model)
    inputs = torch.randn(3, 1, 28, 28, device=device)
    targets = torch.tensor([0, 1, 2], device=device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    if outputs.shape != (3, 4):
        raise RuntimeError(f"Unexpected supercircuit output shape: {outputs.shape}")
    loss = F.nll_loss(outputs, targets)
    loss.backward()
    optimizer.step()

    search_model = SuperQFCModel(copy.deepcopy(cfg.model.arch)).to(device)
    search_model.set_sample_arch(gene)
    incompatible = search_model.load_state_dict(
        portable_state_dict(model),
        strict=False,
    )
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Search model is missing checkpoint keys: {incompatible.missing_keys}"
        )
    unsupported_keys = [
        key
        for key in incompatible.unexpected_keys
        if not key.endswith((".U3_params", ".CU3_params"))
    ]
    if unsupported_keys:
        raise RuntimeError(
            f"Search model rejected checkpoint keys: {unsupported_keys}"
        )
    with torch.no_grad():
        search_outputs = search_model(inputs)
    if search_outputs.shape != outputs.shape:
        raise RuntimeError("Search model rejected the trained checkpoint state")
    if not torch.isfinite(search_outputs).all():
        raise RuntimeError("Checkpoint compatibility check produced invalid output")

    logger.info(
        f"Supercircuit smoke test passed on {device} "
        f"(gene={gene}, loss={loss.item():.6f})."
    )


def run_training(
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Build all configured components and train the supercircuit."""
    run_dir = args.run_dir or default_run_dir(args.config)
    set_run_dir(str(run_dir))
    logger.info(f"Training command: {' '.join([sys.executable] + sys.argv)}")
    displayed_config = (
        cfg if args.print_configs else get_cared_configs(cfg, "train")
    )
    logger.info(f"Training started in {run_dir}:\n{displayed_config}")

    dataflow = build_dataflow(cfg, args.num_workers)
    model = builder.make_model()
    checkpoint_state: dict[str, Any] = {}
    solution = None
    score = None
    if cfg.ckpt.load_ckpt:
        checkpoint_path = resolve_checkpoint_path(
            args.checkpoint_dir,
            cfg.ckpt.name,
        )
        checkpoint_state, solution, score = load_training_checkpoint(
            model,
            checkpoint_path,
            cfg,
        )

    transpile_model_if_requested(model, cfg, solution)
    configure_sample_architecture(model, cfg)
    configure_pruning_model(model, cfg)
    model.to(device)
    logger.info(
        f"Trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters())}"
    )

    _, _, _, trainer = build_training_components(model)
    trainer.solution = solution
    trainer.score = score
    callbacks = builder.make_callbacks(dataflow, checkpoint_state)
    trainer.train_with_defaults(
        dataflow["train"],
        num_epochs=int(cfg.run.n_epochs),
        callbacks=callbacks,
    )


def main() -> None:
    """Load configuration and run a smoke test or full training."""
    args, overrides = parse_args()
    configs.load(args.config, recursive=True)
    configs.update(overrides)
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = float(configs.optimizer.lr)
    if args.pdb or configs.debug.pdb:
        breakpoint()

    device = configure_runtime(configs, args.gpu)
    if args.smoke_test:
        run_smoke_test(configs, device)
    else:
        run_training(configs, args, device)


if __name__ == "__main__":
    main()
