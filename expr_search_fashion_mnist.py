"""Expressibility-guided architecture search for Fashion-MNIST.

This entry point uses the shared, tested architecture-search implementation from
``expr_search_mnist`` with Fashion-MNIST data. Candidate circuits classify the
first four Fashion-MNIST classes and are compared across IBM fake-backend noise
models.

Run the lightweight integration check before the full experiment::

    python expr_search_fashion_mnist.py --smoke-test
"""

from __future__ import annotations

import argparse

from torchpack.utils.config import configs

from expr_search_mnist import (
    configure_runtime,
    run_experiment,
    run_smoke_test,
)


DEFAULT_CONFIG = "configs_fashion_mnist.yaml"
FASHION_MNIST_DATA_ROOT = "./fashion_mnist_data"
RUN_NAME_PREFIX = "expr_search_fashion_mnist"
DATASET_LABEL = "Fashion-MNIST"


def parse_args() -> argparse.Namespace:
    """Parse Fashion-MNIST experiment arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny clean/noisy check without loading Fashion-MNIST.",
    )
    return parser.parse_args()


def main() -> None:
    """Load configuration and run the smoke test or Fashion-MNIST experiment."""
    args = parse_args()
    configs.load(args.config)
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = float(configs.optimizer.lr)
    device = configure_runtime(configs)

    if args.smoke_test:
        run_smoke_test(configs, device, dataset_label=DATASET_LABEL)
    else:
        run_experiment(
            configs,
            device,
            fashion=True,
            data_root=FASHION_MNIST_DATA_ROOT,
            run_name_prefix=RUN_NAME_PREFIX,
            dataset_label=DATASET_LABEL,
        )


if __name__ == "__main__":
    main()
