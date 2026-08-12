"""Estimate the expressibility of parameterized quantum circuit architectures.

Expressibility is measured as the Kullback-Leibler (KL) divergence between the
distribution of pairwise circuit-state fidelities and the corresponding Haar
distribution. A smaller KL divergence indicates a more expressive architecture.

Two estimators are provided:

* :func:`estimate_ideal_expressibility` uses noiseless statevectors.
* :func:`estimate_noisy_expressibility` uses density matrices generated with a
  Qiskit Aer noise model.

Both functions use local random generators, so they are reproducible without
modifying NumPy's global random state.
"""

from __future__ import annotations

from math import pi
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from scipy.special import rel_entr


__all__ = [
    "estimate_ideal_expressibility",
    "estimate_noisy_expressibility",
]

DEFAULT_N_PAIRS = 200
DEFAULT_N_BINS = 75
DEFAULT_SEED = 123
_PROBABILITY_EPSILON = 1e-12


def _validate_inputs(
    architecture: Sequence[int],
    n_qubits: int,
    n_pairs: int,
    n_bins: int,
) -> None:
    """Validate architecture and sampling parameters."""
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2")
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if len(architecture) < 3:
        raise ValueError("architecture must contain layer choices and a block count")

    n_blocks = int(architecture[-1])
    max_described_blocks = (len(architecture) - 1) // 2
    if not 1 <= n_blocks <= max_described_blocks:
        raise ValueError(
            f"architecture requests {n_blocks} blocks but describes "
            f"{max_described_blocks}"
        )
    if any(int(value) < 0 for value in architecture[:-1]):
        raise ValueError("architecture layer choices must be non-negative")


def _ring_entanglement_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Return directed nearest-neighbor pairs on a circular qubit layout."""
    return [(index, (index + 1) % n_qubits) for index in range(n_qubits)]


def _sample_circuit(
    architecture: Sequence[int],
    n_qubits: int,
    rng: np.random.Generator,
    save_density_matrix: bool = False,
) -> QuantumCircuit:
    """Sample one parameterized circuit for an architecture."""
    circuit = QuantumCircuit(n_qubits)
    entanglement_pairs = _ring_entanglement_pairs(n_qubits)

    for block_index in range(int(architecture[-1])):
        n_single_qubit_gates = int(architecture[2 * block_index])
        n_controlled_gates = int(architecture[2 * block_index + 1])

        for wire in range(min(n_single_qubit_gates, n_qubits)):
            theta, phi, lam = 2 * pi * rng.random(3)
            circuit.append(UGate(theta, phi, lam), [wire])

        active_pairs = entanglement_pairs[:n_controlled_gates]
        for control, target in active_pairs:
            theta, phi, lam = 2 * pi * rng.random(3)
            controlled_u_gate = UGate(theta, phi, lam).control(1)
            circuit.append(controlled_u_gate, [control, target])

    if save_density_matrix:
        circuit.save_density_matrix()
    return circuit


def _haar_bin_probabilities(n_qubits: int, bin_edges: np.ndarray) -> np.ndarray:
    """Integrate the Haar fidelity distribution over histogram bins."""
    hilbert_space_dimension = 2**n_qubits

    def haar_cdf(fidelity: np.ndarray) -> np.ndarray:
        return 1 - (1 - fidelity) ** (hilbert_space_dimension - 1)

    probabilities = haar_cdf(bin_edges[1:]) - haar_cdf(bin_edges[:-1])
    return probabilities / probabilities.sum()


def _estimate_kl_divergence(
    fidelities: Sequence[float],
    n_qubits: int,
    n_bins: int,
) -> float:
    """Compare empirical fidelities with the discretized Haar distribution."""
    fidelity_array = np.clip(np.asarray(fidelities, dtype=float), 0.0, 1.0)
    empirical_probabilities, bin_edges = np.histogram(
        fidelity_array,
        bins=n_bins,
        range=(0.0, 1.0),
    )
    empirical_probabilities = empirical_probabilities.astype(float)
    empirical_probabilities /= empirical_probabilities.sum()

    haar_probabilities = _haar_bin_probabilities(n_qubits, bin_edges)
    empirical_probabilities = np.clip(
        empirical_probabilities,
        _PROBABILITY_EPSILON,
        1.0,
    )
    haar_probabilities = np.clip(
        haar_probabilities,
        _PROBABILITY_EPSILON,
        1.0,
    )
    empirical_probabilities /= empirical_probabilities.sum()
    haar_probabilities /= haar_probabilities.sum()
    return float(np.sum(rel_entr(empirical_probabilities, haar_probabilities)))


def estimate_ideal_expressibility(
    architecture: Sequence[int],
    n_qubits: int,
    n_pairs: int = DEFAULT_N_PAIRS,
    n_bins: int = DEFAULT_N_BINS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Estimate expressibility KL divergence using noiseless statevectors.

    Args:
        architecture: Alternating U3/CU3 layer sizes followed by the active
            block count.
        n_qubits: Number of circuit qubits.
        n_pairs: Number of independently sampled circuit pairs.
        n_bins: Number of fidelity histogram bins.
        seed: Seed for locally sampled gate parameters.

    Returns:
        KL divergence from the Haar fidelity distribution. Lower is better.
    """
    _validate_inputs(architecture, n_qubits, n_pairs, n_bins)
    rng = np.random.default_rng(seed)
    fidelities = []

    for _ in range(n_pairs):
        first_state = Statevector.from_instruction(
            _sample_circuit(architecture, n_qubits, rng)
        )
        second_state = Statevector.from_instruction(
            _sample_circuit(architecture, n_qubits, rng)
        )
        fidelities.append(state_fidelity(first_state, second_state))

    return _estimate_kl_divergence(fidelities, n_qubits, n_bins)


def estimate_noisy_expressibility(
    architecture: Sequence[int],
    n_qubits: int,
    backend: Any,
    n_pairs: int = DEFAULT_N_PAIRS,
    n_bins: int = DEFAULT_N_BINS,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Estimate expressibility with a backend-derived Aer noise model.

    Returns:
        A pair containing KL divergences based on Hilbert-Schmidt overlap and
        Uhlmann fidelity, respectively. Lower values indicate distributions
        closer to Haar-random states.
    """
    _validate_inputs(architecture, n_qubits, n_pairs, n_bins)
    if backend is None:
        raise ValueError("backend is required for noisy expressibility")

    rng = np.random.default_rng(seed)
    simulator = AerSimulator(
        method="density_matrix",
        noise_model=NoiseModel.from_backend(backend),
    )
    circuits = [
        _sample_circuit(
            architecture,
            n_qubits,
            rng,
            save_density_matrix=True,
        )
        for _ in range(2 * n_pairs)
    ]
    transpiled_circuits = transpile(
        circuits,
        simulator,
        seed_transpiler=seed,
    )
    result = simulator.run(
        transpiled_circuits,
        seed_simulator=seed,
    ).result()
    density_matrices = [
        np.asarray(result.data(index)["density_matrix"])
        for index in range(len(transpiled_circuits))
    ]

    hilbert_schmidt_overlaps = []
    uhlmann_fidelities = []
    for index in range(0, len(density_matrices), 2):
        first_density_matrix = density_matrices[index]
        second_density_matrix = density_matrices[index + 1]
        hilbert_schmidt_overlaps.append(
            float(
                np.real(
                    np.trace(first_density_matrix @ second_density_matrix)
                )
            )
        )
        uhlmann_fidelities.append(
            state_fidelity(first_density_matrix, second_density_matrix)
        )

    return (
        _estimate_kl_divergence(
            hilbert_schmidt_overlaps,
            n_qubits,
            n_bins,
        ),
        _estimate_kl_divergence(
            uhlmann_fidelities,
            n_qubits,
            n_bins,
        ),
    )
