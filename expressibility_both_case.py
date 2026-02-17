import numpy as np
from math import pi
import random
from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.special import rel_entr
from qiskit import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator


# ---------------------------------------------------------
# Circular nearest-neighbor ring pairs
# ---------------------------------------------------------
def make_ring_pairs(n):
    return [(i, (i+1) % n) for i in range(n)]

# ---------------------------------------------------------
# expressibility without noise calculation
# ---------------------------------------------------------
def compute_expressibility_without_noise(
    sample_arch,
    n_qubits,
    n_pairs=200,
    nbins=75,
    seed=123,
):
    
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------ 1. Build circuit generator ------------------------
    n_blocks = sample_arch[-1]
    pairs = make_ring_pairs(n_qubits)

    def sample_circuit():
        qc = QuantumCircuit(n_qubits)
        for b in range(n_blocks):
            u3_k = sample_arch[2*b]
            cu3_m = sample_arch[2*b + 1]

            # U3 layer (all parameters independent)
            for w in range(min(u3_k, n_qubits)):
                theta, phi, lam = 2*pi*np.random.rand(3)
                qc.append(UGate(theta, phi, lam), [w])

            # CU3 layer (all parameters independent)
            for (ctrl, tgt) in pairs[:min(cu3_m, len(pairs))]:
                theta, phi, lam = 2*pi*np.random.rand(3)
                CU = UGate(theta, phi, lam).control(1)
                qc.append(CU, [ctrl, tgt])

        return qc

    # ------------------------ 2. Collect fidelities ------------------------
    fidelities = []
    for i in range(n_pairs):
        # deterministic sequence because random seeds are fixed
        qc1 = sample_circuit()
        qc2 = sample_circuit()

        psi1 = Statevector.from_instruction(qc1)
        psi2 = Statevector.from_instruction(qc2)

        f = state_fidelity(psi1, psi2)
        fidelities.append(f)

    fidelities = np.array(fidelities)

    # ------------------------ 3. P_emp histogram ------------------------
    weights = np.ones_like(fidelities, dtype=float) / len(fidelities)
    P_emp, bin_edges = np.histogram(fidelities, bins=nbins, range=(0.0,1.0), weights=weights)
    # P_emp = counts.astype(float)
    # P_emp /= P_emp.sum()

    # ------------------------ 4. Haar discrete distribution ------------------------
    d = 2 ** n_qubits
    a = d - 2

    bin_left = bin_edges[:-1]
    bin_right = bin_edges[1:]

    def haar_cdf(x):
        return 1 - (1 - x) ** (a + 1)

    P_haar = haar_cdf(bin_right) - haar_cdf(bin_left)
    P_haar /= P_haar.sum()

    # ------------------------ 5. KL divergence ------------------------
    eps = 1e-12
    P = np.clip(P_emp,  eps, 1); P = P / P.sum()
    Q = np.clip(P_haar, eps, 1); Q = Q / Q.sum()
    KL = float(np.sum(rel_entr(P, Q)))

    return KL


def compute_expressibility_noisy(
    sample_arch,
    n_qubits,
    fake_backend=None,
    n_pairs=200,
    nbins=75,
    seed=123,
):
    """
    Expressibility (KL) under noise using density matrices.
    Returns:
        KL_HS, KL_Uhlmann
    Deterministic given fixed seeds.
    """

    # ------------------------ 0. RNG seeds ------------------------
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------ 1. Noise model ------------------------
    if fake_backend is None:
        raise ValueError("You must provide a fake backend (e.g. FakeFez())")

    noise_model = NoiseModel.from_backend(fake_backend)

    sim = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
    )

    # ------------------------ 2. Circuit generator ------------------------
    n_blocks = sample_arch[-1]
    pairs = make_ring_pairs(n_qubits)

    def sample_circuit():
        qc = QuantumCircuit(n_qubits)
        for b in range(n_blocks):
            u3_k = sample_arch[2 * b]
            cu3_m = sample_arch[2 * b + 1]

            for w in range(min(u3_k, n_qubits)):
                theta, phi, lam = 2 * pi * np.random.rand(3)
                qc.append(UGate(theta, phi, lam), [w])

            for (ctrl, tgt) in pairs[:min(cu3_m, len(pairs))]:
                theta, phi, lam = 2 * pi * np.random.rand(3)
                qc.append(UGate(theta, phi, lam).control(1), [ctrl, tgt])

        qc.save_density_matrix()
        return qc

    # ------------------------ 3. Collect noisy density matrices ------------------------
    circuits = []
    for _ in range(n_pairs):
        circuits.extend([sample_circuit(), sample_circuit()])

    tcircuits = transpile(circuits, sim)
    result = sim.run(tcircuits).result()

    rhos = [result.data(i)["density_matrix"] for i in range(len(tcircuits))]

    # ------------------------ 4. Fidelities ------------------------
    fidelities_hs = []
    fidelities_uhlmann = []

    for i in range(0, len(rhos), 2):
        rho1, rho2 = rhos[i], rhos[i + 1]

        f_hs = float(np.real(np.trace(rho1.data @ rho2.data)))
        f_uhl = state_fidelity(rho1, rho2)

        fidelities_hs.append(f_hs)
        fidelities_uhlmann.append(f_uhl)

    # ------------------------ 5. Histogram helper ------------------------
    def compute_kl(fidelities):
        weights = np.ones_like(fidelities) / len(fidelities)

        P_emp, bin_edges = np.histogram(
            fidelities, bins=nbins, range=(0.0, 1.0), weights=weights
        )

        # Haar reference
        d = 2 ** n_qubits
        a = d - 2

        bin_left = bin_edges[:-1]
        bin_right = bin_edges[1:]

        def haar_cdf(x):
            return 1 - (1 - x) ** (a + 1)

        P_haar = haar_cdf(bin_right) - haar_cdf(bin_left)
        P_haar /= P_haar.sum()

        eps = 1e-12
        P = np.clip(P_emp, eps, 1); P /= P.sum()
        Q = np.clip(P_haar, eps, 1); Q /= Q.sum()

        return float(np.sum(rel_entr(P, Q)))

    # ------------------------ 6. KL divergences ------------------------
    KL_HS = compute_kl(fidelities_hs)
    KL_Uhlmann = compute_kl(fidelities_uhlmann)

    return KL_HS, KL_Uhlmann
