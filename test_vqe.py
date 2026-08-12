import random
import torch
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from collections import Counter, OrderedDict
from torchquantum.functional import mat_dict
from torchquantum.macro import F_DTYPE
import torchquantum.operator as op


# ============================================================
# measure()
# ============================================================

def gen_bitstrings(n_wires):
    return [
        "{:0{}b}".format(k, n_wires)
        for k in range(2 ** n_wires)
    ]


def measure(qdev, n_shots=1024):

    bitstring_candidates = gen_bitstrings(
        qdev.n_wires
    )

    state_mag = qdev.get_states_1d().abs().detach().cpu().numpy()

    distri_all = []

    for state_mag_one in state_mag:

        state_prob_one = np.abs(state_mag_one) ** 2

        measured = random.choices(
            population=bitstring_candidates,
            weights=state_prob_one,
            k=n_shots,
        )

        counter = Counter(measured)

        counter.update({
            key: 0
            for key in bitstring_candidates
        })

        distri = dict(counter)

        distri = OrderedDict(
            sorted(distri.items())
        )

        distri_all.append(distri)

    return distri_all


# ============================================================
# METHOD 1
# Original expval()
# ============================================================

def expval(
        qdev: tq.QuantumDevice,
        wires,
        observables,
):

    all_dims = np.arange(qdev.states.dim())

    # rotate basis
    for wire, observable in zip(wires, observables):
        for rotation in observable.diagonalizing_gates():
            rotation(qdev, wires=wire)

    states = qdev.states

    state_mag = torch.abs(states) ** 2

    expectations = []

    for wire, observable in zip(wires, observables):

        reduction_dims = np.delete(
            all_dims,
            [0, wire + 1]
        )

        if reduction_dims.size == 0:
            probs = state_mag
        else:
            probs = state_mag.sum(list(reduction_dims))

        res = probs.mv(
            observable.eigvals.real.to(probs.device)
        )

        expectations.append(res)

    return torch.stack(expectations, dim=-1)


# ============================================================
# Original MeasureMultipleTimes
# ============================================================

class MeasureMultipleTimes(tq.QuantumModule):

    def __init__(self, obs_list):
        super().__init__()

        self.obs_list = obs_list

    def forward(self, qdev: tq.QuantumDevice):

        res_all = []

        for layer in self.obs_list:

            qdev_new = tq.QuantumDevice(
                n_wires=qdev.n_wires
            )

            qdev_new.clone_states(
                existing_states=qdev.states
            )

            observables = []

            for wire in range(qdev.n_wires):
                observables.append(tq.I())

            for wire, observable in zip(
                    layer["wires"],
                    layer["observables"]):

                observables[wire] = tq.op_name_dict[
                    observable
                ]()

            res = expval(
                qdev_new,
                wires=list(range(qdev.n_wires)),
                observables=observables,
            )

            res_all.append(res)

        return torch.cat(res_all)


# ============================================================
# METHOD 2
# Correct analytical joint expectation
# ============================================================

def expval_joint_analytical(
        qdev: tq.QuantumDevice,
        observable: str,
):

    paulix = mat_dict["paulix"]
    pauliy = mat_dict["pauliy"]
    pauliz = mat_dict["pauliz"]
    iden = mat_dict["i"]

    pauli_dict = {
        "X": paulix,
        "Y": pauliy,
        "Z": pauliz,
        "I": iden
    }

    observable = observable.upper()

    states = qdev.get_states_1d()

    hamiltonian = pauli_dict[
        observable[0]
    ].to(states.device)

    for op_char in observable[1:]:

        hamiltonian = torch.kron(
            hamiltonian,
            pauli_dict[op_char].to(states.device)
        )

    return (
        (
            states.conj()
            *
            torch.mm(
                hamiltonian,
                states.transpose(0, 1)
            ).transpose(0, 1)
        ).sum(-1).real
    )


# ============================================================
# METHOD 3
# Correct sampling-based joint expectation
# ============================================================

def expval_joint_sampling(
        qdev: tq.QuantumDevice,
        observable: str,
        n_shots=10000,
):

    n_wires = qdev.n_wires

    paulix = op.op_name_dict["paulix"]
    pauliy = op.op_name_dict["pauliy"]
    pauliz = op.op_name_dict["pauliz"]
    iden = op.op_name_dict["i"]

    pauli_dict = {
        "X": paulix,
        "Y": pauliy,
        "Z": pauliz,
        "I": iden
    }

    qdev_clone = tq.QuantumDevice(
        n_wires=qdev.n_wires,
        bsz=qdev.bsz
    )

    qdev_clone.clone_states(qdev.states)

    observable = observable.upper()

    # rotate basis
    for wire in range(n_wires):

        for rotation in pauli_dict[
                observable[wire]
        ]().diagonalizing_gates():

            rotation(qdev_clone, wires=wire)

    mask = np.ones(
        len(observable),
        dtype=bool
    )

    mask[np.array([*observable]) == "I"] = False

    distributions = measure(
        qdev_clone,
        n_shots=n_shots
    )

    expval_all = []

    for distri in distributions:

        n_eigen_one = 0
        n_eigen_minus_one = 0

        for bitstring, n_count in distri.items():

            parity = np.dot(
                list(map(lambda x: eval(x), [*bitstring])),
                mask
            ).sum()

            if parity % 2 == 0:
                n_eigen_one += n_count
            else:
                n_eigen_minus_one += n_count

        expval = (
            n_eigen_one / n_shots
            -
            n_eigen_minus_one / n_shots
        )

        expval_all.append(expval)

    return torch.tensor(
        expval_all,
        dtype=F_DTYPE
    )


# ============================================================
# Create Bell state
# ============================================================

qdev = tq.QuantumDevice(
    n_wires=2,
    bsz=1
)

tqf.hadamard(qdev, wires=0)
tqf.cnot(qdev, wires=[0, 1])

print("=" * 70)
print("Bell state")
print("=" * 70)

print(qdev.get_states_1d())

print("\n")


# ============================================================
# METHOD 1
# ============================================================

print("=" * 70)
print("METHOD 1")
print("Original MeasureMultipleTimes + cumprod")
print("=" * 70)

obs_list = [
    {
        "wires": [0, 1],
        "observables": ["z", "z"]
    }
]

measure_old = MeasureMultipleTimes(obs_list)

x_old = measure_old(qdev)

print("Output of MeasureMultipleTimes:")
print(x_old)

print("\n")

product = torch.cumprod(
    x_old,
    dim=-1
)[:, -1]

print("After cumprod:")
print(product)

print("\n")

print("Interpretation:")
print("<Z0><Z1> =", product.item())

print("\n")


# ============================================================
# METHOD 2
# ============================================================

print("=" * 70)
print("METHOD 2")
print("Correct analytical joint expectation")
print("=" * 70)

joint_analytical = expval_joint_analytical(
    qdev,
    "ZZ"
)

print("expval_joint_analytical(qdev, 'ZZ'):")
print(joint_analytical)

print("\n")

print("Interpretation:")
print("<ZZ> =", joint_analytical.item())

print("\n")


# ============================================================
# METHOD 3
# ============================================================

print("=" * 70)
print("METHOD 3")
print("Correct sampling-based joint expectation")
print("=" * 70)

joint_sampling = expval_joint_sampling(
    qdev,
    "ZZ",
    n_shots=100000
)

print("expval_joint_sampling(qdev, 'ZZ'):")
print(joint_sampling)

print("\n")

print("Interpretation:")
print("<ZZ> ≈", joint_sampling.item())

print("\n")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(f"Method 1: <Z0><Z1>   = {product.item()}")
print(f"Method 2: <ZZ> exact = {joint_analytical.item()}")
print(f"Method 3: <ZZ> samp  = {joint_sampling.item()}")

print("\n")

print("Conclusion:")
print("Method 1 computes product of marginals.")
print("Methods 2 and 3 compute the true joint expectation.")