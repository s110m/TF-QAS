import torchquantum as tq
import torch
import torch.nn.functional as F
from torchquantum.util.vqe_utils import parse_hamiltonian_file
from torchquantum.dataset import VQE
import random
import numpy as np
import argparse
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR



## I added the following: 

# class MeasureMultipleTimesSampling(tq.QuantumModule):
#     """
#     Similar to MeasureMultipleTimes, but computes
#     JOINT Pauli expectation values using sampling.

#     Each observable string is measured as a full tensor-product observable.

#     Example:
#     obs_list = [
#         {
#             "wires": [0, 1],
#             "observables": ["z", "z"]
#         },
#         {
#             "wires": [0, 1],
#             "observables": ["x", "x"]
#         }
#     ]

#     This computes:
#         <ZZ>
#         <XX>

#     using sampling-based estimation.
#     """

#     def __init__(
#             self,
#             obs_list,
#             n_shots=8192,
#             v_c_reg_mapping=None
#     ):
#         super().__init__()

#         self.obs_list = obs_list
#         self.n_shots = n_shots
#         self.v_c_reg_mapping = v_c_reg_mapping

#     def forward(self, qdev: tq.QuantumDevice):

#         res_all = []

#         for layer in self.obs_list:

#             # build full observable string
#             observable_full = ["I"] * qdev.n_wires

#             for wire, observable in zip(
#                     layer["wires"],
#                     layer["observables"]
#             ):
#                 observable_full[wire] = observable.upper()

#             observable_str = "".join(observable_full)

#             # compute JOINT expectation value
#             res = tq.expval_joint_sampling(
#                 qdev=qdev,
#                 observable=observable_str,
#                 n_shots=self.n_shots
#             )

#             # shape:
#             # [bsz]
#             # convert to [bsz, 1]
#             res = res.unsqueeze(-1)

#             if self.v_c_reg_mapping is not None:

#                 c2v_mapping = self.v_c_reg_mapping["c2v"]

#                 perm = []

#                 for k in range(res.shape[-1]):
#                     if k in c2v_mapping.keys():
#                         perm.append(c2v_mapping[k])

#                 res = res[:, perm]

#             res_all.append(res)

#         return torch.cat(res_all, dim=-1)

#     def set_v_c_reg_mapping(self, mapping):
#         self.v_c_reg_mapping = mapping


class MeasureMultipleTimesAnalytical(tq.QuantumModule):

    def __init__(self, obs_list):
        super().__init__()
        self.obs_list = obs_list

    def forward(self, qdev):

        res_all = []

        for layer in self.obs_list:

            observable_full = ["I"] * qdev.n_wires

            for wire, observable in zip(
                    layer["wires"],
                    layer["observables"]
            ):
                observable_full[wire] = observable.upper()

            observable_str = "".join(observable_full)

            res = tq.expval_joint_analytical(
                qdev,
                observable_str
            )

            res_all.append(res)

        return torch.stack(res_all, dim=-1)


class QVQEModel(tq.QuantumModule):
    def __init__(self, arch, hamil_info):
        super().__init__()
        self.arch = arch
        self.hamil_info = hamil_info
        self.n_wires = hamil_info['n_wires']
        self.n_blocks = arch['n_blocks']
        self.u3_layers = tq.QuantumModuleList()
        self.cu3_layers = tq.QuantumModuleList()
        for _ in range(self.n_blocks):
            self.u3_layers.append(tq.Op1QAllLayer(op=tq.U3,
                                                  n_wires=self.n_wires,
                                                  has_params=True,
                                                  trainable=True,
                                                  ))
            self.cu3_layers.append(tq.Op2QAllLayer(op=tq.CU3,
                                                   n_wires=self.n_wires,
                                                   has_params=True,
                                                   trainable=True,
                                                   circular=True
                                                   ))
        # self.measure = tq.MeasureMultipleTimes(
        #     obs_list=hamil_info['hamil_list'])

        self.measure = MeasureMultipleTimesAnalytical(
                    obs_list=hamil_info['hamil_list']
                )

    # def forward(self, q_device):
    #     q_device.reset_states(bsz=1)
    #     for k in range(self.n_blocks):
    #         self.u3_layers[k](q_device)
    #         self.cu3_layers[k](q_device)
    #     x = self.measure(q_device)

    #     hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
    #                                        self.hamil_info['hamil_list']],
    #                                       device=x.device).double()

    #     for k, hamil in enumerate(self.hamil_info['hamil_list']):
    #         for wire, observable in zip(hamil['wires'], hamil['observables']):
    #             if observable == 'i':
    #                 x[k][wire] = 1
    #         for wire in range(q_device.n_wires):
    #             if wire not in hamil['wires']:
    #                 x[k][wire] = 1

    #     x = torch.cumprod(x, dim=-1)[:, -1].double()
    #     x = torch.dot(x, hamil_coefficients)

    #     if x.dim() == 0:
    #         x = x.unsqueeze(0)

    #     return x

    def forward(self, q_device):

        q_device.reset_states(bsz=1)

        for k in range(self.n_blocks):
            self.u3_layers[k](q_device)
            self.cu3_layers[k](q_device)

        # shape: [num_hamiltonian_terms]
        x = self.measure(q_device).squeeze(0).double()

        hamil_coefficients = torch.tensor(
            [
                hamil['coefficient']
                for hamil in self.hamil_info['hamil_list']
            ],
            device=x.device
        ).double()

        # weighted sum of Pauli expectations
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x


def train(dataflow, q_device, model, device, optimizer):
    for _ in dataflow['train']:
        outputs = model(q_device)
        loss = outputs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Expectation of energy: {loss.item()}")


def valid_test(dataflow, q_device, split, model, device):
    with torch.no_grad():
        for _ in dataflow[split]:
            outputs = model(q_device)
    loss = outputs.mean()

    print(f"Expectation of energy: {loss}")


# Now, let's use the quantum circuits and ML strategy for approximate this value.
"""VQE (Variational Quantum Eigensolver) tries to avoid storing the full 2^n-dimensional wavefunction classically.

Instead:
Prepare a parameterized quantum state: ∣ψ(θ)⟩ using a quantum circuit.

Measure:
E(θ)=⟨ψ(θ)∣H∣ψ(θ)⟩ 
Optimize parameters θ with classical ML/optimization. (Note: a theorem states that E(θ)>= Ground-state energy. So, by 
minimization process we get closer and closer to Ground-state energy)""" 

class Args(object):
  def __init__(self):
    pass

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pdb', action='store_true', help='debug with pdb')
    # parser.add_argument('--n_blocks', type=int, default=2,
    #                     help='number of blocks, each contain one layer of '
    #                          'U3 gates and one layer of CU3 with '
    #                          'ring connections')
    # parser.add_argument('--steps_per_epoch', type=int, default=10,
    #                     help='number of training epochs')
    # parser.add_argument('--epochs', type=int, default=100,
    #                     help='number of training epochs')
    # parser.add_argument('--hamil_filename', type=str, default='./h2_new.txt',
    #                     help='number of training epochs')

    args = Args()
    args.n_blocks = 2
    args.steps_per_epoch=10
    args.epochs=10
    args.hamil_filename = 'h2_new.txt'

    # if args.pdb:
    #     import pdb
    #     pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = VQE(steps_per_epoch=args.steps_per_epoch)

    dataflow = dict()

    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
        else:
            sampler = torch.utils.data.SequentialSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=1,
            sampler=sampler,
            num_workers=1,
            pin_memory=True)

    hamil_info = parse_hamiltonian_file(args.hamil_filename)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = QVQEModel(arch={"n_blocks": args.n_blocks},
                       hamil_info=hamil_info)

    model.to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    q_device = tq.QuantumDevice(n_wires=hamil_info['n_wires'])
    q_device.reset_states(bsz=1)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        train(dataflow, q_device, model, device, optimizer)

        # valid
        valid_test(dataflow, q_device, 'valid', model, device)
        scheduler.step()

    # final valid
    valid_test(dataflow, q_device, 'valid', model, device)


if __name__ == '__main__':
    main()