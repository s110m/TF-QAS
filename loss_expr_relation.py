#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-05T10:19:42.669Z
"""
# # Setup

import argparse
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd())))
import pdb
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torchquantum as tq
import tqdm 
import random

from torchpack.utils import io
# from torchpack import distributed as dist
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchquantum.dataset import MNIST
import torch.optim as optim
from expressibility_both_case import compute_expressibility_without_noise
from expressibility_both_case import compute_expressibility_noisy
from torchquantum.plugin.qiskit.qiskit_processor import QiskitProcessor

# from torchquantum.super_utils import get_named_sample_arch
from qiskit_aer.primitives import SamplerV2 as Sampler
print(f"Using torchquantum from: {os.path.dirname(tq.__file__)}")

from qiskit_ibm_runtime.fake_provider import FakeYorktownV2


# **Load configs**
# 
# The config file describes everything about the model structure.


config_str = '''model:
  arch:
    n_wires: 4
    encoder_op_list_name: 4x4_ryzxy
    n_blocks: 3
    n_layers_per_block: 2
    q_layer_name: u3cu3_s0
    down_sample_kernel_size: 6
    n_front_share_blocks: 1
    n_front_share_wires: 1
    n_front_share_ops: 1
  sampler:
    strategy:
      name: plain
  transpile_before_run: False
  load_op_list: False

dataset:
  name: mnist
  input_name: image
  target_name: digit

optimizer:
  name: adam
  lr: 5e-2
  weight_decay: 1e-4
  lambda_lr: 1e-2

run:
  n_epochs: 40
  bsz: 256
  workers_per_gpu: 1
  device: cpu

debug:
  pdb: False
  set_seed: True
  seed: 42

callbacks:
  - callback: 'InferenceRunner'
    split: 'valid'
    subcallbacks:
      - metrics: 'CategoricalAccuracy'
        name: 'acc/valid'
      - metrics: 'NLLError'
        name: 'loss/valid'
  - callback: 'InferenceRunner'
    split: 'test'
    subcallbacks:
      - metrics: 'CategoricalAccuracy'
        name: 'acc/test'
      - metrics: 'NLLError'
        name: 'loss/test'
  - callback: 'MaxSaver'
    name: 'acc/valid'
  - callback: 'Saver'
    max_to_keep: 10

qiskit:
  use_qiskit: False
  use_real_qc: False
  backend_name: null
  noise_model_name: null
  basis_gates_name: null
  n_shots: 8192
  initial_layout: null
  seed_transpiler: 42
  seed_simulator: 42
  optimization_level: 0
  est_success_rate: False
  max_jobs: 1


es:
  random_search: False
  population_size: 100
  parent_size: 20
  mutation_size: 40
  mutation_prob: 0.5
  crossover_size: 40
  n_iterations: 5
  est_success_rate: False
  score_mode: loss_succ
  gene_mask: null
  eval:
    use_noise_model: True
    use_real_qc: False
    bsz: qiskit_max
    n_test_samples: 150


prune:
  target_pruning_amount : 0.5
  init_pruning_amount : 0.1
  start_epoch : 0
  end_epoch : 30

'''
f = open("configs_relation.yml", "w")
f.write(config_str)
f.close()

configs.load('configs_relation.yml')
if configs.debug.set_seed:
    torch.manual_seed(configs.debug.seed)
    np.random.seed(configs.debug.seed)

    torch.cuda.manual_seed_all(configs.debug.seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.algorithm.quantumnas.super_layers import super_layer_name_dict
import torch.nn.functional as F
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)


class SuperQFCModel0(tq.QuantumModule):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        # self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict[arch['encoder_op_list_name']]
        )
        self.q_layer = super_layer_name_dict[arch['q_layer_name']](arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.sample_arch = None

    def set_sample_arch(self, sample_arch):
        self.sample_arch = sample_arch
        self.q_layer.set_sample_arch(sample_arch)

    def count_sample_params(self):
        return self.q_layer.count_sample_params()

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, record_op=True, device=x.device)
        # self.q_device.reset_states(bsz=bsz)

        if getattr(self.arch, 'down_sample_kernel_size', None) is not None:
            x = F.avg_pool2d(x, self.arch['down_sample_kernel_size'])

        x = x.view(bsz, -1)

        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for trainable quantum layers
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measurement
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)

            # assemble the encoder, trainable quantum layers, and measurement circuits
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )

            # call the qiskit processor to process the circuit
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs, parallel=False).to(  # type: ignore
                x.device
            )
            x = x0

        else:
            self.encoder(qdev, x)
            self.q_layer(qdev)
            x = self.measure(qdev)

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        if getattr(self.arch, 'output_len', None) is not None:
            x = x.reshape(bsz, -1, self.arch.output_len).sum(-1)

        if x.dim() > 2:
            x = x.squeeze()

        x = F.log_softmax(x, dim=1)
        return x

    @property
    def arch_space(self):
        space = []
        for layer in self.q_layer.super_layers_all:
            space.append(layer.arch_space)
        # for the number of sampled blocks
        space.append(list(range(self.q_layer.n_front_share_blocks,
                                self.q_layer.n_blocks + 1)))
        return space


# Load the model.


import torch.nn.functional as F
import torchquantum.device
import torchquantum.algorithm.quantumnas.super_layers
import torchquantum.operator
import torchquantum.measurement
print(f"Using torchquantum from: {os.path.dirname(tq.__file__)}")
device = torch.device('cuda')
if isinstance(configs.optimizer.lr, str):
    configs.optimizer.lr = eval(configs.optimizer.lr)
dataset = MNIST(
    root='./mnist_data',
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest=[0, 1, 2, 3],
    n_test_samples=300,
    n_train_samples=5000,
    n_valid_samples=3000,
)

dataflow = dict()
for split in dataset:
    sampler = torch.utils.data.RandomSampler(dataset[split])
    dataflow[split] = torch.utils.data.DataLoader(
        dataset[split],
        batch_size=configs.run.bsz,
        sampler=sampler,
        num_workers=0, #configs.run.workers_per_gpu,
        pin_memory=True)
model = SuperQFCModel0(configs.model.arch)
# fix import paths 
sys.modules['torchquantum.devices'] = torchquantum.device
sys.modules['torchquantum.super_layers'] = torchquantum.algorithm.quantumnas.super_layers
sys.modules['torchquantum.operators'] = torchquantum.operator
sys.modules['torchquantum.measure'] = torchquantum.measurement
state_dict = io.load('max-acc-valid.pt', map_location='cpu', weights_only=False)
model.load_state_dict(state_dict['model'], strict=False)
model.to(device)

# -----------------------------------------------------
# (A) Connect to IBM Quantum
# -----------------------------------------------------
# service = QiskitRuntimeService()
# backend = service.backend("ibm_fez")

# or fake backend
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeYorktownV2
# Load fake backend
fake_backend = FakeYorktownV2()
noise_model = NoiseModel.from_backend(fake_backend)

processor_real_qc = QiskitProcessor(
    use_real_qc=False,                  # simulate, not run on real QC
    noise_model=noise_model,    # IBM backend to pull noise from
    ibm_quantum_token="---", # your IBM Quantum API token
)

model.set_qiskit_processor(processor_real_qc)



def log_acc(output_all, target_all, k=1):
    _, indices = output_all.topk(k, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Loss: {loss}")
    return accuracy, loss



def evaluate_gene(gene=None, use_qiskit=True):
    if gene is not None:
        model.set_sample_arch(gene)
    with torch.no_grad():
        target_all = None
        output_all = None
        for feed_dict in tqdm.tqdm(dataflow['test']):
            if configs.run.device == 'gpu':
                # pdb.set_trace()
                inputs = feed_dict[configs.dataset.input_name].cuda(non_blocking=True)
                targets = feed_dict[configs.dataset.target_name].cuda(non_blocking=True)
            else:
                inputs = feed_dict[configs.dataset.input_name]
                targets = feed_dict[configs.dataset.target_name]
            outputs = model(inputs, use_qiskit=use_qiskit)
            if target_all is None:
                target_all = targets
                output_all = outputs
            else:
                target_all = torch.cat([target_all, targets], dim=0)
                output_all = torch.cat([output_all, outputs], dim=0)
        accuracy, loss = log_acc(output_all, target_all)
    return accuracy, loss



def evaluate_population(
    population_size=100,
    use_qiskit=True,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    gene_choice = model.arch_space
    gene_len = len(gene_choice)

    # -------------------------------
    # 1. Sample population
    # -------------------------------
    population = []
    for _ in range(population_size):
        gene = [random.choice(gene_choice[i]) for i in range(gene_len)]
        population.append(gene)

    # -------------------------------
    # 2. Evaluate population
    # -------------------------------
    results = []
    for gene in tqdm.tqdm(population, desc="Evaluating genes"):
        acc, loss = evaluate_gene(
            gene=gene,
            use_qiskit=use_qiskit
        )

        KL_HS, KL_Uhlmann = compute_expressibility_noisy(
        sample_arch=gene,
        n_qubits=model.n_wires,
        fake_backend=fake_backend)

        KL_no_noise = compute_expressibility_without_noise(gene, n_qubits=model.n_wires)

        results.append({
            "gene": gene,
            "accuracy": acc,
            "loss": loss,
            "KL_no_noise": KL_no_noise,
            "KL_HS": KL_HS,
            "KL_Uhlmann": KL_Uhlmann
        })

    return results




results = evaluate_population(population_size=100, use_qiskit=True)


from spearman_utils import export_all_loss_kl


# ======================================================
# Export CSVs for TikZ figures
# ======================================================

export_all_loss_kl(results, save_dir="files")

print("\n[INFO] CSV export finished. Ready for TikZ.")


# \begin{tikzpicture}
# \begin{axis}[
#     width=0.8\columnwidth,
#     grid=major,
#     grid style={dashed,gray!40},
#     xlabel=\textbf{Loss},
#     ylabel=\textbf{KL (HS)},
#     label style={font=\small},
#     tick label style={font=\small},
# ]
# \addplot[
#     blue,
#     only marks,
#     mark size=1.5pt
# ]
# table[x=loss,y=kl,col sep=comma] {files/loss_kl_hs.csv};
# \end{axis}
# \end{tikzpicture}



