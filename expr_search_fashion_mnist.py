import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
import torchquantum as tq
import tqdm
import random
import copy

from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchquantum.dataset import MNIST
from expressibility_both_case import compute_expressibility_without_noise
from torchquantum.plugin.qiskit.qiskit_processor import QiskitProcessor

from datetime import datetime
print(f"Using torchquantum from: {os.path.dirname(tq.__file__)}")

from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.algorithm.quantumnas.super_layers import super_layer_name_dict
import torch.nn.functional as F
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)

from qiskit_ibm_runtime.fake_provider import FakeYorktownV2, FakeTorino, FakeManilaV2
from qiskit_aer.noise import NoiseModel

# **Load configs**
# The config file describes everything about the model structure.

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="default_fashion_mnist.yaml")
args = parser.parse_args()

configs.load(args.config)

if configs.debug.set_seed:
    torch.manual_seed(configs.debug.seed)
    np.random.seed(configs.debug.seed)

    torch.cuda.manual_seed_all(configs.debug.seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


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
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs, parallel=True).to(  # type: ignore
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





print(f"Using torchquantum from: {os.path.dirname(tq.__file__)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if isinstance(configs.optimizer.lr, str):
    configs.optimizer.lr = eval(configs.optimizer.lr)
dataset = MNIST(
    root='./fashion_mnist_data',
    fashion=True,
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest=[0, 1, 2, 3],
    n_test_samples=300,
    n_train_samples=5000,
    n_valid_samples=3000,
)

g = torch.Generator()
g.manual_seed(configs.debug.seed)
dataflow = dict()
for split in dataset:
    sampler = torch.utils.data.RandomSampler(dataset[split], generator=g)
    dataflow[split] = torch.utils.data.DataLoader(
        dataset[split],
        batch_size=configs.run.bsz,
        sampler=sampler,
        num_workers=configs.run.workers_per_gpu,
        pin_memory=True)



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



def evaluate_gene_model(model, gene=None, use_qiskit=False):
    if gene is not None:
        model.set_sample_arch(gene)
    with torch.no_grad():
        target_all = []
        output_all = []
        for feed_dict in tqdm.tqdm(dataflow['test']):
            if configs.run.device == 'gpu':
                inputs = feed_dict[configs.dataset.input_name].cuda(non_blocking=True)
                targets = feed_dict[configs.dataset.target_name].cuda(non_blocking=True)
            else:
                inputs = feed_dict[configs.dataset.input_name]
                targets = feed_dict[configs.dataset.target_name]

            outputs = model(inputs, use_qiskit=use_qiskit)
            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)
    accuracy, loss = log_acc(output_all, target_all)

    return accuracy, loss



def build_model_for_gene(
    gene,
    noisy_backend,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
   
    model = SuperQFCModel0(
        copy.deepcopy(configs.model.arch)
    )

    noise_model = NoiseModel.from_backend(noisy_backend)

    processor_real_qc = QiskitProcessor(
    use_real_qc=False,                  # simulate, not run on real QC
    noise_model=noise_model,    # IBM backend to pull noise from
    ibm_quantum_token="---", # your IBM Quantum API token
)
    model.set_qiskit_processor(processor_real_qc)

    model.to(device)

    model.set_sample_arch(gene)

    return model





def train_subcircuit_simple(
    gene,
    noisy_backend,
    n_epochs=5,
    lr=1e-3,
    weight_decay=1e-4,
    use_qiskit_eval=True,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=False,
):
    """
    Train a fixed subcircuit (gene) and return the best accuracy during training.
    """
    # -------------------------
    # Build model
    # -------------------------
    model = build_model_for_gene(gene, noisy_backend, device=device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs)

    criterion = torch.nn.NLLLoss()

    best_acc = -1.0
    eval_interval = 5  # or even 10

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        # eval_interval = 5  # or even 10

        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].cuda(non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataflow['train'])

        # -------------------------
        # Evaluation
        # -------------------------
        if epoch % eval_interval == 0 or epoch == n_epochs-1:
            acc, _ = evaluate_gene_model(
                model,
                gene=gene,
                use_qiskit=use_qiskit_eval,
            )
            if acc > best_acc:
                best_acc = acc

        scheduler.step()

        if verbose:
            print(
                f"[Epoch {epoch:02d}] "
                f"Loss={epoch_loss:.4f}, "
                f"Acc={acc:.4f}, "
                f"Best={best_acc:.4f}"
            )

    # # -------------------------
    # # Restore best model (optional)
    # # -------------------------
    # model.load_state_dict(best_state)

    return best_acc




def accuracy_predictor(gene, noisy_backend=None, use_qiskit=True):
    best_acc = train_subcircuit_simple(
        gene=gene,
        noisy_backend=noisy_backend,
        n_epochs=configs.run.n_epochs,
        lr=configs.optimizer.lr,
        use_qiskit_eval=use_qiskit,
        verbose=False,
    )
    return best_acc



class RandomAndEvolutionarySearcher:
    def __init__(self,
                 gene_choice,
                 accuracy_predictor,
                 expr_predictor,
                 configs):
        self.gene_choice = gene_choice
        self.gene_len = len(self.gene_choice)
        self.accuracy_predictor = accuracy_predictor
        self.expr_predictor = expr_predictor
        self.n_iterations = configs.es.n_iterations
        self.parent_size = configs.es.parent_size
        self.mutation_size = configs.es.mutation_size
        self.mutation_prob = configs.es.mutation_prob
        self.crossover_size = configs.es.crossover_size

    def random_sample(self, sample_num):
        # randomly sample genes
        population = []
        i = 0
        while i < sample_num:
            samp_gene = []
            for k in range(self.gene_len):
                samp_gene.append(random.choices(self.gene_choice[k])[0])
            population.append(samp_gene)
            i += 1
        return population

    def ask(self):
        """return the solutions"""
        return self.population

    def select_and_transform(self, scores):
        """perform evo search according to the scores"""
        
        # sort the index according to the scores (descending order)
        sorted_idx = (-np.array(scores)).argsort()[:self.parent_size]

        # hint: update self.best_solution and self.best_score
        self.best_solution = self.population[sorted_idx[0]]
        self.best_score = scores[sorted_idx[0]]

        self.best_solutions = [self.population[i] for i in sorted_idx]
        self.best_scores = [scores[i] for i in sorted_idx]

        parents = [self.population[i] for i in sorted_idx]

        # mutation
        mutate_population = []
        k = 0
        while k < self.mutation_size:
            mutated_gene = self.mutate(random.choices(parents)[0])
            mutate_population.append(mutated_gene)
            k += 1

        # crossover
        crossover_population = []
        k = 0
        while k < self.crossover_size:
            crossovered_gene = self.crossover(random.sample(parents, 2))
            crossover_population.append(crossovered_gene)
            k += 1

        self.population = parents + mutate_population + crossover_population

    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])
        return crossovered_gene

    def mutate(self, gene):
        mutated_gene = []
        for i in range(self.gene_len):        
            # use np.random.uniform() to decide whether to mutate position i
            # mutate ith position of gene with self.mutation_prob as mutation probability
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(self.gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])
        return mutated_gene  
    
    def search_best_expr(self, n_experiments):
        best_gene_list = []

        for i in range(n_experiments):
            # NO seed reset here

            self.population = self.random_sample(
                self.parent_size + self.mutation_size + self.crossover_size
            )

            for _ in range(self.n_iterations):
                expr_scores = [
                    -self.expr_predictor(gene, n_qubits=4)
                    for gene in self.population
                ]
                self.select_and_transform(expr_scores)

            best_gene_list.append(self.best_solution)

        return best_gene_list

         
    def run_search(self, noisy_backends, n_experiments):
        randome_gene_list = self.random_sample(n_experiments) 
        logger.info(f'random gene list is { randome_gene_list }')
        best_gene_list = self.search_best_expr(n_experiments)
        # best_gene_list = [[3, 1, 4, 4, 2, 4, 3]]
        logger.info(f'expressive gene list is {best_gene_list}')
        accs_random_all = []
        accs_evo_all = []
        for backend in noisy_backends:
            accs_random = []
            accs_evo = []
            for gene in randome_gene_list:
                acc_random = self.accuracy_predictor(
                    gene=gene, noisy_backend = backend, use_qiskit=True
                )
                accs_random.append(acc_random)
            accs_random = np.array(accs_random)
            accs_random_all.append(accs_random)
            logger.info(f'random accuracies for backend {backend.name } are: { accs_random} with mean { accs_random.mean()}')

            for gene in best_gene_list:
                acc_evo = self.accuracy_predictor(gene=gene, noisy_backend=backend, use_qiskit=True)
                accs_evo.append(acc_evo)
            accs_evo = np.array(accs_evo)
            accs_evo_all.append(accs_evo)
            logger.info(f'Expressibility guided accuracies for backend { backend.name } are: { accs_evo} with mean { accs_evo.mean()}')


        logger.info(f"Random accs: {accs_random_all}")
        logger.info(f"Evo accs: {accs_evo_all}")


        return {
            "random": accs_random_all,
            "evolutionary": accs_evo_all,
        }


if __name__ == "__main__":

    run_name = f"expr_search_fashion_mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = f"runs/{run_name}"
    set_run_dir(run_dir)

    logger.info(f"Run directory set to: {run_dir}")

    noisy_backends = [FakeYorktownV2(), FakeTorino(), FakeManilaV2()]

    gene_choises = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3]]
    # model = SuperQFCModel0(configs.model.arch)

    logger.info("========== EXPERIMENT METADATA ==========")
    logger.info(f"Seed: {configs.debug.seed}")
    logger.info(f"Epochs: {configs.run.n_epochs}")
    logger.info(f"Batch size: {configs.run.bsz}")
    logger.info(f"Gene choices: {gene_choises}")
    logger.info(f"Noisy backends: {[b.name for b in noisy_backends]}")
    logger.info("========================================")


    agent = RandomAndEvolutionarySearcher(gene_choises, accuracy_predictor, compute_expressibility_without_noise, configs)

    # get the accuracy and gene of the best subcircuit
    results = agent.run_search(noisy_backends, n_experiments=5)

    print("Random all accs:", results["random"])
    print("Evo all accs:", results["evolutionary"])

