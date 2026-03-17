# Training-Free Quantum Architecture Search via Expressibility

Official implementation of the paper:

**"Training-Free Quantum Architecture Search Under Realistic Noise via Expressibility-Guided Evolution"**
Published in *Entropy (MDPI), 2026* https://www.mdpi.com/1099-4300/28/3/330

---

## Overview

Designing robust parameterized quantum circuits (PQCs) in the NISQ era is challenging due to:

* hardware noise
* expensive SuperCircuit training
* device-specific evaluation

This work proposes a **training-free and device-agnostic quantum architecture search (QAS)** framework based on **expressibility**.

---

## Key Idea

Instead of:

❌ Training large SuperCircuits + noisy evaluation

We use:

✅ **KL-based noise-free expressibility** as a structural proxy

This enables:

* no training during search
* no noise simulation during search
* reusable architectures across quantum devices

---

##  Framework Comparison

![QAS Comparison](images/fig_qas_comparison.png)

**Figure:** Comparison between:

* (a) SuperCircuit-based QAS (e.g., QuantumNAS)
* (b) Proposed expressibility-guided training-free QAS

 Our method removes:

* SuperCircuit training
* repeated noisy evaluation during search

---

##  Installation

Create environment and install dependencies:

```bash
pip install -r requirements.txt
```
---

##  Usage

### 1️⃣ Train SuperCircuit (Baseline)

```bash
python train_supercircuit.py
```
---

### 2️⃣ Expressibility-Guided Search (Ours)
```bash
python expr_search_mnist.py 
```
Fashion-MNIST:
```bash
python expr_search_fashion_mnist.py
```
👉 This implements:

* random search
* expressibility-guided evolutionary search (our method)

---

### 3️⃣ QuantumNAS-style Baseline
```bash
python quantumnas_style_search.py 
```
👉 Includes:

* SuperCircuit-based evaluation
* noisy performance estimation
 
---

## 🔧 Implementation Details

This project builds upon and utilizes components from:

* TorchQuantum: https://github.com/mit-han-lab/torchquantum

We thank the authors for providing an open-source quantum machine learning framework.

---

##  Key Result: Search Efficiency

The main contribution of this work is **drastically reducing the computational cost of quantum architecture search**.

| Method                      | SuperCircuit Training | Search / Ranking        | Final Training |
|---------------------------|----------------------|-------------------------|----------------|
| Random Search             | —                    | —                       | ~5 h           |
| QuantumNAS-style          | ~3 h                 | ~4 days 18 h            | ~5 h           |
| **Ours (Expressibility)** | —                    | **~2 h**                | ~5 h           |

👉 **Key insight:**
- Our method eliminates **SuperCircuit training**
- Avoids **expensive noisy evaluations during search**
- Reduces search time from **~5 days → ~2 hours**

---

##  Performance

| Method                      | Accuracy |
|---------------------------|----------|
| Random Search             | 0.62     |
| QuantumNAS-style          | **0.75** |
| **Ours (Expressibility)** | 0.71     |

👉 We achieve **competitive performance** while being **orders of magnitude more efficient**.

##  Advantages

* No SuperCircuit training
* No noisy evaluation during search
* Device-agnostic architectures
* Efficient and scalable
* Reduced computational cost

---

##  Citation

If you use this code, please cite:


@Article{e28030330,
AUTHOR = {Mousavi, Seyedali and Mousavi, Seyedhamidreza and Pettersson, Paul and Daneshtalab, Masoud},
TITLE = {Training-Free Quantum Architecture Search Under Realistic Noise via Expressibility-Guided Evolution},
JOURNAL = {Entropy},
VOLUME = {28},
YEAR = {2026},
NUMBER = {3},
ARTICLE-NUMBER = {330},
URL = {https://www.mdpi.com/1099-4300/28/3/330},
ISSN = {1099-4300},
DOI = {10.3390/e28030330}
}

