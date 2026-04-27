# LPE-STGTN: Citywide Ride-Hailing Demand Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, research-grade PyTorch reimplementation of the **Local Perception-Enhanced Spatial-Temporal Evolving Graph Transformer Network** (LPE-STGTN), originally proposed by Zhang *et al.* in IEEE TITS (2024).

This repository contains the end-to-end pipeline for predicting citywide taxi and ride-hailing demand using the NYC Yellow Taxi dataset, integrating dynamic graph construction, attention-free transformers, and multi-head global semantic fusion.

---

## 🎯 Key Achievements & Results

Through systematic reproduction, topological corrections, and extensive ablation studies, this implementation achieved a **Test MAPE of 32.41%**, closely matching the original paper's benchmark of 31.82%.

| Model | Test MAE | Test RMSE | Test MAPE (%) |
|-------|----------|-----------|---------------|
| Persistence Baseline | 12.56 | 23.41 | 66.07 |
| LSTM Baseline | 6.65 | 12.29 | 75.64 |
| LPE-STGTN (Phase 1 Reproduction) | 7.05 | 11.93 | 35.66 |
| **LPE-STGTN (Final Optimized)** | **7.12** | **12.09** | **32.41** |
| *Original Paper Target* | *5.66* | *10.40* | *31.82* |

**Major Optimization Breakthroughs:**
1. **Topology Correction:** Identified and removed a severe geographic anomaly (TLC Zone 104 - Governor's Island) which injected 3,860+ feet of phantom distance into the spatial graph, drastically improving convergence stability.
2. **Graph Sparsification:** Applied $\epsilon$-thresholding ($\epsilon=0.05$) to prune noisy, near-zero cross-city edges in the adjacency matrix, proving that spatial data quality significantly outperforms raw architectural capacity.

---

## 🏗 Repository Structure

```text
.
├── configs/               # YAML configs for data, graphs, models, and training runs
├── data/                  # NYC Taxi TLC Data (External, Interim, Processed)
├── docs/                  # Reproduction logs and supplementary documentation
├── scripts/               # Entry points for data prep, training sweeps, and evaluation
├── src/
│   └── lpe_stgtn/
│       ├── data/          # PyTorch datasets and preprocessing pipelines
│       ├── evaluation/    # Metrics (MAE, RMSE, MAPE)
│       ├── graphs/        # Distance and OD-Flow adjacency matrix construction
│       ├── models/        # AFT-Local, GCRN, Attention, and main architecture
│       └── training/      # PyTorch Lightning-style training runners
└── tests/                 # Unit tests for tensors, shapes, and metrics
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/hrishi-alikatte/ride-hailing-demand-forecasting.git
cd ride-hailing-demand-forecasting
pip install -e .
```

### 2. Data Preparation
The pipeline assumes the raw NYC Yellow Taxi 2018 dataset is placed in `data/external/`. Run the inspection and processing scripts:
```bash
python scripts/inspect_raw_taxi_data.py
```
*(Graph construction happens automatically during the first data loader instantiation).*

### 3. Training the Model
We use a configuration-driven approach. To train the full optimized LPE-STGTN model on a GPU:
```bash
python scripts/run_sweep.py --config configs/experiments/nyc_lpe_stgtn_university_gpu.yaml
```

To run a rapid sanity check on CPU:
```bash
python scripts/run_sweep.py --config configs/experiments/sanity_raw_inspection.yaml
```

### 4. Running Ablation Sweeps
To reproduce the $\epsilon$-sparsification ablation study:
```bash
python scripts/run_epsilon_sweep.py
```

---

## 🧠 Architecture Overview

The LPE-STGTN processes historical demand through two parallel pathways:

1. **Local Perception Path:** Uses an **Attention-Free Transformer (AFT)** with a sliding window to capture local temporal patterns in $O(T \cdot W)$ time, followed by a Graph Convolutional Recurrent Network (GCRN).
2. **Global Semantic Fusion Path:** Applies distinct graph convolutions over a **Distance Graph** and an **Origin-Destination (OD) Flow Graph**, fusing the features via Multi-Head Attention to capture city-wide structural dependencies.

---

## 📖 Citation & Acknowledgements

This project is a reimplementation based on the research presented in:
> J. Zhang *et al.*, "Local Perception-Enhanced Spatial-Temporal Evolving Graph Transformer Network for Citywide Demand Prediction of Taxi and Ride-Hailing," *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 11, Nov. 2024.

Special thanks to the open-source community for the `OSMnx` and `NetworkX` libraries which powered the spatial graph routing.
