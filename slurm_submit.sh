#!/bin/bash
#SBATCH --job-name=lpe_stgtn_gpu
#SBATCH --time=04:00:00          # 4-hour hard limit
#SBATCH --cpus-per-task=1        # 1 CPU Core allocated
#SBATCH --mem=8G                 # 8GB System RAM
#SBATCH --gres=gpu:1             # Requesting the fractional GPU
#SBATCH --output=slurm_training_%j.log

echo "🚀 Starting SLURM Run on Newton Cluster..."
echo "Node: $HOSTNAME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Starting Time: $(date)"

# Activate virtualenv if present (assuming .venv)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# We use python -c to inject constraints directly before importing torch/lpe_stgtn
python -c '
import os
import torch
from pathlib import Path
from lpe_stgtn.training.lpe_stgtn_runner import run_lpe_stgtn_experiment

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.16)

experiment_config = Path("configs/experiments/nyc_lpe_stgtn_university_gpu.yaml")
root = Path(".")
run_lpe_stgtn_experiment(experiment_config_path=experiment_config, project_root=root)
'
echo "✅ Finished on $(date)"
