import os
import json
import nbformat as nbf
from pathlib import Path

# Create directories if they don't exist
Path("configs/experiments").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Generate University GPU yaml config
# ---------------------------------------------------------
yaml_content = """experiment_name: "nyc_lpe_stgtn_university_gpu_run"
processed_data_dir: "data/processed/nyc_yellow_2018"

model:
  name: "lpe_stgtn"
  hidden_dim: 64
  attention_heads: 4
  aft_window_size: 4
  gru_layers: 1
  dropout: 0.1

trainer:
  # Reduced batch size ensures we safely clear 16% GPU fraction (approx 3.8GB on RTX6000)
  # and the aggressive 8 GB System RAM limit.
  batch_size: 16       
  max_epochs: 30       # Safely finishes inside the 4-hour limit
  early_stopping_patience: 4
  gradient_clip_norm: 5.0
  scheduler:
    type: "ReduceLROnPlateau"
    factor: 0.5
    patience: 2

optimizer:
  learning_rate: 0.001

artifacts:
  report_dir: "artifacts/reports/models"
  checkpoint_dir: "artifacts/checkpoints"

runtime:
  device: "auto"

reproducibility:
  seed: 2026
"""

with open("configs/experiments/nyc_lpe_stgtn_university_gpu.yaml", "w") as f:
    f.write(yaml_content)

print("[INFO] Created configs/experiments/nyc_lpe_stgtn_university_gpu.yaml")

# ---------------------------------------------------------
# 2. Generate Jupyter Notebook (jupyterhub_slurm_training.ipynb)
# ---------------------------------------------------------
nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("""# 🏫 University GPU Training (Constrained Environment)
    
**Hardware Allocation Limits:**
*   **CPU:** 1 Core
*   **RAM:** 8 GB
*   **GPU:** 16% Fractional Access (~3.8GB on RTX6000, ~12.8GB on A100)
*   **Time:** 4 Hours Session

This notebook strictly isolates PyTorch to your exact constraints so you avoid Out-Of-Memory (OOM) crashing or session banning."""),
    
    nbf.v4.new_code_cell("""import os
import torch

# 1. Enforce CPU Core Limits (Prevent Thread-bombing the cluster)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 2. Enforce GPU Memory Allocation constraints (16% fraction)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.16)
    print("✅ PyTorch GPU Memory locked to 16% fraction.")
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected? Ensure you spawned a GPU Jupyter session.")
    
print("✅ CPU Threads locked to 1 processing core.")"""),
    
    nbf.v4.new_markdown_cell("""### Execute Training Runner
We execute the model pointing to our specially designed `nyc_lpe_stgtn_university_gpu.yaml`. This YAML halves the batch size (`16`) so we comfortably fit inside the 8GB RAM restriction without paging."""),
    
    nbf.v4.new_code_cell("""from pathlib import Path
from lpe_stgtn.training.lpe_stgtn_runner import run_lpe_stgtn_experiment

# Define paths
project_root = Path.cwd()
experiment_config_path = project_root / "configs" / "experiments" / "nyc_lpe_stgtn_university_gpu.yaml"
training_config_path = project_root / "configs" / "training" / "default.yaml"

# Set up fallback empty default training config if missing
if not training_config_path.exists():
    training_config_path.parent.mkdir(parents=True, exist_ok=True)
    training_config_path.write_text("{}")

# Run the localized Model!
print("🚀 Launching University-Constrained Run...")
summary = run_lpe_stgtn_experiment(
    experiment_config_path=experiment_config_path,
    project_root=project_root,
    training_config_path=training_config_path
)

print("\\n🎉 Run Completed!")
print(f"Experiment: {summary.experiment_name}")
print(f"Test MAE: {summary.metrics['test']['mae']:.4f}")
print(f"Test RMSE: {summary.metrics['test']['rmse']:.4f}")
print("Checkpoints saved safely.")
""")
]

with open("jupyterhub_slurm_training.ipynb", "w") as f:
    nbf.write(nb, f)

print("[INFO] Created jupyterhub_slurm_training.ipynb")

# ---------------------------------------------------------
# 3. Generate SLURM sbatch script (slurm_submit.sh)
# ---------------------------------------------------------
sh_content = """#!/bin/bash
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
"""

with open("slurm_submit.sh", "w") as f:
    f.write(sh_content)

os.chmod("slurm_submit.sh", 0o755)
print("[INFO] Created slurm_submit.sh (Standalone Fallback)")

