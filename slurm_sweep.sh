#!/bin/bash
#SBATCH --job-name=lpe_sweep
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_sweep_%j.log

echo "============================================================"
echo "LPE-STGTN Final Hyperparameter Sweep"
echo "============================================================"
echo "Node: $HOSTNAME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "============================================================"

# Activate virtualenv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Create output directories
mkdir -p artifacts/reports/sweep
mkdir -p artifacts/checkpoints/sweep

# Set environment constraints for GPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=src:$PYTHONPATH

# Print GPU info
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB (using 16%)')
else:
    print('WARNING: No GPU detected')
"

# Run the full sweep
python scripts/run_sweep.py

echo ""
echo "============================================================"
echo "Sweep finished at $(date)"
echo "============================================================"

# Compile results
echo "Compiling results..."
PYTHONPATH=src python scripts/compile_results.py

echo "Done!"
