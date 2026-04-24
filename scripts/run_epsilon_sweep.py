"""Run the Epsilon Sparsification Sweep."""

import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent

    epsilons = ["00", "01", "10"]

    print("=" * 60)
    print("PHASE 1: Building Sparsified Graphs")
    print("=" * 60)

    for eps in epsilons:
        graph_cfg = f"configs/graphs/nyc_manhattan_sensitivity_67_eps_{eps}.yaml"
        print(f"\nBuilding graphs for epsilon {eps} using {graph_cfg}...")
        
        # Check if already built
        graph_dir = project_root / f"data/processed/nyc_yellow_taxi_2018_manhattan_pickups_15min_sensitivity_67/graphs_eps_{eps}"
        if (graph_dir / "distance_adjacency.npy").exists():
            print(f"✅ Graphs for epsilon {eps} already exist. Skipping build.")
            continue

        cmd = [
            sys.executable, "-m", "lpe_stgtn.cli", "build-graphs",
            "--config", graph_cfg
        ]
        result = subprocess.run(cmd, env={"PYTHONPATH": "src"})
        if result.returncode != 0:
            print(f"❌ Failed to build graphs for epsilon {eps}")
            return 1

    print("\n" + "=" * 60)
    print("PHASE 2: Training Epsilon Models")
    print("=" * 60)

    # Use the existing sweep runner but point it to the sweep_epsilon folder
    # We can just call it directly with the 3 configs
    
    for eps in epsilons:
        exp_cfg = f"configs/experiments/sweep_epsilon/run_eps_{eps}.yaml"
        print(f"\nTraining model for epsilon {eps} using {exp_cfg}...")
        
        # Use python -m to run it
        cmd = [
            sys.executable, "scripts/run_sweep.py",
            "--config", exp_cfg
        ]
        result = subprocess.run(cmd, env={"PYTHONPATH": "src"})
        if result.returncode != 0:
            print(f"❌ Training failed for epsilon {eps}")
            return 1

    print("\n✅ Epsilon Sparsification Sweep Complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
