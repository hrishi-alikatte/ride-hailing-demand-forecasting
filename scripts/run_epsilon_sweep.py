"""Run the Epsilon Sparsification Sweep."""

import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent

import shutil
import numpy as np
from lpe_stgtn.graphs.distance import distance_matrix_to_adjacency, resolve_distance_sigma
from lpe_stgtn.graphs.od_flow import od_counts_to_adjacency
from lpe_stgtn.graphs.normalize import normalize_adjacency_with_self_loop

def fast_build_graphs(eps_name: str, eps_val: float, project_root: Path):
    base_dir = project_root / "data/processed/nyc_yellow_taxi_2018_manhattan_pickups_15min_sensitivity_67/graphs"
    out_dir = project_root / f"data/processed/nyc_yellow_taxi_2018_manhattan_pickups_15min_sensitivity_67/graphs_eps_{eps_name}"
    
    if (out_dir / "distance_adjacency.npy").exists():
        print(f"✅ Graphs for epsilon {eps_name} already exist. Skipping build.")
        return True
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy un-thresholded base files
    for fname in ["zone_ids.npy", "zone_centroids.csv", "zone_road_network_snaps.csv", "metadata.json", "distance_matrix.npy", "od_flow_counts.npy"]:
        if (base_dir / fname).exists():
            shutil.copy2(base_dir / fname, out_dir / fname)
            
    # Re-calculate distance thresholding
    dist_mat = np.load(base_dir / "distance_matrix.npy")
    sigma = resolve_distance_sigma(dist_mat)
    dist_adj = distance_matrix_to_adjacency(dist_mat, sigma=sigma, epsilon_threshold=eps_val)
    dist_norm = normalize_adjacency_with_self_loop(dist_adj)
    np.save(out_dir / "distance_adjacency.npy", dist_adj.astype(np.float32))
    np.save(out_dir / "distance_adjacency_normalized.npy", dist_norm.astype(np.float32))
    
    # Re-calculate OD flow thresholding
    od_counts = np.load(base_dir / "od_flow_counts.npy")
    od_adj, _ = od_counts_to_adjacency(od_counts, epsilon_threshold=eps_val)
    od_norm = normalize_adjacency_with_self_loop(od_adj)
    np.save(out_dir / "od_flow_adjacency.npy", od_adj.astype(np.float32))
    np.save(out_dir / "od_flow_adjacency_normalized.npy", od_norm.astype(np.float32))
    return True

def main():
    project_root = Path(__file__).resolve().parent.parent

    epsilons = [("00", 0.0), ("01", 0.01), ("10", 0.10)]

    print("=" * 60)
    print("PHASE 1: Fast-Building Sparsified Graphs")
    print("=" * 60)

    for eps_name, eps_val in epsilons:
        print(f"\nApplying threshold epsilon={eps_val} to pre-computed base matrices...")
        success = fast_build_graphs(eps_name, eps_val, project_root)
        if not success:
            print(f"❌ Failed to build graphs for epsilon {eps_name}")
            return 1

    print("\n" + "=" * 60)
    print("PHASE 2: Training Epsilon Models")
    print("=" * 60)

    # Use the existing sweep runner but point it to the sweep_epsilon folder
    # We can just call it directly with the 3 configs
    
    for eps_name, _ in epsilons:
        exp_cfg = f"configs/experiments/sweep_epsilon/run_eps_{eps_name}.yaml"
        print(f"\nTraining model for epsilon {eps_name} using {exp_cfg}...")
        
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        
        # Use python -m to run it
        cmd = [
            sys.executable, "scripts/run_sweep.py",
            "--config", exp_cfg
        ]
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"❌ Training failed for epsilon {eps_name}")
            return 1

    print("\n✅ Epsilon Sparsification Sweep Complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
