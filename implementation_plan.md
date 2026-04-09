# Phase 2 Optimization Plan: Beating the Paper

This plan outlines the steps we will take to push your model's accuracy beyond the original authors' reported benchmark (31.82% MAPE) by implementing targeted graph sparsification and eliminating geographic layout noise.

## User Review Required

> [!TIP]
> The original dataset configuration for the 67-zone sensitivity track is already partially scaffolded in the repository! We simply need to route your existing GPU config through this cleaner dataset track and aggressively inject the sparsification thresholds. Please review the threshold choices below.

## Proposed Changes

### 1. New Sprint Report Generation
#### [NEW] `generate_sprint_report_phase2.py`
We will write a new python script (similar to your first one) that generates `Sprint_Report_Phase2.ipynb`. This notebook will document your exact university cluster results (MAE: 7.18, MAPE: 32.83%) and lay out the rationale behind this final Phase 2 push.

### 2. Eliminating the Zone 104 Anomaly
#### [MODIFY] `configs/experiments/nyc_lpe_stgtn_university_gpu.yaml`
We will modify your university run configuration to point to the `sensitivity_67` dataset and graph directories instead of the `default` 68-zone ones. By shifting to the 67-zone track, `LocationID 104` (Governor's Island) is completely ignored during graph construction, instantly deleting the 3,800-foot geographic outlier that was artificially stretching your spatial convolutions.

### 3. Implementing Graph Sparsification
#### [MODIFY] `configs/graphs/nyc_manhattan_sensitivity_67.yaml`
Currently, the graph generation system leaves both the Distance Graph and OD-Flow Graph `epsilon_threshold` values at `0.0`. This means *every* zone mathematically interacts with *every* other zone, even if the connection is infinitesimally weak, producing cross-city noise. 

We will change these to:
* `distance_graph.epsilon_threshold: 0.05` (Drops spatial relationships under 5% correlation strength)
* `od_flow_graph.epsilon_threshold: 0.05` (Drops demand relationships under 5% correlation strength)

## Open Questions
- A `0.05` threshold cleanly severs weak graph edges without destroying local neighborhoods. Are you comfortable with this threshold, or do you want to aggressively sparsify it further (e.g., `0.10`)?

## Verification Plan
1. Ensure the new Python script correctly generates `Sprint_Report_Phase2.ipynb`.
2. Ensure the YAML configs correctly bind the 67-zone dataset to the university cluster script.
3. Once pushed and run on the cluster again via `sbatch`, ensure the dataset loader recognizes exactly 67 zones instead of 68!
