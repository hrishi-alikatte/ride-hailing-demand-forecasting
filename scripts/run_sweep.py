"""Sweep runner — execute all experiment configs in configs/experiments/sweep/ sequentially."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    sweep_dir = project_root / "configs" / "experiments" / "sweep"
    training_config = project_root / "configs" / "training" / "default.yaml"

    configs = sorted(sweep_dir.glob("*.yaml"))
    if not configs:
        print(f"No YAML configs found in {sweep_dir}")
        return 1

    # Support --config flag for single runs
    single_config = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            single_config = Path(sys.argv[idx + 1])
            if not single_config.exists():
                print(f"Config not found: {single_config}")
                return 1
            configs = [single_config]

    # Support --dry-run flag
    dry_run = "--dry-run" in sys.argv

    print(f"{'=' * 60}")
    print(f"LPE-STGTN Hyperparameter Sweep")
    print(f"{'=' * 60}")
    print(f"Project root: {project_root}")
    print(f"Configs to run: {len(configs)}")
    for cfg in configs:
        print(f"  - {cfg.name}")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[DRY RUN] Validating configs...")
        from lpe_stgtn.config import load_yaml
        for cfg in configs:
            try:
                config = load_yaml(cfg)
                name = config.get("experiment_name", "UNNAMED")
                h = config.get("model", {}).get("hidden_dim", "?")
                d = config.get("model", {}).get("dropout", "?")
                lr = config.get("optimizer", {}).get("learning_rate", "?")
                bs = config.get("trainer", {}).get("batch_size", "?")
                print(f"  ✅ {cfg.name}: {name} (h={h}, d={d}, lr={lr}, bs={bs})")
            except Exception as exc:
                print(f"  ❌ {cfg.name}: {exc}")
        print("\n[DRY RUN] All configs validated.")
        return 0

    # Import here to avoid slow torch import during dry-run
    from lpe_stgtn.training.lpe_stgtn_runner import run_lpe_stgtn_experiment

    results: list[dict] = []
    sweep_start = time.time()

    for i, cfg in enumerate(configs, 1):
        report_file = project_root / "artifacts" / "reports" / "sweep" / f"sweep_{cfg.stem}.json"
        if report_file.exists():
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(configs)}] Skipping: {cfg.name} (already completed)")
            print(f"{'=' * 60}")
            try:
                import json
                with open(report_file) as f:
                    data = json.load(f)
                results.append({
                    "config": cfg.name,
                    "status": "success",
                    "mae": data["metrics"]["test"]["mae"],
                    "rmse": data["metrics"]["test"]["rmse"],
                    "mape": data["metrics"]["test"]["mape"],
                    "elapsed": 0.0,
                })
            except Exception:
                pass
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] Starting: {cfg.name}")
        print(f"{'=' * 60}")

        run_start = time.time()
        try:
            summary = run_lpe_stgtn_experiment(
                cfg,
                project_root=project_root,
                training_config_path=training_config,
            )
            elapsed = time.time() - run_start
            print(f"\n✅ {cfg.name} completed in {int(elapsed // 60)}m {int(elapsed % 60)}s")
            print(f"   Test MAE:  {summary.metrics['test']['mae']:.4f}")
            print(f"   Test RMSE: {summary.metrics['test']['rmse']:.4f}")
            print(f"   Test MAPE: {summary.metrics['test']['mape']:.2f}%")
            print(f"   Report: {summary.report_path}")
            results.append({
                "config": cfg.name,
                "status": "success",
                "mae": summary.metrics["test"]["mae"],
                "rmse": summary.metrics["test"]["rmse"],
                "mape": summary.metrics["test"]["mape"],
                "elapsed": round(elapsed, 1),
            })
        except Exception as exc:
            elapsed = time.time() - run_start
            print(f"\n❌ {cfg.name} FAILED after {int(elapsed // 60)}m {int(elapsed % 60)}s")
            print(f"   Error: {exc}")
            traceback.print_exc()
            results.append({
                "config": cfg.name,
                "status": "failed",
                "error": str(exc),
                "elapsed": round(elapsed, 1),
            })

    sweep_elapsed = time.time() - sweep_start
    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE — {int(sweep_elapsed // 60)}m {int(sweep_elapsed % 60)}s total")
    print(f"{'=' * 60}")

    # Print summary table
    print(f"\n{'Config':<30} {'Status':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'Time':<10}")
    print("-" * 80)
    for r in results:
        if r["status"] == "success":
            print(f"{r['config']:<30} {'✅':<10} {r['mae']:<10.4f} {r['rmse']:<10.4f} {r['mape']:<10.2f} {r['elapsed']:.0f}s")
        else:
            print(f"{r['config']:<30} {'❌':<10} {'—':<10} {'—':<10} {'—':<10} {r['elapsed']:.0f}s")

    # Save sweep summary
    import json
    summary_path = project_root / "artifacts" / "reports" / "sweep" / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSweep summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
