"""Compile all sweep results into a ranked comparison table."""

from __future__ import annotations

import json
from pathlib import Path


PAPER_METRICS = {"mae": 5.66, "rmse": 10.40, "mape": 31.82}


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    sweep_dir = project_root / "artifacts" / "reports" / "sweep"
    models_dir = project_root / "artifacts" / "reports" / "models"

    reports: list[dict] = []
    for report_dir in [sweep_dir, models_dir]:
        if not report_dir.exists():
            continue
        for path in sorted(report_dir.glob("*.json")):
            if path.name == "sweep_summary.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "metrics" not in data or "test" not in data.get("metrics", {}):
                    continue
                test = data["metrics"]["test"]
                val = data["metrics"].get("validation", {})
                hp = data.get("hyperparameters", {})
                history = data.get("training_history", [])
                reports.append({
                    "name": data.get("experiment_name", path.stem),
                    "source": report_dir.name,
                    "hidden_dim": hp.get("hidden_dim", "?"),
                    "dropout": hp.get("dropout", "?"),
                    "lr": hp.get("learning_rate", "?"),
                    "batch_size": hp.get("batch_size", "?"),
                    "test_mae": test.get("mae"),
                    "test_rmse": test.get("rmse"),
                    "test_mape": test.get("mape"),
                    "val_mae": val.get("mae"),
                    "val_rmse": val.get("rmse"),
                    "val_mape": val.get("mape"),
                    "epochs": len(history),
                    "time": data.get("wall_clock_human", "—"),
                    "path": str(path),
                })
            except Exception as exc:
                print(f"⚠ Skipping {path.name}: {exc}")

    if not reports:
        print("No reports found.")
        return 1

    # Sort by test MAE
    reports.sort(key=lambda r: r["test_mae"] if r["test_mae"] is not None else 999)

    # Print terminal table
    print(f"\n{'=' * 120}")
    print("FINAL RESULTS — Ranked by Test MAE")
    print(f"{'=' * 120}")
    print(f"{'#':<4} {'Experiment':<40} {'h':<5} {'d':<6} {'lr':<8} {'bs':<4} "
          f"{'MAE':<8} {'RMSE':<8} {'MAPE%':<8} {'Ep':<4} {'Time':<8} {'vs Paper':<10}")
    print("-" * 120)

    for i, r in enumerate(reports, 1):
        mae_str = f"{r['test_mae']:.4f}" if r["test_mae"] is not None else "—"
        rmse_str = f"{r['test_rmse']:.4f}" if r["test_rmse"] is not None else "—"
        mape_str = f"{r['test_mape']:.2f}" if r["test_mape"] is not None else "—"

        # Compare to paper
        if r["test_mae"] is not None and r["test_mae"] <= PAPER_METRICS["mae"]:
            vs_paper = "✅ BEAT"
        elif r["test_mae"] is not None:
            gap = ((r["test_mae"] - PAPER_METRICS["mae"]) / PAPER_METRICS["mae"]) * 100
            vs_paper = f"+{gap:.1f}%"
        else:
            vs_paper = "—"

        print(f"{i:<4} {r['name']:<40} {str(r['hidden_dim']):<5} {str(r['dropout']):<6} "
              f"{str(r['lr']):<8} {str(r['batch_size']):<4} "
              f"{mae_str:<8} {rmse_str:<8} {mape_str:<8} {r['epochs']:<4} {r['time']:<8} {vs_paper:<10}")

    print(f"\n{'Paper Target':<53} {'—':<5} {'—':<6} {'—':<8} {'—':<4} "
          f"{PAPER_METRICS['mae']:<8.4f} {PAPER_METRICS['rmse']:<8.4f} {PAPER_METRICS['mape']:<8.2f}")
    print(f"{'=' * 120}")

    # Generate markdown report
    md_path = project_root / "artifacts" / "reports" / "final_comparison.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Final Benchmark Comparison",
        "",
        f"Generated from {len(reports)} experiment reports.",
        "",
        "## Paper Target",
        "",
        f"| MAE | RMSE | MAPE |",
        f"|-----|------|------|",
        f"| {PAPER_METRICS['mae']} | {PAPER_METRICS['rmse']} | {PAPER_METRICS['mape']}% |",
        "",
        "## Results (Ranked by Test MAE)",
        "",
        "| # | Experiment | h_dim | Dropout | LR | Batch | Test MAE | Test RMSE | Test MAPE | Epochs | Time | vs Paper |",
        "|---|-----------|-------|---------|-----|-------|----------|-----------|-----------|--------|------|----------|",
    ]

    for i, r in enumerate(reports, 1):
        mae_str = f"{r['test_mae']:.4f}" if r["test_mae"] is not None else "—"
        rmse_str = f"{r['test_rmse']:.4f}" if r["test_rmse"] is not None else "—"
        mape_str = f"{r['test_mape']:.2f}%" if r["test_mape"] is not None else "—"

        if r["test_mae"] is not None and r["test_mae"] <= PAPER_METRICS["mae"]:
            vs_paper = "✅ BEAT"
        elif r["test_mae"] is not None:
            gap = ((r["test_mae"] - PAPER_METRICS["mae"]) / PAPER_METRICS["mae"]) * 100
            vs_paper = f"+{gap:.1f}%"
        else:
            vs_paper = "—"

        lines.append(
            f"| {i} | {r['name']} | {r['hidden_dim']} | {r['dropout']} | "
            f"{r['lr']} | {r['batch_size']} | {mae_str} | {rmse_str} | {mape_str} | "
            f"{r['epochs']} | {r['time']} | {vs_paper} |"
        )

    lines.extend([
        "",
        "## Best Run",
        "",
    ])

    best = reports[0]
    lines.append(f"**{best['name']}** — Test MAE {best['test_mae']:.4f}, "
                 f"RMSE {best['test_rmse']:.4f}, MAPE {best['test_mape']:.2f}%")
    lines.append("")
    lines.append(f"Configuration: hidden_dim={best['hidden_dim']}, dropout={best['dropout']}, "
                 f"lr={best['lr']}, batch_size={best['batch_size']}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMarkdown report saved to: {md_path}")

    # Save JSON
    json_path = project_root / "artifacts" / "reports" / "final_comparison.json"
    json_path.write_text(json.dumps({
        "paper_metrics": PAPER_METRICS,
        "results": reports,
    }, indent=2), encoding="utf-8")
    print(f"JSON report saved to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
