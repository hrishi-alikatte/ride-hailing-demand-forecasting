# Final Benchmark Comparison

Generated from 8 experiment reports.

## Paper Target

| MAE | RMSE | MAPE |
|-----|------|------|
| 5.66 | 10.4 | 31.82% |

## Results (Ranked by Test MAE)

| # | Experiment | h_dim | Dropout | LR | Batch | Test MAE | Test RMSE | Test MAPE | Epochs | Time | vs Paper |
|---|-----------|-------|---------|-----|-------|----------|-----------|-----------|--------|------|----------|
| 1 | nyc_lpe_stgtn_default_gpu_paperlike | ? | ? | ? | ? | 6.1523 | 11.2014 | 40771.18% | 95 | — | +8.7% |
| 2 | sweep_run_h128_d015 | 128 | 0.15 | 0.0005 | 32 | 7.1156 | 12.0873 | 32.41% | 64 | 24m 0s | +25.7% |
| 3 | sweep_run_baseline | 64 | 0.1 | 0.001 | 16 | 7.1460 | 12.1498 | 32.61% | 58 | 13m 20s | +26.3% |
| 4 | sweep_run_h128_d02 | 128 | 0.2 | 0.0005 | 32 | 7.1479 | 12.1242 | 32.93% | 58 | 21m 46s | +26.3% |
| 5 | nyc_lpe_stgtn_university_gpu_run | ? | ? | ? | ? | 7.1545 | 12.1498 | 0.33% | 93 | — | +26.4% |
| 6 | sweep_run_h64_d02 | 64 | 0.2 | 0.001 | 16 | 7.1607 | 12.1905 | 33.19% | 58 | 13m 3s | +26.5% |
| 7 | sweep_run_h128_d01 | 128 | 0.1 | 0.001 | 16 | 7.1762 | 12.1339 | 32.61% | 39 | 15m 26s | +26.8% |
| 8 | nyc_lpe_stgtn_default_quickcpu | ? | ? | ? | ? | 11.5580 | 21.5852 | 200496.12% | 2 | — | +104.2% |

## Best Run

**nyc_lpe_stgtn_default_gpu_paperlike** — Test MAE 6.1523, RMSE 11.2014, MAPE 40771.18%

Configuration: hidden_dim=?, dropout=?, lr=?, batch_size=?
