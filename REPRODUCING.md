# Reproducing the experiments

The [README](README.md) covers installation, the smoke test, and the main
Stage 6 → Stage 6.5 path. This page records the earlier commands and their
dependencies. The committed reports and lightweight results can be inspected
without training. Model checkpoints, downloaded data, oracle token tables, and
feature caches are deliberately excluded from Git.

All commands run from the repository root. Start with the README's hardware
check and smoke test. The published runs used a local RTX 3060 Laptop GPU with
6 GB VRAM. Full runs take much longer than the smoke test.

## Stages 1–4: one base checkpoint

Stage 1 trains the small Top-2 MoE:

```powershell
python training/train.py --config configs/baseline_small.yaml
```

The trainer prints a timestamped directory under `results/`. Its `last.pt` is
the checkpoint needed by the following stages. The saved Stage 2–4 YAML files
refer to the original run IDs. When replaying them, update only these path
fields to your newly produced outputs:

| Config | Path fields to update |
|---|---|
| `configs/oracle_analysis.yaml` | `checkpoint` → Stage 1 `last.pt` |
| `configs/stage3_calibrated_routing.yaml` | `checkpoint` → Stage 1 `last.pt`; `stage2_validation_results` → Stage 2 `token_results.parquet` |
| `configs/stage4_learned_compute_gate.yaml` | `checkpoint` and `stage2_validation_results` as above; `stage3_results` → Stage 3 result directory |

Run each evaluation after updating the paths it needs:

```powershell
python evaluation/run_oracle_analysis.py --config configs/oracle_analysis.yaml
python evaluation/stage3_benchmark.py --config configs/stage3_calibrated_routing.yaml
python evaluation/stage4_compute_gate.py --config configs/stage4_learned_compute_gate.yaml
```

Stage 2 creates the oracle token table used by later analyses. Stage 3 fits
post-hoc calibration and compares adaptive rules. Stage 4 trains the small
compute gate with the base MoE frozen. Keep the other saved config values if
you want the same protocol. The corresponding
[Stage 1](STAGE1_REPORT.md), [Stage 2](STAGE2_REPORT.md),
[Stage 3](STAGE3_REPORT.md), and [Stage 4](STAGE4_REPORT.md) reports describe
the splits, metrics, and limitations.

## Stage 5: independent small-model replication

Stage 5 trains five new base MoEs and three gate initializations per base model.
It does not need a Stage 1 checkpoint:

```powershell
python evaluation/stage5_replication.py --config configs/stage5_cross_checkpoint_replication.yaml
```

See [STAGE5_REPORT.md](STAGE5_REPORT.md) for the matched-compute comparison.

## Stage 6 and Stage 6.5

Stage 6 trains three larger base MoEs and evaluates the frozen-gate method.
Stage 6.5 then reads that run's checkpoints and cached final-validation
decisions; it does not train another model. Their commands are in the
[README](README.md#reproduction), with details in
[STAGE6_REPORT.md](STAGE6_REPORT.md) and
[STAGE6_5_REPORT.md](STAGE6_5_REPORT.md).
