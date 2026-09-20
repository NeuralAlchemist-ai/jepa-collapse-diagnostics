# Distribution Shift Evaluation — Result Note

## Protocol

- Protocol: `distribution_shift_v1`
- Frozen protocol commit: `0d78c616cfb644e66178be5984254a346638f5b7`
- Reproduction command:
  `python run_experiment.py --protocol distribution_shift_v1`
- Training condition: `standard_unshifted`
- Seeds: `42, 100, 2026, 3141, 404`

## Primary Evaluation

The primary evaluation compares the accuracy degradation between the
standard and shifted test domains.

The frozen metric is:

`accuracy_drop = standard_accuracy - shifted_accuracy`

The paired per-seed comparison is:

`D_seed = accuracy_drop_AE - accuracy_drop_JEPA`

## Aggregate Results

| Model | Standard Accuracy | Shifted Accuracy | Accuracy Drop |
|---|---:|---:|---:|
| AutoEncoder | 0.6312 | 0.4312 | 0.2000 |
| JEPA | 0.3750 | 0.2815 | 0.0935 |

Mean paired difference:

`D_seed_mean = 0.1065`

Equivalent to:

`10.65 percentage points`

Standard deviation across the five paired differences:

`0.02157`

95% paired t-based confidence interval:

`[0.079718, 0.133282]`

## Frozen Decision Rule

The frozen decision rule requires:

1. Mean JEPA robustness advantage > `0.05`
2. 95% confidence interval for the paired difference excludes zero

Observed:

- Mean advantage: `0.1065` → threshold met
- 95% CI: `[0.079718, 0.133282]` → excludes zero

**Decision rule: MET**

## Per-Seed Results

| Seed | D_seed |
|---:|---:|
| 42 | 0.1100 |
| 100 | 0.0865 |
| 2026 | 0.1100 |
| 3141 | 0.0870 |
| 404 | 0.1390 |

## Retained Artifacts

Per-seed metrics and aggregate results are retained under:

`results/distribution_shift_v1_20260920_094617_610280Z/`

Artifacts include:

- `seed_42/metrics.json`
- `seed_100/metrics.json`
- `seed_2026/metrics.json`
- `seed_3141/metrics.json`
- `seed_404/metrics.json`
- `aggregate.json`
- `paired_summary.json`
- `translation_offsets.json`

## Execution Notes

The frozen protocol was executed without protocol changes after
outcomes were observed.

The matched-augmentation condition is secondary/descriptive and is not
used for the primary robustness decision.

No seeds were excluded and no hyperparameters, metrics, thresholds, or
analysis procedures were changed after observing results.