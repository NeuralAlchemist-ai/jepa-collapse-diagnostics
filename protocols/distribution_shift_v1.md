# Distribution Shift Evaluation Protocol v1

## Status

**Protocol status:** Pre-outcome / awaiting review
**Experiment status:** Not yet executed

This protocol defines the next bounded JEPA evaluation after the frozen closure run. No result-bearing run should be executed until this protocol is reviewed and its implementation is frozen.

The previous closure experiment remains frozen evidence and is not modified by this protocol.

## 1. Research Question

Does the JEPA representation retain more useful information under a small, label-preserving spatial distribution shift than the matched autoencoder control?

The evaluation tests robustness to a bounded image translation rather than a change in digit identity.

## 2. Failure Mode

**Failure mode:** Controlled distribution shift.

Training and evaluation use MNIST digits. The shifted evaluation set is created by applying a small deterministic spatial translation to the same test images.

The shift is intended to change pixel location while preserving the underlying digit class.

Large rotations, transformations that can alter digit identity, and other ambiguous transformations are excluded from this protocol.

## 3. Models

The comparison uses the same two learned models as the frozen closure experiment:

* JEPA
* Matched autoencoder control

The model architectures, latent dimension, and parameter budgets are unchanged.

No architecture changes, hyperparameter tuning, or additional model variants are introduced for this experiment.

The collapsed control from the closure experiment is not used as a primary model in this distribution-shift comparison.

## 4. Dataset and Fixed Splits

Dataset:

* MNIST
* 10 classes
* Standard MNIST preprocessing

Fixed dataset budget:

* Training examples: **10,000**
* Test examples: **2,000**

The committed manifests are:

```text
data/splits/train_indices.json
data/splits/test_indices.json
```

They contain the first 10,000 MNIST training indices and first 2,000 MNIST test
indices. This is a deterministic fixed subset, not a newly randomized split.

Both models use exactly the same training indices.

Both models are evaluated on exactly the same 2,000 test indices.

The shifted test set is generated from those same test images; it is not sampled independently.

No examples may be added, removed, or replaced after the protocol is frozen.

## 5. Distribution Shift

### Standard condition

The standard test set uses the normal MNIST images after the fixed preprocessing pipeline.

### Shifted condition

The shifted test set applies a bounded translation independently to each test image.

Frozen transformation:

* transformation: 2D translation
* horizontal displacement: integer value in `[-2, +2]` pixels
* vertical displacement: integer value in `[-2, +2]` pixels
* interpolation: bilinear
* fill value: raw-space `0.0` (black)
* transformation generation: deterministic
* transformation seed: `12345`

The exact transformation implementation must be committed before the experiment is run.

The same generated shifted test set is used for both models.

### Excluded transformations

The following are excluded because they may change the semantic identity of an MNIST digit or introduce an ambiguous label change:

* 90° rotations
* arbitrary large rotations
* horizontal or vertical reflection
* transformations that substantially crop or remove the digit
* transformations outside the frozen translation bounds

No transformation may be added after seeing experimental results.

## 6. Training Conditions

The primary evaluation trains both models on the standard, unshifted training data.

Training configuration:

| Setting               | Value                    |
| --------------------- | ------------------------ |
| Training samples      | 10,000                   |
| Batch size            | 128                      |
| Training epochs       | 3                        |
| Optimizer             | Adam                     |
| Learning rate         | 1e-3                     |
| Weight decay          | 0                        |
| Scheduler             | None                     |
| Training augmentation | None                     |
| Seeds                 | 42, 100, 2026, 3141, 404 |

The same training budget and preprocessing are used for JEPA and the autoencoder.

No hyperparameter selection is performed using the shifted test set.

## 7. Matched Augmentation Control

A secondary matched condition evaluates whether any observed robustness difference is simply caused by exposure to the same nuisance transformation during training.

In this condition:

* JEPA and the autoencoder receive the same training examples.
* Both receive the same bounded translation augmentation policy.
* The same transformation bounds and generation procedure are used for both models.
* Training sample budget, epochs, batch size, optimizer, learning rate, and seeds remain unchanged.

The augmentation policy is frozen before any result-bearing run. The training
translation seed is exactly `12345 + model_seed`; this does not alter the test
shift, whose seed remains `12345`.

The primary conclusion must distinguish robustness observed without augmentation from robustness observed after matched augmentation.

## 8. Linear Probe

A linear probe is trained using the frozen representation produced by each learned model.

Probe configuration:

* classifier: single linear layer
* input: 64-dimensional latent representation
* output: 10 MNIST classes
* optimizer: Adam
* learning rate: 1e-3
* probe epochs: 2
* weight decay: 0
* scheduler: None

The probe is trained only using standard/unshifted training representations.
The same trained probe is then evaluated on both standard and shifted test
representations.

For each model and seed, the same trained probe is evaluated separately on:

1. the standard test set
2. the shifted test set

The probe must not be retrained on shifted test data.

## 9. Primary Metric

The primary metric is **relative robustness measured by linear-probe accuracy degradation**.

For each model and seed:

```text
accuracy_drop = accuracy_standard - accuracy_shifted
```

A smaller accuracy drop indicates less degradation under the defined shift.

For each seed, calculate the paired model difference:

```text
D_seed = accuracy_drop_AE - accuracy_drop_JEPA
```

Positive `D_seed` means the JEPA model experienced a smaller accuracy drop for that seed.

The final analysis reports:

* mean `D_seed`
* standard deviation of `D_seed`
* 95% paired confidence interval for `D_seed`

The confidence interval is a paired t-based 95% interval over the five
`D_seed` values, with the method frozen before execution.

The five seeds are treated as paired observations because JEPA and the autoencoder use the same seed and data split.

## 10. Frozen Decision Rule

Before inspecting any distribution-shift results, the following decision rule is fixed:

JEPA will be described as showing **greater robustness under this protocol** only if:

1. the mean paired difference satisfies

```text
mean(D_seed) > 0.05
```

and

2. the 95% paired confidence interval for `D_seed` excludes zero.

Otherwise, the result will be reported as not meeting the predefined robustness criterion.

This decision rule applies only to this bounded experiment and does not establish a general claim about JEPA representations.

## 11. Secondary Metrics

The following metrics are recorded as secondary representation diagnostics:

* centered effective rank
* feature variance
* average pairwise cosine similarity

These metrics are evaluated on the shifted test representations.

They are descriptive secondary evidence and do not override the primary decision rule.

## 12. Seeds and Reproducibility

The exact seeds are:

```text
42
100
2026
3141
404
```

Seeds must be applied consistently to:

* Python random state
* PyTorch random state
* CUDA random state where available
* model initialization
* training data-loader shuffling
* probe initialization
* deterministic transformation generation where applicable

Deterministic PyTorch settings used by the existing experiment should remain enabled.

Each seed must produce a separate result directory.

Existing result files must never be overwritten.

## 13. Parameter and Compute Matching

The existing JEPA and autoencoder architectures are retained without modification.

Before execution, the implementation must record:

* trainable parameter count for JEPA
* trainable parameter count for the autoencoder
* training epochs
* batch size
* optimizer
* learning rate
* number of training examples
* number of probe epochs

No architecture or compute-budget adjustment may be made after seeing results.

## 14. Artifacts to Retain

For every seed, retain a JSON artifact containing at minimum:

* seed
* model name
* parameter count
* standard test accuracy
* shifted test accuracy
* accuracy drop
* effective rank
* feature variance
* average pairwise cosine similarity
* frozen protocol identifier

The final experiment must additionally produce:

```text
aggregate.json
paired_summary.json
```

`aggregate.json` contains mean and standard deviation across the five seeds.

`paired_summary.json` contains the per-seed and aggregate JEPA-minus-control robustness comparison, including the confidence interval used by the decision rule.

The protocol SHA and exact reproduction command must also be recorded with the final results.

## 15. Pre-Outcome Freeze

Before running the experiment, the following must be committed:

```text
protocols/distribution_shift_v1.md
protocols/distribution_shift_v1.json
data/splits/train_indices.json
data/splits/test_indices.json
```

as well as the implementation changes required to reproduce this protocol.

No result JSON, aggregate metric, plot, or other outcome-bearing artifact may be committed before the protocol is reviewed and frozen.

The resulting commit SHA is the protocol version used for review.

If any outcome-affecting detail is changed after review, a new commit and protocol review are required before execution.

## 16. Reproduction

After the protocol has been approved and frozen, the experiment must be executable from a clean checkout using one documented command.

The exact command used for the final run will be recorded in the result note.

Example form:

```bash
python run_experiment.py --protocol distribution_shift
```

The final reproduction command must correspond exactly to the reviewed implementation.

## 17. Interpretation Limits

This experiment tests only the predefined bounded translation shift on the fixed MNIST split and budget.

A result satisfying the decision rule would support the narrower statement that JEPA showed lower linear-probe accuracy degradation than the matched autoencoder under this specific distribution shift.

It would not by itself establish:

* general semantic robustness,
* superiority across datasets,
* robustness to arbitrary transformations,
* superiority of JEPA architectures in general,
* improved performance on real-world distribution shifts.

Negative or inconclusive outcomes must be retained and reported rather than used to motivate post-hoc tuning.

## 18. Approval Gate

**Current state:** Awaiting protocol review.

No result-bearing execution should occur until the protocol, exact implementation, fixed indices, transformation procedure, metrics, and decision rule have been reviewed and frozen.
