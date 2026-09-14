# Result note

The experiment includes a deliberately negative control. `CollapsedControl`
returns one constant vector for every image.
That model cannot preserve digit information: a linear probe receives the same
input for every label and should therefore remain close to the majority-class
or chance baseline (about 10% on balanced MNIST).

Its diagnostics should expose the failure directly. Centered feature variance
should be zero, the centered representation has no non-zero singular values,
and average pairwise cosine similarity should be one because every output is
identical. This is why linear-probe accuracy alone is insufficient: a model can
produce a numerically stable-looking embedding while carrying no usable
sample-to-sample information.

The executable trial uses the same 392-pixel context for both learned models;
the autoencoder has 98,254 trainable parameters versus 97,956 for JEPA. The
primary metric is centered effective rank.

## Closure result

The closure run used seeds `42`, `100`, `2026`, `3141`, and `404`, three
training epochs, two probe epochs, and fixed budgets of 10,000 training and
2,000 test examples. The per-seed reports and aggregate summary are retained
under `results/closure_20260914_094423_899940Z/`.

Primary metric: centered effective rank.

Secondary diagnostics: linear-probe accuracy, feature variance, and average
pairwise cosine similarity.

Aggregate mean +/- standard deviation:

- Autoencoder: rank `23.234960 +/- 0.937810`, accuracy `0.641300 +/- 0.008035`.
- JEPA: rank `26.001224 +/- 0.867318`, accuracy `0.386700 +/- 0.036774`.
- JEPA minus autoencoder: rank `+2.766264`, accuracy `-0.254600`.

The result is mixed. JEPA had higher effective rank on average, but lower probe
accuracy and lower feature variance, with higher cosine similarity. The primary
metric supports greater rank under this setup, while probe accuracy does not
support the broader claim that JEPA learned better semantic representations.
The collapsed control remained at zero variance and effective rank with cosine
similarity 1.0, confirming that the diagnostics detect the deliberate failure.
