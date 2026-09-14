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

The bounded closure run used seeds 42, 43, and 44, three training epochs, two
probe epochs, and fixed budgets of 2,048 training and 1,024 test examples. The
per-seed reports and aggregate summary are retained under
`results/closure_20260914_083005_266639Z/`.

This is a mixed result. JEPA had higher centered effective rank for all three
seeds, with a paired mean difference of `+12.312186` (standard deviation
`0.362480`). Its representation variance was lower by `-0.062393` on average,
and its average pairwise cosine similarity was higher by `+0.036617`.
However, the linear-probe accuracy difference was `-0.063833` on average, with
JEPA below the autoencoder on two of three seeds. The primary metric therefore
supports the representation-diversity part of the hypothesis, while probe
accuracy does not support a broad claim of better semantic representations in
this small trial. The collapsed control remained at zero variance and
effective rank with cosine similarity 1.0, confirming that the diagnostics
detect the deliberately failed representation.
