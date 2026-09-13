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

The autoencoder and JEPA results are intentionally left to the executable
trial. Their relative ranking is an empirical question, and the short MNIST
setup is not evidence for a general claim about JEPA quality.

The executable trial uses the same 392-pixel context for both learned models;
the autoencoder has 98,254 trainable parameters versus 97,956 for JEPA. The
primary metric is centered effective rank.
