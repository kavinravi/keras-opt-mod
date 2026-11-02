# keras-opt

Full-batch SciPy optimization for TensorFlow/Keras models, including matrix-free
second-order methods. Model activations and derivative tapes are retained for
only one batch at a time; no dense Hessian is constructed.

## Usage

```python
import numpy as np
from tensorflow import keras
from keras_opt import scipy_optimizer

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(1, use_bias=False),
])
model.compile(loss='mse', metrics=['mae'])

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 4))
y = X @ np.array([1., 2., 3., 4.])

model.train_function = scipy_optimizer.make_train_function(
    model,
    method='trust-ncg',
    maxiter=50,
    gtol=1e-5,
    verbose=0,
)
history = model.fit(X, y, batch_size=32, epochs=1)
result = model.train_function.__self__.result
print(result.success, result.message, result.nit)
```

Each Keras epoch runs **one** SciPy solve. `maxiter` is the iteration budget for
that solve, and a solve can make multiple objective and Hessian-vector calls per
iteration. A short final batch is weighted by its example count, so changing
`batch_size` does not change a standard mean-loss objective.

`make_train_function` installs a small `fit` adapter that temporarily configures
Keras' execution size to the actual epoch length. It restores the setting and
normal evaluation behavior afterwards, including on errors. Training batch
callbacks fire once per full epoch; epoch callbacks, validation, and early
stopping still work. Recompiling the model clears the SciPy train function and
allows ordinary Keras training again.

## Methods and options

- Second order: `trust-ncg`, `trust-krylov`, `newton-cg`.
- First order: `cg` (default), `bfgs`, `l-bfgs-b`.
- Method names are case insensitive.
- `trust-exact` is deliberately unsupported: SciPy requires a dense Hessian.
- Extra keyword arguments are SciPy **method options**, such as
  `initial_trust_radius`, `gtol`, or `maxls`. They are not top-level arguments
  such as `bounds`, `constraints`, or `callback`.
- Trust-region methods use `gtol`; they do not accept `xtol`. Newton-CG uses
  `xtol`. See [SciPy's method documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

The latest `OptimizeResult` is retained on the optimizer, including failures and
iteration limits. Check `success` and `message`; a returned Keras history does
not by itself mean SciPy converged. With optimizer `verbose=1`, each solve prints
its status, iteration count, and objective value.

## Determinism and memory

SciPy must see the same objective at the same parameter vector. This package:

- Calls the model through its public call interface with **`training=False`**.
  Dropout and training-only augmentation inside the model are disabled;
  BatchNorm uses its existing moving statistics and does not update them.
  Trainable BatchNorm scale and offset can still be optimized. Choose this
  objective intentionally; it differs from ordinary stochastic Keras training.
- Freezes the input batches once per epoch, including random transformations in
  a `tf.data` pipeline. All objective, gradient, Hessian-product, and final
  metric evaluations share those exact batches.
- Defaults to `cache='disk'`, using an automatically cleaned temporary directory.
  Set `cache='memory'` when the entire input epoch comfortably fits in RAM.
  Derivative activation memory remains bounded to one batch in either mode.
- Reuses compiled TensorFlow functions across optimizer calls and passes each
  Hessian direction as a tensor, avoiding repeated graph construction.

Stateful recurrent layers, mixed float16/bfloat16 computation, and multi-replica
strategies are rejected. Custom layers/losses must also be deterministic in
inference mode and must not mutate model state during evaluation.

## Data, loss, and compatibility

Python 3.9+, TensorFlow 2.12+, NumPy 1.23+, and SciPy 1.10+ are required. Both
Keras 2 and TensorFlow-backed Keras 3 are supported; CI exercises TensorFlow
2.15.1 and 2.20.0, including the legacy `tf-keras` API.

Arrays, nested model inputs, and batched TensorFlow datasets are supported.
Generators and datasets with unknown/infinite cardinality require an explicit
`steps_per_epoch`. Partial epochs consume that many batches from the actual
Keras iterator, preserving its position across epochs. `validation_steps` is
respected independently of the training epoch size.

Use standard **mean-reduced losses** (`sum_over_batch_size`, `mean`, or legacy
`auto`). Sum reduction, unreduced loss objects, and `mean_with_sample_weight`
are rejected because their normalization differs. Sample weights follow Keras'
mean-loss convention: weighted losses are divided by the number of examples,
not the sum of weights. Weight regularizers are included once in the overall
objective. Custom `compute_loss` and activity `add_loss` terms should return
batch means so example-count weighting has the intended meaning.

SciPy vectors and dataset accumulators use float64, while the model retains its
configured dtype. For tight second-order convergence tolerances, build the model
and loss in float64 (for example, set `keras.backend.set_floatx('float64')` before
creating them); converting a float32 gradient afterwards cannot restore precision.

## Tests and benchmarks

```bash
python -m pip install -e '.[test]'
python -m ruff check .
python -m pytest -q
# Also supported without pytest:
python -m unittest discover -s tests

# Run the same suite using the legacy Keras API:
python -m pip install tf-keras==2.20.1
TF_USE_LEGACY_KERAS=1 python -m pytest -q

# Compare the previous implementation with this checkout:
python -m benchmarks.optimizer --output /tmp/keras-opt-benchmark.json
```

The regression suite checks finite-difference Hessian products, rejected trial
steps, uneven and weighted batches, regularization, sparse/disconnected
parameters, graph reuse, deterministic Dropout/BatchNorm behavior, model-level
`add_loss`, one solve per epoch, partial epochs, metric reporting, and validation.

The benchmark checks matching objective/gradient/Hessian values and measures
both warm callback latency and cold end-to-end training. It separates a single
SciPy solve from `model.fit`, because the old Keras 3 integration repeated a full
solve for every batch. Fixed iteration-budget timings are not time-to-equal-
accuracy measurements; the output includes true MSE and actual solver work.

### Measured results

On November 2, 2026, on this macOS ARM64 machine using one CPU thread,
TensorFlow 2.20 / Keras 3.12, float64, 128 examples and four batches:

| Measurement | Linear regression, before → after | MLP (8–32–16–1), before → after |
| --- | --- | --- |
| Warm loss + gradient | 31.66 → 1.74 ms (**18.2×**) | 49.39 → 1.82 ms (**27.1×**) |
| Warm Hessian-vector product | 45.73 → 1.93 ms (**23.7×**) | 105.50 → 2.16 ms (**48.9×**) |
| One five-iteration solve, including setup | 0.666 → 0.419 s (**1.59×**) | 1.551 → 0.553 s (**2.80×**) |
| One `fit` epoch, including setup | 2.558 → 0.408 s (**6.26×**) | 10.091 → 0.549 s (**18.37×**) |

Warm callback figures are medians of seven calls; training figures are medians
of three fresh-model runs, using the default disk cache. The single-solve
comparison has matching final true MSE and matching solver evaluation counts.
The `fit` comparison includes removal of the old four-solves-per-epoch bug:
its work and final accuracy differ, so it is **not** an equal-accuracy speedup.
Initial graph tracing is slower in the new implementation; its benefit comes
from reuse. These small CPU workloads do not establish GPU or large-model gains.

[Raw timings, losses, evaluation counts, and environment](benchmarks/results-macos-arm64.json)
are committed alongside the reproducible benchmark script.
