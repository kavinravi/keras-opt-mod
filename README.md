# keras-opt

Keras Scipy optimize interface

Interface between keras optimizers and scipy.optimize. It is used to run full
batch optimization rather than mini-batch stochastic gradient descent. It
is applicable to factorization of very sparse matrices where stochastic
gradient descent is not able to converge.

## Features

- Supports all scipy.optimize methods including first-order methods (CG, BFGS, L-BFGS-B)
- **[NEW]** Supports second-order methods with Hessian-vector products (trust-ncg, trust-krylov, trust-exact, newton-cg)
- Full-batch optimization for better convergence on small to medium datasets
- Drop-in replacement for standard Keras training

## Example usage:

```python
#%%
# Model definition (linear regression)
from keras_opt import scipy_optimizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(4,)))
model.add(Dense(1, use_bias=False))
model.compile(loss='mse')

#%%
# Generate test data

import numpy as np

np.random.seed(42)
X = np.random.uniform(size=40).reshape(10, 4)
y = np.dot(X, np.array([1, 2, 3, 4])[:, np.newaxis])

#%%
# Use scipy.optimize to minimize the cost
model.train_function = scipy_optimizer.make_train_function(
            model, maxiter=20)
history = model.fit(X, y)

#%%
# Show weights.
model.trainable_weights
```

### **[NEW]** Using second-order methods (trust-ncg)

Second-order methods like `trust-ncg` use Hessian information for potentially faster convergence:

```python
# Use trust-ncg with Hessian-vector products
model.train_function = scipy_optimizer.make_train_function(
    model, 
    method='trust-ncg',  # Other options: 'trust-krylov', 'trust-exact', 'newton-cg'
    maxiter=20
)
history = model.fit(X, y)
```

You can also use tolerance parameters instead of (or in addition to) `maxiter`:

```python
# Use trust-ncg with gradient tolerance
model.train_function = scipy_optimizer.make_train_function(
    model, 
    method='trust-ncg',
    maxiter=100,  # Maximum iterations
    gtol=1e-5,    # Gradient tolerance
    xtol=1e-8     # Parameter tolerance
)
history = model.fit(X, y)
```

**Supported second-order methods:**
- `trust-ncg`: Newton Conjugate Gradient trust-region algorithm
- `trust-krylov`: Nearly exact trust-region algorithm (Krylov method)
- `trust-exact`: Exact trust-region algorithm
- `newton-cg`: Newton-CG algorithm

**Supported first-order methods:**
- `cg`: Conjugate Gradient (default)
- `bfgs`: Broyden-Fletcher-Goldfarb-Shanno
- `l-bfgs-b`: Limited-memory BFGS with box constraints
- And all other scipy.optimize methods

**Common tolerance parameters for second-order methods:**
- `gtol`: Gradient norm tolerance for convergence
- `xtol`: Parameter change tolerance
- `maxiter`: Maximum number of iterations

## Testing

To verify the implementation is working correctly:

```bash
# Run unit tests
python -m pytest tests/

# Or run specific tests
python tests/test_scipy_optimizer.py
python tests/test_trust_ncg.py
```

The `test_trust_ncg.py` file verifies that Hessian-vector products are being computed and used correctly by second-order methods.

## Examples

See the `examples/` directory for usage examples:

- `examples/example_trust_ncg.py` - Demonstrates using trust-ncg with both maxiter and tolerance parameters
