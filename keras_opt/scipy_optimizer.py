"""Deterministic, full-dataset SciPy optimization for TensorFlow/Keras models."""

import functools
import inspect
import itertools
import os
import tempfile

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras


class ScipyOptimizer:
    """Minimize a mean-reduced Keras loss using bounded-memory derivatives.

    Models are evaluated with ``training=False``: dropout is disabled and batch
    normalization uses its existing moving statistics. Each epoch's inputs are
    cached once so data augmentation and shuffling cannot change SciPy's target.
    ``cache='disk'`` (default) bounds input memory; ``'memory'`` avoids disk I/O.
    The latest SciPy OptimizeResult is available as ``result``.
    """

    HESSIAN_METHODS = {'trust-ncg', 'trust-krylov', 'newton-cg'}
    METHODS = HESSIAN_METHODS | {'cg', 'bfgs', 'l-bfgs-b'}

    def __init__(self, model, method='cg', verbose=1, maxiter=1,
                 cache='disk', **optimizer_kwargs):
        self.model = model
        self.method = method.lower()
        if self.method == 'trust-exact':
            raise ValueError('trust-exact requires a dense Hessian; use trust-ncg '
                             'or trust-krylov for Hessian-vector products.')
        if self.method not in self.METHODS:
            raise ValueError(f'Unsupported method {method!r}; choose {sorted(self.METHODS)}')
        if cache not in ('disk', 'memory'):
            raise ValueError("cache must be 'disk' or 'memory'")
        if self.method.startswith('trust-') and 'xtol' in optimizer_kwargs:
            raise ValueError('trust-ncg/trust-krylov use gtol, not xtol')
        self.verbose = verbose
        self.maxiter = maxiter
        self.cache = cache
        self.optimizer_kwargs = optimizer_kwargs
        self.result = None
        self._epoch_batches = None
        self._variables = None
        self._sizes = None
        self._shapes = None
        self._loss_has_training = 'training' in inspect.signature(model.compute_loss).parameters
        self._validate_model()
        # These objects survive all SciPy calls. Directions and datasets are
        # arguments, never closure constants. Tapes only retain one batch.
        self._assign = tf.function(self._assign_impl, reduce_retracing=True)
        self._value_and_grad = tf.function(self._value_and_grad_impl, reduce_retracing=True)
        self._hessian_product = tf.function(self._hessian_product_impl, reduce_retracing=True)
        self._evaluate = tf.function(self._evaluate_impl, reduce_retracing=True)
        self._count = tf.function(self._count_impl, reduce_retracing=True)

    def _validate_model(self):
        if self.model.distribute_strategy.num_replicas_in_sync != 1:
            raise ValueError('ScipyOptimizer currently supports a single replica only')
        pending = [self.model]
        while pending:
            layer = pending.pop()
            if getattr(layer, 'stateful', False):
                raise ValueError('Stateful layers are not supported by a fixed SciPy objective')
            if tf.as_dtype(layer.compute_dtype) not in (tf.float32, tf.float64):
                raise ValueError('ScipyOptimizer requires float32 or float64 computation')
            pending.extend(getattr(layer, 'layers', ()))
        for loss in tf.nest.flatten(self.model.loss):
            if isinstance(loss, keras.losses.Loss) and loss.reduction not in (
                    'auto', 'sum_over_batch_size', 'mean'):
                raise ValueError('ScipyOptimizer requires mean-reduced losses '
                                 '(sum_over_batch_size or mean)')

    def _initialize(self):
        variables = tuple(self.model.trainable_variables)
        if self._variables is not None:
            if len(variables) != len(self._variables) or any(
                    v is not old for v, old in zip(variables, self._variables)):
                raise ValueError('Trainable variables changed; recreate the train function')
            return
        if not variables:
            raise ValueError('Build the model with trainable variables before optimizing')
        self._variables = variables
        self._shapes = [tuple(v.shape) for v in variables]
        self._sizes = [int(np.prod(shape)) for shape in self._shapes]

    def _predict(self, x_data, training=False):
        # __call__ establishes Keras' scope and clears transient add_loss values.
        return self.model(x_data, training=training)

    def _loss(self, batch):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(batch)
        y_pred = self._predict(x)
        kwargs = dict(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        if self._loss_has_training:
            kwargs['training'] = False
        loss = self.model.compute_loss(**kwargs)
        tf.debugging.assert_rank(loss, 0, message='The model loss must be a scalar mean')
        count = tf.cast(tf.shape(tf.nest.flatten(x)[0])[0], tf.float64)
        return tf.cast(loss, tf.float64), count

    def _flatten(self, values):
        return tf.concat([
            tf.reshape(tf.cast(tf.convert_to_tensor(value), tf.float64), [-1])
            if value is not None else tf.zeros([size], tf.float64)
            for value, size in zip(values, self._sizes)
        ], axis=0)

    def _assign_impl(self, x):
        for value, variable, shape in zip(tf.split(x, self._sizes),
                                          self._variables, self._shapes):
            variable.assign(tf.cast(tf.reshape(value, shape), variable.dtype))

    def _update_weights(self, x):
        self._initialize()
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (sum(self._sizes),):
            raise ValueError('Parameter vector has the wrong shape')
        self._assign(tf.convert_to_tensor(x))

    def _value_and_grad_impl(self, dataset):
        total_loss = tf.constant(0., tf.float64)
        total_count = tf.constant(0., tf.float64)
        total_grad = tf.zeros([sum(self._sizes)], tf.float64)
        for batch in dataset:
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            with tf.GradientTape() as tape:
                loss, count = self._loss(batch)
            grads = tape.gradient(loss, self._variables)
            total_loss += loss * count
            total_grad += self._flatten(grads) * count
            total_count += count
        tf.debugging.assert_positive(total_count, message='Cannot optimize empty data')
        return total_loss / total_count, total_grad / total_count

    def _hessian_product_impl(self, dataset, direction):
        directions = [tf.cast(tf.reshape(p, shape), v.dtype)
                      for p, shape, v in zip(tf.split(direction, self._sizes),
                                             self._shapes, self._variables)]
        total_hvp = tf.zeros([sum(self._sizes)], tf.float64)
        total_count = tf.constant(0., tf.float64)
        for batch in dataset:
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            with tf.GradientTape() as outer:
                with tf.GradientTape() as inner:
                    loss, count = self._loss(batch)
                grads = inner.gradient(loss, self._variables)
                products = []
                for grad, p in zip(grads, directions):
                    if isinstance(grad, tf.IndexedSlices):
                        products.append(tf.cast(tf.reduce_sum(
                            grad.values * tf.gather(p, grad.indices)), tf.float64))
                    elif grad is not None:
                        products.append(tf.cast(tf.reduce_sum(grad * p), tf.float64))
                dot = tf.add_n(products) if products else tf.constant(0., tf.float64)
            hvp = outer.gradient(dot, self._variables)
            total_hvp += self._flatten(hvp) * count
            total_count += count
        tf.debugging.assert_positive(total_count, message='Cannot optimize empty data')
        return total_hvp / total_count

    @staticmethod
    def _as_dataset(data):
        if isinstance(data, tf.data.Dataset):
            return data
        dataset = getattr(data, '_dataset', None)
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError('Expected a TensorFlow dataset or single-replica dataset iterator')
        return dataset

    def _fun_generator(self, x, data):
        self._update_weights(x)
        # Passing an iterator produces a reusable graph while-loop. Passing a
        # Dataset instead makes AutoGraph use Dataset.reduce, whose function
        # pipeline is optimized again on every SciPy callback.
        loss, grads = self._value_and_grad(iter(self._as_dataset(data)))
        return float(loss.numpy()), grads.numpy()

    def _hessp_generator(self, x, p, data):
        # A rejected trial can leave model weights different from the requested x.
        self._update_weights(x)
        direction = tf.convert_to_tensor(np.asarray(p, dtype=np.float64))
        return self._hessian_product(iter(self._as_dataset(data)), direction).numpy()

    def _evaluate_impl(self, dataset):
        for batch in dataset:
            # Keras' test_step handles loss weighting and metric containers in
            # both Keras 2 and 3, without compiled_metrics' deprecated shim.
            self.model.test_step(batch)
        results = {}
        for metric in self.model.metrics:
            value = metric.result()
            if isinstance(value, dict):
                results.update(value)
            else:
                results[metric.name] = value
        return results

    @staticmethod
    def _count_impl(dataset):
        count = tf.constant(0, tf.int64)
        for _ in dataset:
            count += 1
        return count

    def train_function(self, iterator):
        """Run exactly one SciPy solve on a frozen epoch, then report metrics."""
        source = self._as_dataset(iterator)
        steps = self._epoch_batches
        if steps is None:
            steps = int(source.cardinality().numpy())
        if steps <= 0:
            raise ValueError('Provide a nonempty finite dataset or explicit steps_per_epoch')
        # Consume the supplied iterator, so partial epochs and repeat() preserve
        # their position. Only this initial snapshot crosses Python per batch.
        snapshot = tf.data.Dataset.from_generator(
            lambda: itertools.islice(iterator, steps),
            output_signature=source.element_spec,
        ).apply(tf.data.experimental.assert_cardinality(steps))
        with tempfile.TemporaryDirectory(prefix='keras-opt-') as directory:
            filename = os.path.join(directory, 'epoch') if self.cache == 'disk' else ''
            dataset = snapshot.cache(filename)
            # Fully populate the cache before optimization (including its EOF).
            self._count(iter(dataset)).numpy()
            if not self.model.built:
                inputs, _, _ = keras.utils.unpack_x_y_sample_weight(next(iter(dataset)))
                self._predict(inputs)
            self._initialize()
            x0 = np.concatenate([v.numpy().reshape(-1) for v in self._variables]).astype(np.float64)
            options = dict(maxiter=self.maxiter, **self.optimizer_kwargs)
            kwargs = {'hessp': self._hessp_generator} if self.method in self.HESSIAN_METHODS else {}
            self.result = minimize(self._fun_generator, x0, method=self.method,
                                   jac=True, options=options, args=(dataset,), **kwargs)
            self._update_weights(self.result.x)
            if self.verbose:
                print(f'{self.method}: {self.result.message} '
                      f'(iterations={self.result.nit}, loss={self.result.fun:.6g})')
            self.model.reset_metrics()
            results = self._evaluate(iter(dataset))
            return {key: float(value.numpy()) for key, value in results.items()}


def _set_steps(model, steps):
    """Account for Keras 2's variable and Keras 3's integer setting."""
    if hasattr(model, '_configure_steps_per_execution'):
        model._configure_steps_per_execution(steps)
    else:
        model.steps_per_execution = steps


def _get_steps(model):
    steps = getattr(model, 'steps_per_execution', None)
    if steps is None:
        steps = model._steps_per_execution
    return int(steps.numpy()) if hasattr(steps, 'numpy') else int(steps)


def _fit_batches(arguments):
    """Resolve the epoch size before Keras constructs its epoch iterator."""
    steps = arguments.get('steps_per_epoch')
    if steps is not None:
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError('steps_per_epoch must be a positive integer')
        return int(steps)
    x = arguments.get('x')
    if isinstance(x, tf.data.Dataset):
        steps = int(x.cardinality().numpy())
        if steps <= 0:
            raise ValueError('Datasets with unknown/infinite cardinality require steps_per_epoch')
        return steps
    first = tf.nest.flatten(x)[0]
    if not hasattr(first, 'shape') or first.shape[0] is None:
        raise ValueError('Generators require explicit steps_per_epoch')
    size = int(first.shape[0] * (1 - arguments.get('validation_split', 0.)))
    if size <= 0:
        raise ValueError('Cannot optimize empty training data')
    batch_size = arguments.get('batch_size') or 32
    return (size + batch_size - 1) // batch_size


def make_train_function(model, **kwargs):
    """Return a SciPy train function and configure ``fit`` for full epochs.

    Assign the result to ``model.train_function`` after ``compile``. The fit
    adapter temporarily sets the execution size to the actual epoch length;
    evaluation keeps its usual batch granularity. Settings are restored even
    if training raises. Recompiling the model restores ordinary Keras training.
    """
    compiled = getattr(model, 'compiled', getattr(model, '_is_compiled', False))
    if not compiled:
        raise RuntimeError('Compile the model before creating its train function')
    opt = ScipyOptimizer(model, **kwargs)
    original_fit = getattr(model.fit, '_keras_opt_original', model.fit)
    signature = inspect.signature(original_fit)

    @functools.wraps(original_fit)
    def fit(*args, **fit_kwargs):
        # A subsequent compile() clears train_function: leave ordinary fit alone.
        if model.train_function != opt.train_function:
            return original_fit(*args, **fit_kwargs)
        bound = signature.bind(*args, **fit_kwargs)
        bound.apply_defaults()
        opt._epoch_batches = _fit_batches(bound.arguments)
        previous_steps = _get_steps(model)
        original_evaluate = model.evaluate

        @functools.wraps(original_evaluate)
        def evaluate(*eval_args, **eval_kwargs):
            _set_steps(model, 1)
            model.test_function = None
            # fit may have constructed its cached evaluation iterator before
            # entering this adapter, using the training execution size. Let
            # evaluate build one with its own (single-batch) execution size.
            if '_use_cached_eval_dataset' in eval_kwargs:
                eval_kwargs['_use_cached_eval_dataset'] = False
            try:
                return original_evaluate(*eval_args, **eval_kwargs)
            finally:
                _set_steps(model, opt._epoch_batches)

        _set_steps(model, opt._epoch_batches)
        model.evaluate = evaluate
        try:
            return original_fit(*args, **fit_kwargs)
        finally:
            model.evaluate = original_evaluate
            _set_steps(model, previous_steps)
            opt._epoch_batches = None
            # Compiled evaluation functions may have captured the temporary size.
            model.test_function = None

    fit._keras_opt_original = original_fit
    model.fit = fit
    return opt.train_function
