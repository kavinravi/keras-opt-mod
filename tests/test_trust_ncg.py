"""Numerical and fit-integration regressions for full-batch second-order training."""
import unittest
from unittest import mock

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras

from keras_opt import scipy_optimizer as so


def dataset(x, y, batch=2, weights=None):
    values = (np.asarray(x, dtype='float64').reshape(-1, 1),
              np.asarray(y, dtype='float64').reshape(-1, 1))
    if weights is not None:
        values += (np.asarray(weights, dtype='float64'),)
    return tf.data.Dataset.from_tensor_slices(values).batch(batch)


def linear(weight=0., regularizer=None):
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype='float64'),
        keras.layers.Dense(1, use_bias=False, dtype='float64',
                           kernel_initializer=keras.initializers.Constant(weight),
                           kernel_regularizer=regularizer),
    ])
    model.compile(loss=keras.losses.MeanSquaredError())
    return model


def flat(model):
    return np.concatenate([v.numpy().reshape(-1) for v in model.trainable_variables])


class ActivityModel(keras.Model):
    """A model-level add_loss must be cleared on every call."""
    def __init__(self):
        super().__init__(dtype='float64')
        self.dense = keras.layers.Dense(1, use_bias=False, dtype='float64',
                                       kernel_initializer='ones')

    def call(self, inputs, training=False):
        outputs = self.dense(inputs)
        self.add_loss(tf.reduce_mean(outputs ** 2))
        return outputs


class TrustNcgTest(unittest.TestCase):
    def setUp(self):
        keras.backend.clear_session()
        self.old_floatx = keras.backend.floatx()
        keras.backend.set_floatx('float64')
        keras.utils.set_random_seed(42)

    def tearDown(self):
        keras.backend.set_floatx(self.old_floatx)

    def test_hvp_at_requested_weights_and_rejected_trial(self):
        model = keras.Sequential([
            keras.Input(shape=(1,)),
            keras.layers.Dense(1, use_bias=False, kernel_initializer='ones'),
            keras.layers.Lambda(tf.square),
        ])
        model.compile(loss='mse')
        opt = so.ScipyOptimizer(model, verbose=0)
        data = dataset([1], [1])
        # No objective call is needed before an HVP.
        np.testing.assert_allclose(opt._hessp_generator([.1], [1.], data), [-3.88])
        opt._fun_generator([2.1], data)
        np.testing.assert_allclose(opt._hessp_generator([.1], [1.], data), [-3.88])
        result = minimize(opt._fun_generator, [.1], jac=True,
                          hessp=opt._hessp_generator, args=(data,), method='trust-ncg',
                          options={'initial_trust_radius': 2., 'gtol': 1e-8})
        self.assertTrue(result.success, result.message)
        self.assertLess(result.fun, 1e-12)

    def test_hvp_matches_finite_difference_and_is_linear(self):
        model = keras.Sequential([keras.Input(shape=(1,)),
                                  keras.layers.Dense(3, activation='tanh'),
                                  keras.layers.Dense(1)])
        model.compile(loss='mse')
        data = dataset([1, 2, 3], [2, 1, 4])
        opt = so.ScipyOptimizer(model, verbose=0)
        x = flat(model)
        p = np.random.randn(len(x))
        q = np.random.randn(len(x))
        eps = 1e-5
        numerical = (opt._fun_generator(x + eps*p, data)[1] -
                     opt._fun_generator(x - eps*p, data)[1]) / (2*eps)
        hp = opt._hessp_generator(x, p, data)
        hq = opt._hessp_generator(x, q, data)
        np.testing.assert_allclose(hp, numerical, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(opt._hessp_generator(x, p+q, data), hp+hq,
                                   rtol=1e-10, atol=1e-10)
        self.assertAlmostEqual(np.dot(q, hp), np.dot(p, hq), places=10)

    def test_uneven_batches_weights_and_regularization(self):
        opt = so.ScipyOptimizer(linear(regularizer=keras.regularizers.l2(.1)), verbose=0)
        for batch in (1, 2, 3):
            data = dataset([1, 2, 3], [0, 0, 9], batch, weights=[1, 2, 3])
            loss, grad = opt._fun_generator([2.], data)
            self.assertAlmostEqual(loss, (4 + 32 + 27)/3 + .4)
            np.testing.assert_allclose(grad, [(4+32-54)/3+.4])
            np.testing.assert_allclose(opt._hessp_generator([2.], [1.], data),
                                       [2*(1+8+27)/3+.2])

    def test_fit_uneven_batches_finds_true_mean(self):
        model = linear()
        model.train_function = so.make_train_function(model, method='trust-ncg',
                                                       verbose=0, maxiter=20)
        history = model.fit(np.ones((3, 1)), np.array([0., 0., 9.]),
                            batch_size=2, shuffle=True, verbose=0)
        np.testing.assert_allclose(flat(model), [3.], atol=1e-7)
        self.assertAlmostEqual(history.history['loss'][0], 18.)

    def test_one_solve_per_epoch_metrics_validation_and_restore(self):
        for metrics in ([], ['mae']):
            model = linear(5.)
            model.compile(loss='mse', metrics=metrics)
            model.train_function = so.make_train_function(model, method='trust-ncg',
                                                           verbose=0, maxiter=2)
            with mock.patch.object(so, 'minimize', wraps=minimize) as solve:
                hist = model.fit(np.ones((9, 1)), np.full((9, 1), 5.), batch_size=4,
                                 epochs=2, verbose=0,
                                 validation_data=(np.ones((5, 1)), np.full((5, 1), 5.)))
            self.assertEqual(solve.call_count, 2)
            np.testing.assert_allclose(hist.history['loss'], [0., 0.])
            np.testing.assert_allclose(hist.history['val_loss'], [0., 0.])
            if metrics:
                np.testing.assert_allclose(hist.history['mae'], [0., 0.])
                np.testing.assert_allclose(hist.history['val_mae'], [0., 0.])
            self.assertEqual(so._get_steps(model), 1)
            self.assertEqual(model.evaluate(np.ones((2, 1)), np.full((2, 1), 5.),
                                            verbose=0, return_dict=True)['loss'], 0.)

    def test_partial_repeated_dataset_preserves_iterator_position(self):
        model = linear()
        model.train_function = so.make_train_function(model, method='trust-ncg',
                                                       verbose=0, maxiter=20)
        data = dataset([1, 1, 1, 1], [1, 1, 3, 3], 1).repeat()
        weights = []
        callback = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: weights.append(float(flat(model)[0])))
        with mock.patch.object(so, 'minimize', wraps=minimize) as solve:
            model.fit(data, steps_per_epoch=2, epochs=2, verbose=0, callbacks=[callback])
        self.assertEqual(solve.call_count, 2)
        np.testing.assert_allclose(weights, [1., 3.], atol=1e-8)

    def test_validation_steps_are_respected(self):
        model = linear(5.)
        model.train_function = so.make_train_function(model, verbose=0)
        validation = dataset([1, 1, 1], [5, 7, 100], 1).repeat()
        hist = model.fit(np.ones((9, 1)), np.full((9, 1), 5.), batch_size=2,
                         validation_data=validation, validation_steps=2, verbose=0)
        self.assertAlmostEqual(hist.history['val_loss'][0], 2.)

    def test_validation_includes_short_last_batch(self):
        model = linear(5.)
        model.train_function = so.make_train_function(model, verbose=0)
        hist = model.fit(np.ones((9, 1)), np.full((9, 1), 5.), batch_size=4,
                         validation_data=(np.ones((5, 1)), np.array([5., 5., 5., 5., 10.])),
                         verbose=0)
        self.assertAlmostEqual(hist.history['val_loss'][0], 5.)

    def test_dataset_is_frozen_during_solve(self):
        model = linear()
        model.train_function = so.make_train_function(model, verbose=0, maxiter=1)
        data = dataset([1, 1, 1], [0, 0, 9], 1).unbatch().shuffle(3).batch(2)
        data = data.map(lambda x, y: (x, y + tf.random.uniform(tf.shape(y), dtype=y.dtype)))
        def checked(fun, x, **kwargs):
            first = fun(x, *kwargs['args'])
            second = fun(x, *kwargs['args'])
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            return minimize(fun, x, **kwargs)
        with mock.patch.object(so, 'minimize', side_effect=checked):
            model.fit(data, steps_per_epoch=2, verbose=0)

    def test_dropout_batchnorm_are_deterministic(self):
        model = keras.Sequential([keras.Input(shape=(1,)), keras.layers.Dense(2),
                                  keras.layers.BatchNormalization(), keras.layers.Dropout(.5),
                                  keras.layers.Dense(1)])
        model.compile(loss='mse')
        opt = so.ScipyOptimizer(model, verbose=0)
        data = dataset([1, 2, 3], [1, 2, 3])
        x = flat(model)
        before = [v.numpy().copy() for v in model.non_trainable_variables]
        a = opt._fun_generator(x, data)
        hp = opt._hessp_generator(x, np.ones_like(x), data)
        b = opt._fun_generator(x, data)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(hp, opt._hessp_generator(x, np.ones_like(x), data))
        for old, new in zip(before, model.non_trainable_variables):
            np.testing.assert_array_equal(old, new.numpy())

    def test_model_level_add_loss_and_graph_reuse(self):
        model = ActivityModel()
        model(tf.ones((1, 1)))
        model.compile(loss='mse')
        opt = so.ScipyOptimizer(model, verbose=0)
        data = dataset([1, 1, 1], [0, 0, 0])
        for p in (1., 2., 3.):
            loss, grad = opt._fun_generator([1.], data)
            self.assertAlmostEqual(loss, 2.)
            np.testing.assert_allclose(grad, [4.])
            np.testing.assert_allclose(opt._hessp_generator([1.], [p], data), [4*p])
        self.assertEqual(opt._value_and_grad.experimental_get_tracing_count(), 1)
        self.assertEqual(opt._hessian_product.experimental_get_tracing_count(), 1)

    def test_sparse_gradients_and_disconnected_variable(self):
        class SparseModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.embedding = keras.layers.Embedding(5, 1)
                self.unused = self.add_weight(name='unused', shape=(1,))
            def call(self, inputs):
                return tf.squeeze(self.embedding(inputs), axis=1)
        model = SparseModel()
        model(tf.constant([[1], [2]]))
        model.compile(loss='mse')
        data = tf.data.Dataset.from_tensor_slices(([[1], [2]], [[1.], [2.]])).batch(2)
        opt = so.ScipyOptimizer(model, verbose=0)
        hp = opt._hessp_generator(flat(model), np.ones_like(flat(model)), data)
        expected = np.concatenate([np.zeros(v.shape).reshape(-1) if v is model.unused
                                   else np.array([0., 1., 1., 0., 0.])
                                   for v in model.trainable_variables])
        np.testing.assert_allclose(hp, expected)

    def test_supported_methods_and_rejections(self):
        for method in ('Trust-NCG', 'trust-krylov', 'Newton-CG', 'l-bfgs-b'):
            model = linear()
            model.train_function = so.make_train_function(model, method=method,
                                                           maxiter=20, verbose=0)
            model.fit(np.ones((3, 1)), np.full((3, 1), 2.), batch_size=2, verbose=0)
            np.testing.assert_allclose(flat(model), [2.], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'dense Hessian'):
            so.ScipyOptimizer(linear(), method='trust-exact')
        with self.assertRaisesRegex(ValueError, 'gtol'):
            so.ScipyOptimizer(linear(), method='trust-ncg', xtol=1e-8)
        model = linear()
        model.compile(loss=keras.losses.MeanSquaredError(reduction='sum'))
        with self.assertRaisesRegex(ValueError, 'mean-reduced'):
            so.ScipyOptimizer(model)

    def test_compile_and_data_guards_and_recompile(self):
        with self.assertRaisesRegex(RuntimeError, 'Compile'):
            so.make_train_function(keras.Sequential([keras.Input(shape=(1,)),
                                                    keras.layers.Dense(1)]))
        model = linear()
        model.train_function = so.make_train_function(model, verbose=0)
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            model.fit(dataset([1], [1]).repeat(), verbose=0)
        with mock.patch.object(so, 'minimize', side_effect=RuntimeError('deliberate')):
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                model.fit(np.ones((5, 1)), np.ones((5, 1)), batch_size=2, verbose=0)
        self.assertEqual(so._get_steps(model), 1)
        model.compile(optimizer='sgd', loss='mse')
        with mock.patch.object(so, 'minimize') as solve:
            model.fit(np.ones((5, 1)), np.ones((5, 1)), batch_size=2, verbose=0)
        solve.assert_not_called()

    def test_memory_cache_unbuilt_model_and_validation_split(self):
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile(loss='mse')
        # Reinstallation must not nest adapters or duplicate solves.
        model.train_function = so.make_train_function(model, verbose=0)
        model.train_function = so.make_train_function(
            model, method='trust-ncg', cache='memory', maxiter=20, verbose=0)
        with mock.patch.object(so, 'minimize', wraps=minimize) as solve:
            hist = model.fit(np.ones((10, 1)), np.full((10, 1), 3.),
                             batch_size=3, validation_split=.2, verbose=0)
        self.assertEqual(solve.call_count, 1)
        self.assertLess(hist.history['loss'][0], 1e-12)
        self.assertLess(hist.history['val_loss'][0], 1e-12)

    def test_stateful_and_low_precision_models_rejected(self):
        stateful = keras.Sequential([
            keras.Input(batch_shape=(1, 2, 1)),
            keras.layers.SimpleRNN(1, stateful=True)])
        stateful.compile(loss='mse')
        with self.assertRaisesRegex(ValueError, 'Stateful'):
            so.ScipyOptimizer(stateful)
        low_precision = keras.Sequential([
            keras.Input(shape=(1,)), keras.layers.Dense(1, dtype='mixed_float16')])
        low_precision.compile(loss='mse')
        with self.assertRaisesRegex(ValueError, 'float32 or float64'):
            so.ScipyOptimizer(low_precision)


if __name__ == '__main__':
    unittest.main()
