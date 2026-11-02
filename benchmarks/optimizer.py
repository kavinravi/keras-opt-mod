"""Reproduce CPU benchmarks against a Git revision without changing the checkout.

Run from the repository root: python -m benchmarks.optimizer --output /tmp/results.json
Reports median warm callback latency, cold latency, one SciPy solve, and model.fit.
All models use float64, deterministic data, and identical seeds and initial weights.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from unittest import mock

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from keras_opt import scipy_optimizer as current


def build_model(kind):
    keras.utils.set_random_seed(123)
    layers = [keras.Input(shape=(8,))]
    if kind == 'mlp':
        layers += [keras.layers.Dense(32, activation='tanh'),
                   keras.layers.Dense(16, activation='tanh')]
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)
    model.compile(loss='mse')
    return model


def data_arrays():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 8))
    y = (x @ np.arange(1., 9.) / 8).reshape(-1, 1)
    return x, y


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def callbacks(module, kind, calls):
    model = build_model(kind)
    opt = module.ScipyOptimizer(model, method='trust-ncg', verbose=0)
    x_data, y_data = data_arrays()
    it = iter(tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(32))
    x = np.concatenate([v.numpy().reshape(-1) for v in model.trainable_variables])
    p = np.random.default_rng(9).normal(size=x.shape)
    cold_fun, reference = timed(lambda: opt._fun_generator(x, it))
    cold_hvp, hp = timed(lambda: opt._hessp_generator(x, p, it))
    fun_times = []
    hvp_times = []
    for index in range(calls):
        # Vary values without changing the trace signature.
        point = x + index * 1e-5
        elapsed, _ = timed(lambda: opt._fun_generator(point, it))
        fun_times.append(elapsed)
        elapsed, _ = timed(lambda: opt._hessp_generator(point, p * (index+1), it))
        hvp_times.append(elapsed)
    return {'cold_fun_s': cold_fun, 'cold_hvp_s': cold_hvp,
            'warm_fun_s': statistics.median(fun_times),
            'warm_hvp_s': statistics.median(hvp_times),
            'loss': reference[0], 'gradient': reference[1].tolist(), 'hvp': hp.tolist()}


def training(module, kind, mode, repeats):
    rows = []
    for _ in range(repeats):
        model = build_model(kind)
        x, y = data_arrays()
        options = dict(method='trust-ncg', maxiter=5, gtol=1e-8,
                       initial_trust_radius=.1, verbose=0)
        if mode == 'fit':
            model.train_function = module.make_train_function(model, **options)
            def run():
                return model.fit(x, y, batch_size=32, shuffle=False, verbose=0)
        else:
            opt = module.ScipyOptimizer(model, **options)
            it = iter(tf.data.Dataset.from_tensor_slices((x, y)).batch(32))
            def run():
                return opt.train_function(it)
        solves = []
        original_minimize = module.minimize
        def capture(*args, **kwargs):
            result = original_minimize(*args, **kwargs)
            solves.append({k: int(result[k]) for k in ('nit', 'nfev', 'nhev')})
            return result
        with mock.patch.object(module, 'minimize', side_effect=capture):
            seconds, _ = timed(run)
        mse = float(np.mean((model(x, training=False).numpy() - y) ** 2))
        rows.append({'seconds': seconds, 'mse': mse, 'solves': solves})
    return {'median_s': statistics.median(row['seconds'] for row in rows), 'runs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='e532f15678818a2f9328a69b0a90af08c0c9258c')
    parser.add_argument('--output', default='/tmp/keras-opt-benchmark.json')
    parser.add_argument('--calls', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.get_logger().setLevel('ERROR')
    warnings.filterwarnings('ignore')
    keras.backend.set_floatx('float64')
    output = {'baseline': args.baseline, 'environment': {
        'python': sys.version, 'platform': platform.platform(),
        'tensorflow': tf.__version__, 'keras': getattr(keras, '__version__', 'legacy'),
        'scipy': scipy.__version__, 'numpy': np.__version__, 'threads': 1,
        'dtype': 'float64', 'examples': 128, 'batch_size': 32,
        'maxiter_per_solve': 5, 'callback_repeats': args.calls, 'training_repeats': args.repeats,
        'new_cache': 'disk'}, 'results': {}}
    with tempfile.TemporaryDirectory(prefix='keras-opt-baseline-') as directory:
        source = subprocess.check_output(
            ['git', 'show', f'{args.baseline}:keras_opt/scipy_optimizer.py'], text=True)
        path = Path(directory) / 'baseline.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('keras_opt_baseline', path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        for kind in ('linear', 'mlp'):
            output['results'][kind] = {}
            references = {}
            for label, module in [('before', baseline), ('after', current)]:
                print(f'{kind}: {label}', flush=True)
                callback = callbacks(module, kind, args.calls)
                references[label] = callback
                output['results'][kind][label] = {
                    'callbacks': {k: v for k, v in callback.items() if k not in ('gradient', 'hvp')},
                    'single_solve': training(module, kind, 'single_solve', args.repeats),
                    'fit': training(module, kind, 'fit', args.repeats)}
            for key in ('loss', 'gradient', 'hvp'):
                np.testing.assert_allclose(references['before'][key], references['after'][key],
                                           rtol=1e-9, atol=1e-10)
            before = output['results'][kind]['before']
            after = output['results'][kind]['after']
            np.testing.assert_allclose(
                [row['mse'] for row in before['single_solve']['runs']],
                [row['mse'] for row in after['single_solve']['runs']],
                rtol=1e-9, atol=1e-10)
            ratios = {key: before['callbacks'][key]/after['callbacks'][key]
                      for key in ('warm_fun_s', 'warm_hvp_s')}
            ratios.update({key: before[key]['median_s']/after[key]['median_s']
                           for key in ('single_solve', 'fit')})
            output['results'][kind]['speedup'] = ratios
            print(json.dumps(ratios), flush=True)
            Path(args.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
