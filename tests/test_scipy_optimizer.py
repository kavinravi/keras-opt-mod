""" Unit tests for scipy_optimizer
"""
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split

import keras_opt.scipy_optimizer as scipy_optimizer


class ScipyOptimizerTest(unittest.TestCase):
    """ Unit tests for the scipy_optimizer module.
    """

    def setUp(self):
        np.random.seed(42)
        tf.keras.utils.set_random_seed(42)

    def test_basic_training(self):
        """ Test basic training on simple data.
        """
        # Create simple training data: y = 2*x + 1
        X = np.random.rand(50, 1).astype(np.float32)
        y = (2 * X + 1).astype(np.float32)
        
        model = Sequential([
            Dense(1, input_dim=1)
        ])
        model.compile(loss='mse')
        
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=50)
        hist = model.fit(X, y, verbose=False)
        
        self.assertTrue('loss' in hist.history)
        self.assertLess(hist.history['loss'][-1], 0.01)

    def test_multi_feature(self):
        """ Test with multiple input features.
        """
        # Create data with multiple features
        X = np.random.rand(100, 4).astype(np.float32)
        y = (X[:, 0] + 2*X[:, 1] + 3*X[:, 2] + 4*X[:, 3]).reshape(-1, 1)
        
        model = Sequential([
            Dense(1, input_dim=4)
        ])
        model.compile(loss='mse')
        
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=50)
        hist = model.fit(X, y, verbose=False)
        
        self.assertLess(hist.history['loss'][-1], 0.1)

    def test_hidden_layer(self):
        """ Test with a hidden layer.
        """
        # Simple nonlinear problem
        X = np.random.rand(100, 3).astype(np.float32)
        y = (X[:, 0]**2 + X[:, 1] + X[:, 2]).reshape(-1, 1)
        
        model = Sequential([
            Dense(10, activation='relu', input_dim=3),
            Dense(1)
        ])
        model.compile(loss='mse')
        
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=100)
        hist = model.fit(X, y, verbose=False)
        
        self.assertLess(hist.history['loss'][-1], 0.2)

    def test_bfgs_method(self):
        """ Test using BFGS optimization method.
        """
        X = np.random.rand(50, 2).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
        
        model = Sequential([
            Dense(1, input_dim=2)
        ])
        model.compile(loss='mse')
        
        model.train_function = scipy_optimizer.make_train_function(
            model, method='bfgs', verbose=0, maxiter=100)
        hist = model.fit(X, y, verbose=False)
        
        self.assertLess(hist.history['loss'][-1], 0.1)

    def test_validation_data(self):
        """ Test with validation data.
        """
        X = np.random.rand(100, 2).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1] * 3).reshape(-1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        model = Sequential([
            Dense(10, activation='relu', input_dim=2),
            Dense(1)
        ])
        model.compile(loss='mse', metrics=['mae'])
        
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=100)
        
        hist = model.fit(X_train, y_train,
                         validation_data=(X_test, y_test), verbose=False)
        self.assertLess(hist.history['loss'][-1], 0.5)
        self.assertLess(hist.history['val_loss'][-1], 0.5)
        self.assertIn('mae', hist.history)
        self.assertIn('val_mae', hist.history)

    def test_multiple_inputs(self):
        """ Test model with multiple inputs.
        """
        X1 = np.random.rand(50, 1).astype(np.float32)
        X2 = np.random.rand(50, 1).astype(np.float32)
        y = (X1 + 2*X2).astype(np.float32)
        
        input1 = Input(shape=(1,))
        input2 = Input(shape=(1,))
        combined = Concatenate()([input1, input2])
        output = Dense(1)(combined)
        model = Model([input1, input2], output)
        
        model.compile(loss='mse')
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=100)
        
        hist = model.fit([X1, X2], y, verbose=False)
        self.assertLess(hist.history['loss'][-1], 0.1)

    def test_larger_dataset(self):
        """ Test with a larger dataset.
        """
        X = np.random.rand(500, 3).astype(np.float32)
        y = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1)
        
        model = Sequential([
            Dense(5, activation='relu', input_dim=3),
            Dense(1)
        ])
        model.compile(loss='mse')
        
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=50)
        hist = model.fit(X, y, verbose=False)
        
        self.assertLess(hist.history['loss'][-1], 0.2)


if __name__ == '__main__':
    unittest.main()
