"""
Simple test for trust-ncg optimizer
"""

import numpy as np
from tensorflow import keras
from keras_opt import scipy_optimizer

# Create a simple neural network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    keras.layers.Dense(1, activation='linear')
])

model.compile(loss='mean_squared_error', metrics=['mae'])

# Generate some test data
np.random.seed(42)
X_train = np.random.randn(100, 5)
y_train = np.sum(X_train, axis=1) + np.random.randn(100) * 0.1

X_test = np.random.randn(20, 5)
y_test = np.sum(X_test, axis=1) + np.random.randn(20) * 0.1

# Use trust-ncg optimizer
model.train_function = scipy_optimizer.make_train_function(
    model, 
    method='trust-ncg',
    maxiter=50
)

print("Training with trust-ncg optimizer...")
history = model.fit(X_train, y_train, 
                    epochs=1, 
                    verbose=1,
                    validation_data=(X_test, y_test))

print("\nTraining complete!")
print(f"Final training loss: {history.history['loss'][0]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][0]:.4f}")

