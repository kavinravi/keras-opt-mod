"""
Example using trust-ncg optimizer for regression
"""

import numpy as np
from tensorflow import keras
from keras_opt import scipy_optimizer
from sklearn.model_selection import train_test_split

# Generate synthetic regression data
np.random.seed(123)
X = np.random.randn(200, 8)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + np.random.randn(200) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(loss='mean_squared_error', metrics=['mae'])

# Configure trust-ncg optimizer
model.train_function = scipy_optimizer.make_train_function(
    model,
    method='trust-ncg',
    maxiter=100
)

print("Training with trust-ncg optimizer...")
history = model.fit(X_train, y_train,
                    epochs=1,
                    verbose=1,
                    validation_data=(X_test, y_test))

print("\nFinal results:")
print(f"Training loss: {history.history['loss'][0]:.4f}")
print(f"Validation loss: {history.history['val_loss'][0]:.4f}")
print(f"Validation MAE: {history.history['val_mae'][0]:.4f}")

