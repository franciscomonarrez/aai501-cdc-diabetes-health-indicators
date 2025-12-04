"""
Multilayer Perceptron (MLP) model for diabetes prediction using TensorFlow/Keras.

This module provides:
- MLP model architecture builder
- Training utilities with early stopping
- Hyperparameter configuration
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path

from src.aai501_diabetes.config import RANDOM_STATE, MODELS_DIR

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_mlp(
    input_dim: int,
    hidden_layers: list = [128, 64, 32],
    dropout_rate: float = 0.3,
    activation: str = "relu",
    output_activation: str = "sigmoid",
) -> keras.Model:
    """
    Build a Multilayer Perceptron model.

    Args:
        input_dim: Number of input features
        hidden_layers: List of neurons per hidden layer
        dropout_rate: Dropout rate for regularization
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer

    Returns:
        keras.Model: Compiled MLP model
    """
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden layers
    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation=activation))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.BatchNormalization())

    # Output layer
    model.add(layers.Dense(1, activation=output_activation))

    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
) -> keras.Model:
    """
    Compile the MLP model.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name or instance

    Returns:
        keras.Model: Compiled model
    """
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def create_callbacks(
    model_name: str,
    patience: int = 10,
    monitor: str = "val_loss",
) -> list:
    """
    Create training callbacks.

    Args:
        model_name: Name for saving checkpoints
        patience: Early stopping patience
        monitor: Metric to monitor

    Returns:
        list: List of callbacks
    """
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks_list


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_config: dict = None,
    epochs: int = 100,
    batch_size: int = 256,
    class_weight: dict = None,
    verbose: int = 1,
) -> tuple:
    """
    Train MLP model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_config: Model configuration dict
        epochs: Maximum number of epochs
        batch_size: Batch size
        class_weight: Class weights dict
        verbose: Verbosity level

    Returns:
        tuple: (trained_model, training_history)
    """
    if model_config is None:
        model_config = {
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
        }

    # Build and compile model
    model = build_mlp(
        input_dim=X_train.shape[1],
        hidden_layers=model_config.get("hidden_layers", [128, 64, 32]),
        dropout_rate=model_config.get("dropout_rate", 0.3),
    )
    model = compile_model(
        model,
        learning_rate=model_config.get("learning_rate", 0.001),
    )

    print("\n" + "=" * 80)
    print("MLP MODEL ARCHITECTURE")
    print("=" * 80)
    model.summary()

    # Create callbacks
    callback_list = create_callbacks("mlp_diabetes")

    # Train model
    print("\n" + "=" * 80)
    print("TRAINING MLP MODEL")
    print("=" * 80)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        class_weight=class_weight,
        verbose=verbose,
    )

    # Save model
    model_path = MODELS_DIR / "mlp_diabetes_model.h5"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    return model, history

