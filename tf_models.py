"""
TensorFlow-based MLP Regressor with scikit-learn compatibility and retracing fixes.
"""

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TFMLPRegressor(BaseEstimator, RegressorMixin):
    """
    A Multi-Layer Perceptron regressor implemented in TensorFlow with scikit-learn API compatibility.
    
    This implementation minimizes tf.function retracing by:
    - Building the model once and reusing it
    - Using fixed batch sizes and input signatures
    - Avoiding dynamic device context switching
    
    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        The number of neurons in each hidden layer.
        
    activation : str, default='relu'
        Activation function for hidden layers.
        
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.
        
    epochs : int, default=50
        Number of training epochs.
        
    batch_size : int, default=256
        Batch size for training and prediction. Must be fixed to avoid retracing.
        
    validation_split : float, default=0.1
        Fraction of training data to use for validation.
        
    early_stopping : bool, default=True
        Whether to use early stopping during training.
        
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
        
    random_state : int, default=42
        Random seed for reproducibility.
        
    verbose : int, default=0
        Verbosity mode.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
                 epochs=50, batch_size=256, validation_split=0.1, early_stopping=True,
                 patience=10, random_state=42, verbose=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seeds
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Initialize model as None - will be built on first fit
        self.model_ = None
        self.n_features_in_ = None
        self._is_fitted = False
    
    def _build_model(self, n_features):
        """Build the TensorFlow MLP model architecture once."""
        model = tf.keras.Sequential()
        
        # Input layer with explicit input shape
        model.add(tf.keras.layers.Input(shape=(n_features,), name='input_layer'))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layer_sizes):
            model.add(tf.keras.layers.Dense(
                units, 
                activation=self.activation,
                kernel_initializer='he_normal',
                name=f'hidden_layer_{i}'
            ))
            model.add(tf.keras.layers.BatchNormalization(name=f'batch_norm_{i}'))
            model.add(tf.keras.layers.Dropout(0.1, name=f'dropout_{i}'))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))
        
        # Compile with fixed settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X, y):
        """
        Fit the MLP model to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Input validation
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
        
        # Store data information
        self.n_features_in_ = X.shape[1]
        
        # Build model only once
        if self.model_ is None:
            self.model_ = self._build_model(self.n_features_in_)
        
        # Adjust batch size if dataset is too small
        effective_batch_size = min(self.batch_size, X.shape[0])
        
        # Build callbacks
        callbacks = []
        if self.early_stopping and self.validation_split > 0:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=self.verbose
            )
            callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(1, self.patience // 2),
            min_lr=1e-6,
            verbose=self.verbose
        )
        callbacks.append(reduce_lr)
        
        # Train the model
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=effective_batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=True
        )
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the trained MLP model with fixed batch size to avoid retracing.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        # Check if fit has been called
        check_is_fitted(self, 'model_')
        
        # Input validation
        X = check_array(X)
        
        # Use fixed batch size for prediction to avoid retracing
        # If dataset is small, use full dataset as one batch
        if X.shape[0] <= self.batch_size:
            predict_batch_size = X.shape[0]
        else:
            predict_batch_size = self.batch_size
        
        # Make predictions with fixed batch size
        y_pred = self.model_.predict(
            X, 
            batch_size=predict_batch_size,
            verbose=0
        )
        
        return y_pred.flatten()
    
    def _strided_predict(self, X, chunk_size=1000):
        """
        Alternative prediction method for large datasets that processes data in chunks
        to avoid memory issues while maintaining performance.
        """
        check_is_fitted(self, 'model_')
        X = check_array(X)
        
        predictions = []
        for i in range(0, len(X), chunk_size):
            chunk = X[i:i + chunk_size]
            chunk_pred = self.model_.predict(
                chunk, 
                batch_size=min(chunk_size, self.batch_size),
                verbose=0
            )
            predictions.append(chunk_pred.flatten())
        
        return np.concatenate(predictions)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        # If critical parameters change, we may need to rebuild the model
        if any(param in parameters for param in ['hidden_layer_sizes', 'activation', 'learning_rate']):
            self.model_ = None
            self._is_fitted = False
            
        return self


# Optimized version for hyperparameter search
class OptimizedTFMLPRegressor(TFMLPRegressor):
    """
    Optimized version specifically for hyperparameter search with reduced retracing.
    Uses simpler architecture and fixed configurations.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
                 epochs=30, batch_size=512, validation_split=0.1, early_stopping=True,
                 patience=5, random_state=42, verbose=0):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,  # Larger default for better performance
            validation_split=validation_split,
            early_stopping=early_stopping,
            patience=patience,
            random_state=random_state,
            verbose=verbose
        )
    
    def _build_model(self, n_features):
        """Simpler model without batch normalization for faster execution."""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(n_features,)))
        
        # Hidden layers (simpler without batch norm)
        for i, units in enumerate(self.hidden_layer_sizes):
            model.add(tf.keras.layers.Dense(
                units, 
                activation=self.activation,
                kernel_initializer='he_normal'
            ))
            # Remove batch normalization and dropout for speed
            # model.add(tf.keras.layers.Dropout(0.1))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model