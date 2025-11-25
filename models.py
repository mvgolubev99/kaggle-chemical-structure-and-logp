"""
TensorFlow-based MLP Regressor with scikit-learn compatibility and GPU detection.
"""

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TFMLPRegressor(BaseEstimator, RegressorMixin):
    """
    A Multi-Layer Perceptron regressor implemented in TensorFlow with scikit-learn API compatibility.
    
    This wrapper automatically detects and uses available GPUs, with fallback to CPU.
    For optimal GPU performance on local machines, ensure proper CUDA toolkit installation.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        The number of neurons in each hidden layer.
        
    activation : str, default='relu'
        Activation function for hidden layers. Options: 'relu', 'tanh', 'sigmoid', etc.
        
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.
        
    epochs : int, default=50  # Reduced for faster experimentation
        Number of training epochs.
        
    batch_size : int, default=256  # Increased for better GPU utilization
        Batch size for training.
        
    validation_split : float, default=0.1
        Fraction of training data to use for validation.
        
    early_stopping : bool, default=True  # Enabled by default for better generalization
        Whether to use early stopping during training.
        
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
        
    random_state : int, default=42
        Random seed for reproducibility.
        
    verbose : int, default=0
        Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
        
    use_gpu : bool, default=None
        Force GPU usage if available. If None, uses automatic detection.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
                 epochs=50, batch_size=256, validation_split=0.1, early_stopping=True,
                 patience=10, random_state=42, verbose=0, use_gpu=None):
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
        self.use_gpu = use_gpu
        
        # Set random seeds for reproducibility
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # GPU configuration
        self._configure_gpu()
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus and (self.use_gpu is not False):
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: Using {len(gpus)} GPU(s)")
                self.device_ = '/GPU:0'
            except RuntimeError as e:
                print(f"GPU configuration failed: {e}. Falling back to CPU.")
                self.device_ = '/CPU:0'
        else:
            print("Using CPU for computation")
            self.device_ = '/CPU:0'
    
    def _build_model(self, n_features):
        """Build the TensorFlow MLP model architecture."""
        with tf.device(self.device_):
            model = tf.keras.Sequential()
            
            # Input layer
            model.add(tf.keras.layers.Input(shape=(n_features,)))
            
            # Hidden layers with batch normalization for faster convergence
            for i, units in enumerate(self.hidden_layer_sizes):
                model.add(tf.keras.layers.Dense(
                    units, 
                    activation=self.activation,
                    kernel_initializer='he_normal'  # Better for ReLU
                ))
                # Add batch normalization for faster training
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(0.1))  # Small dropout for regularization
            
            # Output layer (linear activation for regression)
            model.add(tf.keras.layers.Dense(1, activation='linear'))
            
            # Compile model with optimized settings
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
        
        # Store information about the data
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        # Adjust batch size if dataset is too small
        effective_batch_size = min(self.batch_size, self.n_samples_)
        
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
        
        # Reduce learning rate on plateau for better convergence
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.patience // 2,
            min_lr=1e-6,
            verbose=self.verbose
        )
        callbacks.append(reduce_lr)
        
        # Build and train model
        self.model_ = self._build_model(self.n_features_in_)
        
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
        
        return self
    
    def predict(self, X):
        """
        Predict using the trained MLP model.
        
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
        
        # Make predictions
        with tf.device(self.device_):
            y_pred = self.model_.predict(X, verbose=0, batch_size=self.batch_size)
        
        # Flatten to 1D array for compatibility with sklearn
        return y_pred.flatten()
    
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
            'verbose': self.verbose,
            'use_gpu': self.use_gpu
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# Simple CPU-only version as fallback
class SimpleTFMLPRegressor(TFMLPRegressor):
    """
    Simplified version that forces CPU usage and uses smaller default batch size.
    Better for small datasets and when GPU setup is problematic.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
                 epochs=50, batch_size=128, validation_split=0.1, early_stopping=True,
                 patience=10, random_state=42, verbose=0):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping=early_stopping,
            patience=patience,
            random_state=random_state,
            verbose=verbose,
            use_gpu=False  # Force CPU
        )