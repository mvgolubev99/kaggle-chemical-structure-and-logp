"""
PyTorch Lightning-based MLP Regressor with scikit-learn compatibility.
Provides better device handling and avoids common TensorFlow pitfalls.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset


class PyTorchMLP(nn.Module):
    """
    Simple MLP model for regression.
    """
    def __init__(self, input_size, hidden_sizes=(100,), activation='relu', dropout_rate=0.1, use_batchnorm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.use_batchnorm = use_batchnorm
        
        # Activation function choice
        activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }.get(activation, nn.ReLU())
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Add BatchNorm only if requested and for layers with more than 1 neuron
            if self.use_batchnorm and hidden_size > 1:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(activation_fn)
            
            # Only add dropout if rate > 0
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class LitMLPRegressor(pl.LightningModule):
    """
    PyTorch Lightning module for MLP regression.
    Handles all training logic automatically.
    """
    def __init__(self, input_size, hidden_sizes=(100,), activation='relu', 
                 learning_rate=0.001, dropout_rate=0.1, weight_decay=0.0,
                 use_batchnorm=True, gradient_clip_val=0.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = PyTorchMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5
            # verbose parameter removed in newer PyTorch versions
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class PLMLPRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for PyTorch Lightning MLP Regressor.
    
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
        Batch size for training.
        
    dropout_rate : float, default=0.1
        Dropout rate for regularization.
        
    weight_decay : float, default=0.0
        L2 regularization strength.
        
    early_stopping_patience : int, default=10
        Patience for early stopping.
        
    random_state : int, default=42
        Random seed for reproducibility.
        
    verbose : int, default=0
        Verbosity mode.
        
    use_batchnorm : bool, default=True
        Whether to use batch normalization.
        
    gradient_clip_val : float, default=0.0
        Gradient clipping value. If 0, no clipping.
        
    validation_split : float, default=0.1
        Fraction of training data to use for validation.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
                 epochs=50, batch_size=256, dropout_rate=0.1, weight_decay=0.0,
                 early_stopping_patience=10, random_state=42, verbose=0,
                 use_batchnorm=True, gradient_clip_val=0.0, validation_split=0.1):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.gradient_clip_val = gradient_clip_val
        self.validation_split = validation_split
        
        # Set random seeds
        self._set_random_seeds()
        
        self.model_ = None
        self.trainer_ = None
        self.n_features_in_ = None
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
    
    def _create_data_loaders(self, X, y):
        """Create training and validation data loaders."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split for validation if needed
        if self.validation_split > 0:
            val_size = int(len(dataset) * self.validation_split)
            train_size = len(dataset) - val_size
            
            # Use generator for reproducibility
            generator = torch.Generator().manual_seed(self.random_state)
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=generator
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                generator=generator,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size,
                generator=generator,
                num_workers=0
            )
            
            return train_loader, val_loader
        else:
            train_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0
            )
            return train_loader, None
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
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
        try:
            # Input validation
            X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
            
            # Validate parameters
            if self.validation_split >= 1.0 or self.validation_split < 0:
                raise ValueError("validation_split must be between 0 and 1")
                
            # Adjust batch size if dataset is too small
            if self.batch_size > len(X):
                self.batch_size = max(1, len(X) // 2)
                if self.verbose:
                    print(f"Reduced batch_size to {self.batch_size} due to small dataset")
            
            # Store data information
            self.n_features_in_ = X.shape[1]
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(X, y)
            
            # Initialize model
            self.model_ = LitMLPRegressor(
                input_size=self.n_features_in_,
                hidden_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate=self.learning_rate,
                dropout_rate=self.dropout_rate,
                weight_decay=self.weight_decay,
                use_batchnorm=self.use_batchnorm,
                gradient_clip_val=self.gradient_clip_val
            )
            
            # Configure callbacks
            callbacks = []
            
            # Learning rate monitor for verbose mode
            if self.verbose:
                lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
                callbacks.append(lr_monitor)
            
            # Early stopping
            if self.validation_split > 0 and self.early_stopping_patience > 0:
                early_stop = pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    mode='min',
                    verbose=bool(self.verbose)
                )
                callbacks.append(early_stop)
            
            # Model checkpointing
            checkpoint = pl.callbacks.ModelCheckpoint(
                monitor='val_loss' if self.validation_split > 0 else 'train_loss',
                mode='min',
                save_top_k=1,
                verbose=bool(self.verbose)
            )
            callbacks.append(checkpoint)
            
            # Configure trainer
            self.trainer_ = pl.Trainer(
                max_epochs=self.epochs,
                callbacks=callbacks,
                enable_progress_bar=bool(self.verbose),
                enable_model_summary=bool(self.verbose),
                logger=False,
                deterministic=True,
                gradient_clip_val=self.gradient_clip_val
            )
            
            # Train the model
            if val_loader is not None:
                self.trainer_.fit(self.model_, train_loader, val_loader)
            else:
                self.trainer_.fit(self.model_, train_loader)
            
            # Load best model checkpoint if available
            if hasattr(checkpoint, 'best_model_path') and checkpoint.best_model_path:
                self.model_ = LitMLPRegressor.load_from_checkpoint(
                    checkpoint.best_model_path
                )
            
            return self
            
        except Exception as e:
            # Reset model state on failure
            self.model_ = None
            self.trainer_ = None
            raise RuntimeError(f"Training failed: {str(e)}") from e
    
    def predict(self, X):
        """
        Predict using the trained model.
        
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
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Set model to evaluation mode
        self.model_.eval()
        
        # Move model to same device as input
        device = next(self.model_.parameters()).device
        X_tensor = X_tensor.to(device)
        
        # Make predictions without gradient computation
        with torch.no_grad():
            predictions = self.model_(X_tensor)
        
        # Convert back to numpy
        if device.type == 'cuda':
            predictions = predictions.cpu()
        
        return predictions.numpy()
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        y : array-like of shape (n_samples,)
            True values for X.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'use_batchnorm': self.use_batchnorm,
            'gradient_clip_val': self.gradient_clip_val,
            'validation_split': self.validation_split
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        # Reset model if architecture parameters change
        reset_params = [
            'hidden_layer_sizes', 'activation', 'learning_rate', 
            'dropout_rate', 'weight_decay', 'use_batchnorm',
            'gradient_clip_val', 'validation_split'
        ]
        if any(param in parameters for param in reset_params):
            self.model_ = None
            self.trainer_ = None
            
        # Reset random seeds if random_state changes
        if 'random_state' in parameters:
            self._set_random_seeds()
            
        return self


class FastPLMLPRegressor(PLMLPRegressor):
    """
    Faster version for hyperparameter search with reduced epochs and simpler architecture.
    """
    def __init__(self, hidden_layer_sizes=(64,), activation='relu', learning_rate=0.001,
                 epochs=30, batch_size=512, dropout_rate=0.0, weight_decay=0.0,
                 early_stopping_patience=5, random_state=42, verbose=0,
                 use_batchnorm=False, gradient_clip_val=0.0, validation_split=0.1):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            verbose=verbose,
            use_batchnorm=use_batchnorm,
            gradient_clip_val=gradient_clip_val,
            validation_split=validation_split
        )


# Example usage and test
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Create sample dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test the regressor
    model = PLMLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        learning_rate=0.001,
        epochs=10,  # Reduced for quick testing
        batch_size=32,
        dropout_rate=0.1,
        verbose=1
    )
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate score
    score = model.score(X_test_scaled, y_test)
    print(f"R² Score: {score:.4f}")
    
    # Test fast version
    fast_model = FastPLMLPRegressor(verbose=1)
    fast_model.fit(X_train_scaled, y_train)
    fast_score = fast_model.score(X_test_scaled, y_test)
    print(f"Fast Model R² Score: {fast_score:.4f}")