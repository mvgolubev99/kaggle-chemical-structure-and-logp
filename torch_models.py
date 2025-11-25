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
from torch.utils.data import DataLoader, TensorDataset


class PyTorchMLP(nn.Module):
    """
    Simple MLP model for regression.
    """
    def __init__(self, input_size, hidden_sizes=(100,), activation='relu', dropout_rate=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
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
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                activation_fn,
                nn.Dropout(dropout_rate)
            ])
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
                 learning_rate=0.001, dropout_rate=0.1, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = PyTorchMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
            patience=5,
            # verbose=True
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
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
                 epochs=50, batch_size=256, dropout_rate=0.1, weight_decay=0.0,
                 early_stopping_patience=10, random_state=42, verbose=0):
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
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
        
        self.model_ = None
        self.trainer_ = None
        self.n_features_in_ = None
    
    def _create_data_loaders(self, X, y, validation_split=0.1):
        """Create training and validation data loaders."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split for validation if needed
        if validation_split > 0:
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size,
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
    
    def fit(self, X, y, validation_split=0.1):
        """
        Fit the model to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target values.
            
        validation_split : float, default=0.1
            Fraction of training data to use for validation.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Input validation
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
        
        # Store data information
        self.n_features_in_ = X.shape[1]
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X, y, validation_split)
        
        # Initialize model if not already done
        if self.model_ is None:
            self.model_ = LitMLPRegressor(
                input_size=self.n_features_in_,
                hidden_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate=self.learning_rate,
                dropout_rate=self.dropout_rate,
                weight_decay=self.weight_decay
            )
        
        # Configure callbacks
        callbacks = []
        
        # Early stopping
        if validation_split > 0 and self.early_stopping_patience > 0:
            early_stop = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                mode='min',
                verbose=bool(self.verbose)
            )
            callbacks.append(early_stop)
        
        # Model checkpointing
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val_loss' if validation_split > 0 else 'train_loss',
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
            deterministic=True,  # For reproducibility
            # Lightning automatically handles devices!
            # It will use GPU if available, CPU if not
        )
        
        # Train the model
        if val_loader is not None:
            self.trainer_.fit(self.model_, train_loader, val_loader)
        else:
            self.trainer_.fit(self.model_, train_loader)
        
        # Load best model checkpoint
        if checkpoint.best_model_path:
            self.model_ = LitMLPRegressor.load_from_checkpoint(
                checkpoint.best_model_path
            )
        
        return self
    
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
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Set model to evaluation mode
        self.model_.eval()
        
        # Move model to same device as input (handled by Lightning)
        device = next(self.model_.parameters()).device
        X_tensor = X_tensor.to(device)
        
        # Make predictions without gradient computation
        with torch.no_grad():
            predictions = self.model_(X_tensor)
        
        # Convert back to numpy
        if device.type == 'cuda':
            predictions = predictions.cpu()
        
        return predictions.numpy()
    
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
            'verbose': self.verbose
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        # Reset model if architecture parameters change
        if any(param in parameters for param in [
            'hidden_layer_sizes', 'activation', 'learning_rate', 
            'dropout_rate', 'weight_decay'
        ]):
            self.model_ = None
            self.trainer_ = None
            
        return self


# Fast version for hyperparameter search
class FastPLMLPRegressor(PLMLPRegressor):
    """
    Faster version for hyperparameter search with reduced epochs and simpler architecture.
    """
    def __init__(self, hidden_layer_sizes=(64,), activation='relu', learning_rate=0.001,
                 epochs=30, batch_size=512, dropout_rate=0.0, weight_decay=0.0,
                 early_stopping_patience=5, random_state=42, verbose=0):
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
            verbose=verbose
        )