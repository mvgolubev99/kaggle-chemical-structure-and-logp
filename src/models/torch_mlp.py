from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from torch.nn import functional as F
from typing import Optional, Tuple, List
from typing import Dict, Any, Optional, Tuple, Callable

from src.path_handling import resolve_path


class MLP(nn.Module):
    """Multilayer Perceptron for regression."""
    
    def __init__(self, input_size: int = 1024, hidden_layer_sizes: Optional[List[int]] = None, 
                 dropout_prob: float = 0.0):
        """
        Initialize MLP model.
        
        Args:
            input_size: Size of input features
            hidden_layer_sizes: List of hidden layer sizes. If None, uses default architecture.
            dropout_prob: Dropout probability
        """
        super().__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [128, 64, 32]

        layers = []
        in_dim = input_size

        # Build hidden layers
        for i, h in enumerate(hidden_layer_sizes[:-1]):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_dim = h

        # Add LayerNorm before the last hidden layer if there are at least 2 hidden layers
        if len(hidden_layer_sizes) >= 2:
            layers.append(nn.LayerNorm(hidden_layer_sizes[-2]))

        # Final latent layer
        latent_dim = hidden_layer_sizes[-1]
        layers.append(nn.Linear(in_dim, latent_dim))

        self.latent_embedder = nn.Sequential(*layers)

        # Output regressor
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        latent = self.latent_embedder(x)
        return self.regressor(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.latent_embedder(x)


class Pipeline(L.LightningModule):
    """Lightning pipeline for training MLP models."""
    
    def __init__(
            self,
            model_cls,
            inv_transform_y,
            input_size: int = 1024,
            hidden_layer_sizes: Optional[List[int]] = None,
            dropout_prob: float = 0.0,
            weight_decay: float = 0.0,
            learning_rate: float = 1e-3,
            random_state: int = 42,
        ):
        """
        Initialize the pipeline.
        
        Args:
            model_cls: Model class (should be MLP)
            inv_transform_y: Inverse transform function for targets
            input_size: Input feature size
            hidden_layer_sizes: Hidden layer sizes for MLP
            dropout_prob: Dropout probability
            weight_decay: Weight decay for optimizer
            learning_rate: Learning rate
            random_state: Random seed
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_cls", "inv_transform_y"])

        # SEED BEFORE EVERY MODEL INITIALIZATION
        L.seed_everything(random_state, workers=True)

        self.model = model_cls(
            input_size=input_size, 
            hidden_layer_sizes=hidden_layer_sizes,
            dropout_prob=dropout_prob
        )
        self.criterion = nn.MSELoss()
        self.inv_transform_y = inv_transform_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def _calc_mae(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate MAE with inverse transform."""
        mae = F.l1_loss(
            self.inv_transform_y(y_pred),
            self.inv_transform_y(y_true),
        )
        return mae

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_pred = self(x).squeeze(-1)
        loss = self.criterion(y_pred, y)

        # MAE
        mae = self._calc_mae(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch
        y_pred = self(x).squeeze(-1)
        loss = self.criterion(y_pred, y)

        # MAE
        mae = self._calc_mae(y_pred, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_mae": mae}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch
        y_pred = self(x).squeeze(-1)

        mae = self._calc_mae(y_pred, y)

        self.log("test_mae", mae, on_step=False, on_epoch=True)
        return mae

    def configure_optimizers(self) -> Tuple[list, list]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.model.latent_embedder(x)


def run_mlp_hp_search(
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        inv_transform_y: Callable,
        n_trials: int = 2,
        study_name: str = "unknown",
        max_epochs: int = 2, 
    ) -> Dict[str, Any]:
    """
    Run hyperparameter search for MLP.
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        n_trials: Number of Optuna trials
        study_name: Name of the study
        max_epochs: Maximum epochs per trial
        
    Returns:
        Dictionary with best parameters and score
    """
    import optuna
    from optuna.samplers import TPESampler
    import yaml
    from pathlib import Path
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    
    def objective(trial):
        params = {
            "dropout_prob": trial.suggest_categorical("dropout_prob", [0.0, 0.1, 0.2]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 2.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }

        tb_logs_path = resolve_path("./tb_logs/hp_search/")
        logger = TensorBoardLogger(tb_logs_path, name=f"{study_name}_trial_{trial.number}")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mae",       # metric to track
            mode="min",               # minimize val_loss
            save_top_k=1,             # save only the BEST model
            save_last=True,           # also save last epoch
            filename="{epoch}-{val_mae:.4f}",  # optional naming pattern
        )

        pipeline = Pipeline(model_cls=MLP, inv_transform_y=inv_transform_y, **params)

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False, # disable for hp search
            enable_model_summary=False, # disable for hp search
        )

        trainer.fit(pipeline, train_dataloader, val_dataloader, ckpt_path=get_latest_checkpoint(logger))

        val_results = trainer.validate(
            ckpt_path="best",
            dataloaders=val_dataloader,
            verbose=False, # disable for hp search
        )
        val_mae = val_results[0]['val_mae']

        return val_mae

    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        study_name=study_name,
        direction="minimize",
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    print("\n" + "-" * 64)
    print(f"study '{study_name}' finished after {n_trials} trials\n")
    print(yaml.dump({"best_params": best_params}))
    print(f"best_score: {best_score}\n")

    return {
        "study_name": study_name,
        "best_params": best_params,
        "best_score": best_score,
    }


def get_latest_checkpoint(logger) -> Optional[Path]:
    """
    Get the path to the last checkpoint of the latest previous experiment.
    
    Args:
        logger: TensorBoardLogger instance
        
    Returns:
        Path to checkpoint or None if not found
    """
    from pathlib import Path
    
    # Go up one level to the parent folder that contains all versions
    base_dir = Path(logger.log_dir).parent  # e.g., tb_logs/my_model
    if not base_dir.exists():
        return None

    # Find all version directories
    versions = [v for v in base_dir.iterdir() if v.is_dir() and v.name.startswith("version_")]
    if not versions:
        return None

    # Exclude the current version (logger.log_dir) if you want only previous experiments
    current_version = Path(logger.log_dir).name
    previous_versions = [v for v in versions if v.name != current_version]
    if not previous_versions:
        return None

    # Pick the version with the highest number
    latest_version = max(previous_versions, key=lambda x: int(x.name.split("_")[-1]))
    ckpt_path = latest_version / "checkpoints" / "last.ckpt"

    return ckpt_path if ckpt_path.exists() else None