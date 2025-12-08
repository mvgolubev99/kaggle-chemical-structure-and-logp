import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class DatasetSplits:
    """Container for dataset splits with fingerprint data and indices."""
    
    def __init__(self, X_data: Dict[str, np.ndarray], 
                 train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray,
                 y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """
        Initialize dataset splits.
        
        Args:
            X_data: Dictionary mapping fingerprint names to feature arrays
            train_ids: Training set indices
            val_ids: Validation set indices  
            test_ids: Test set indices
            y_train: Training targets
            y_val: Validation targets
            y_test: Test targets
        """
        self.X_data = X_data
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


def load_split_info(split_file: str = "./data/random_split_info.json") -> Dict[str, np.ndarray]:
    """
    Load split information from JSON file.
    
    Args:
        split_file: Path to JSON file containing split indices
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to numpy arrays
    """
    split_info_dict = json.loads(
        Path(split_file).read_text(encoding="utf-8")
    )
    return {k: np.array(v, dtype=int) for k, v in split_info_dict.items()}


def save_split_info(train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray,
                   split_file: str = "./data/random_split_info.json") -> None:
    """
    Save split information to JSON file.
    
    Args:
        train_ids: Training set indices
        val_ids: Validation set indices
        test_ids: Test set indices
        split_file: Path to save JSON file
    """
    split_info_dict = {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(), 
        "test": test_ids.tolist(),
    }
    Path(split_file).parent.mkdir(parents=True, exist_ok=True)
    Path(split_file).write_text(json.dumps(split_info_dict))


def create_random_splits(data_length: int, random_state: int = 42, 
                        test_size: float = 1/10, val_size: float = 1/9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create random train/val/test splits.
    
    Args:
        data_length: Total number of samples
        random_state: Random seed for reproducibility
        test_size: Fraction of data for test set
        val_size: Fraction of training data for validation set
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    from sklearn.model_selection import train_test_split
    
    data_ids = np.arange(data_length)
    train_ids, test_ids = train_test_split(data_ids, random_state=random_state, test_size=test_size)
    train_ids, val_ids = train_test_split(train_ids, random_state=random_state, test_size=val_size)
    
    return train_ids, val_ids, test_ids


def prepare_data_for_fp(X_data: Dict[str, np.ndarray], y_data: np.ndarray, 
                       split_info: Dict[str, np.ndarray], fp_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare data for a specific fingerprint.
    
    Args:
        X_data: Dictionary of fingerprint features
        y_data: Target values
        split_info: Dictionary with split indices
        fp_name: Name of fingerprint to use
        
    Returns:
        Tuple of (X_dict, y_dict) for train/val/test splits
    """
    X = {}
    y = {}
    for k, v in split_info.items():
        X[k] = X_data[fp_name][v].copy()
        y[k] = y_data[v].copy()
    
    return X, y