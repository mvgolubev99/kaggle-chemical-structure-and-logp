import json
import copy
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from fingerprints import FINGERPRINT_CLASSES

# Available fingerprint names
_available_fp_names: List[str] = list(FINGERPRINT_CLASSES.keys())

# Global cache:
#   "_y"        → dict with train/val/test y-arrays and ids (loaded once)
#   "<fp_name>" → dict storing only X arrays (train/val[/test])
_data_cache: Dict[str, Dict[str, np.ndarray]] = {}


def read_data(
    data_path: str,
    x_columns: List[str] = ["smiles"],
    y_column: str = "logp",
) -> pd.DataFrame:
    """
    Read CSV data into a DataFrame.
    """
    data = pd.read_csv(data_path, names=[*x_columns, y_column], header=None)
    return data


def make_splits_then_save(
        data: pd.DataFrame,
        output_path: str ="./data/split_info.json",
        val_size=None,
        test_size=None,
    ) -> None:
    """
    Make train,val,test split of data (pd.DataFrame).
    Save train,val,test indices into output_path in json format.
    """
    data_ids = data.index.to_numpy()
    train_ids, test_ids = train_test_split(data_ids, random_state=42, test_size=test_size)

    # recalculate val_size
    if isinstance(val_size, float) and isinstance(test_size, float):
        val_size = val_size / (1 - test_size)

    train_ids, val_ids = train_test_split(train_ids, random_state=42, test_size=val_size)

    split_info = {
        'train_ids': train_ids.tolist(),
        'val_ids': val_ids.tolist(),
        'test_ids': test_ids.tolist(),
    }

    output_path = Path(output_path)
    if not output_path.parent.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(split_info, f, indent=2)


def _split_data_with_saved_indices(
    data: pd.DataFrame,
    split_info_path: str = "./data/split_info.json",
    y_column: str = "logp",
) -> Tuple[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train/val/test using pre-saved indices.
    """
    split_info_path = Path(split_info_path)
    if not split_info_path.exists():
        raise FileNotFoundError(f"'{split_info_path}' does not exist")

    with open(split_info_path, "r") as f:
        split_info = json.load(f)

    train_ids = np.array(split_info['train_ids'], dtype=int)
    val_ids = np.array(split_info['val_ids'], dtype=int)
    test_ids = np.array(split_info['test_ids'], dtype=int)

    # Check for intersections
    intersections = {
        "train-val": np.intersect1d(train_ids, val_ids),
        "train-test": np.intersect1d(train_ids, test_ids),
        "val-test": np.intersect1d(val_ids, test_ids),
    }

    for name, intersect in intersections.items():
        if len(intersect) > 0:
            raise RuntimeError(
                f"{split_info_path} contains overlapping indices ({name}): {len(intersect)}"
            )

    max_idx = len(data) - 1
    for arr, name in zip([train_ids, val_ids, test_ids], ["train", "val", "test"]):
        if np.any(arr > max_idx):
            raise IndexError(f"{name}_ids contain indices out of bounds of the data")

    # Extract targets
    y_train = copy.deepcopy(data[y_column].iloc[train_ids].values)
    y_val = copy.deepcopy(data[y_column].iloc[val_ids].values)
    y_test = copy.deepcopy(data[y_column].iloc[test_ids].values)

    return (train_ids, y_train), (val_ids, y_val), (test_ids, y_test)


def _load_fp_from_file(fp_name_or_path: str, fp_base_dir: str) -> np.ndarray:
    """
    Load fingerprint array from a .npy path or by fingerprint name.
    """
    fp_path = Path(fp_name_or_path)

    if fp_path.suffix == ".npy" and fp_path.is_file():
        return np.load(fp_path)

    elif fp_name_or_path in _available_fp_names:
        fp_base_dir = Path(fp_base_dir)
        fp_paths = sorted(fp_base_dir.glob(f"*{fp_name_or_path}*X_data*.npy"))
        if not fp_paths:
            raise FileNotFoundError(f"No fingerprint file found for {fp_name_or_path}")
        return np.load(fp_paths[0])

    else:
        raise ValueError(f"{fp_name_or_path} is neither a valid .npy file nor a fingerprint name")


def _load_and_preprocess(
    fp_name_or_path: str,
    data_path: str = "./data/logP_dataset.csv",
    split_info_path: str = "./data/split_info.json",
    fp_base_dir: str = "./data",
    x_columns: List[str] = ["smiles"],
    y_column: str = "logp",
    save_data_in_cache: bool = True,
    fp_dtype: Any = float,
    return_test: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load and preprocess dataset and fingerprints.
    Caches y globally and fingerprints X per FP.
    """

    # ---------------------------------------------------------
    # 1) Load and cache y ONCE globally
    # ---------------------------------------------------------
    if "_y" not in _data_cache:
        data = read_data(data_path, x_columns=x_columns, y_column=y_column)
        (train_ids, y_train), (val_ids, y_val), (test_ids, y_test) = _split_data_with_saved_indices(
            data=data,
            split_info_path=split_info_path,
            y_column=y_column,
        )

        _data_cache["_y"] = {
            "train_ids": train_ids,
            "val_ids":   val_ids,
            "test_ids":  test_ids,
            "y_train": y_train,
            "y_val":   y_val,
            "y_test":  y_test,
        }

    # Load y data
    y_info = _data_cache["_y"]
    train_ids = y_info["train_ids"]
    val_ids   = y_info["val_ids"]
    test_ids  = y_info["test_ids"]

    y_train = y_info["y_train"]
    y_val   = y_info["y_val"]
    y_test  = y_info["y_test"]

    # ---------------------------------------------------------
    # 2) If fingerprints for this FP already cached — reuse
    # ---------------------------------------------------------
    if save_data_in_cache and fp_name_or_path in _data_cache:
        fp_cache = _data_cache[fp_name_or_path]

        X_train = fp_cache["X_train"]
        X_val   = fp_cache["X_val"]
        X_test  = fp_cache.get("X_test")

        out = {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
        }
        if return_test:
            out["test"] = (X_test, y_test)

        return out

    # ---------------------------------------------------------
    # 3) Load fingerprints from file
    # ---------------------------------------------------------
    X_fp = _load_fp_from_file(fp_name_or_path, fp_base_dir).astype(fp_dtype)

    # считаем общее количество семплов среди доступных y
    total_cached_samples = (
        len(y_info["y_train"]) +
        len(y_info["y_val"]) +
        (len(y_info["y_test"]) if "y_test" in y_info else 0)
    )

    if X_fp.shape[0] != total_cached_samples:
        raise ValueError(
            f"Number of fingerprints ({X_fp.shape[0]}) does not match "
            f"number of cached samples ({total_cached_samples})"
        )


    # Split fingerprints
    X_train = X_fp[train_ids]
    X_val   = X_fp[val_ids]
    X_test  = X_fp[test_ids]

    # ---------------------------------------------------------
    # 4) Pack output
    # ---------------------------------------------------------
    out = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
    }
    if return_test:
        out["test"] = (X_test, y_test)

    # ---------------------------------------------------------
    # 5) Cache ONLY X (fingerprints)
    # ---------------------------------------------------------
    if save_data_in_cache:
        _data_cache[fp_name_or_path] = {
            "X_train": X_train,
            "X_val":   X_val,
            "X_test":  X_test if return_test else None,
        }

    return out
