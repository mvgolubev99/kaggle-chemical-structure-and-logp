# Paths and imports handling

```python
import sys
sys.path.insert(0, "..")

from src.path_handling import resolve_path
```

# Import necessary libraries

```python
# Import necessary libraries
import warnings

import pandas as pd

from lightgbm import LGBMRegressor

from src.chemdata.fingerprints import convert_smiles_to_fingerprints, FINGERPRINT_CLASSES
from src.chemdata.splits import load_split_info, DatasetSplits
from src.models.lgbm_search import FingerprintModelSearch, get_lgbm_params, rerun_fit_score_from_study_results
```

# Fix seed

```python
import numpy as np
import random
np.random.seed(42)
random.seed(42)
```

# Load data

```python
# Load data
data_path = resolve_path("./data/logP_dataset.csv")
data = pd.read_csv(data_path, names=["smiles", "logp"])
print(data)
```

# Replace logp with rdkit predictions


(check notebook 01 if you want to know why)

```python
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

def get_rdkit_logp(smi):
    mol = Chem.MolFromSmiles(smi)
    rdkit_logp = MolLogP(mol)
    return rdkit_logp


data["logp"] = data["smiles"].apply(get_rdkit_logp).values
```

# Load splits

```python
# Load splits
split_info = load_split_info(
    resolve_path("./data/random_split_info.json")
)
print(f"Train: {len(split_info['train'])}, Val: {len(split_info['val'])}, Test: {len(split_info['test'])}")
```

# Convert to fingerprints

```python
# Convert to fingerprints
print("Converting SMILES to fingerprints...")
X_data = convert_smiles_to_fingerprints(
    smiles=data["smiles"].tolist(),
    fingerprint_classes=FINGERPRINT_CLASSES
)
```

# Create dataset splits object

```python
# Create dataset splits object
splits = DatasetSplits(
    X_data=X_data,
    train_ids=split_info["train"],
    val_ids=split_info["val"],
    test_ids=split_info["test"],
    y_train=data["logp"].iloc[split_info["train"]].values,
    y_val=data["logp"].iloc[split_info["val"]].values,
    y_test=data["logp"].iloc[split_info["test"]].values,
)
```

# Run hyperparameter search for a single fingerprint (for demonstration)

```python
with warnings.catch_warnings():
    # Ignore useless warnings
    warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")

    # Run hyperparameter search for a single fingerprint (for demonstration)
    print("Running hyperparameter search for Morgan fingerprint...")
    searcher = FingerprintModelSearch(
        splits=splits,
        fp_name="MorganFingerprint_2048",
        model_cls=LGBMRegressor,
        get_params=get_lgbm_params,
    )
    res = searcher.run(n_trials=10)
```

# Run hyperparameter search for all fingerprints


Here we run hyperparameter search (using Optuna with TPE Sampler) for lgbm regressor on each data representation (fingerprint) and then choose the best model.

During the hyperparameter search we use only one train-val split instead of k-fold cross validation for time economy. 

Score (MAE) on validation is averaged using bootstraping technique.

parameters for lgbm regressor:

random_state: 42 \
verbose: -1 \
n_estimators: [200, 1500] int, log scale \
learning_rate: [0.01, 0.3], float, log scale \
max_depth: [3, 10], int \
min_child_weight: [2, 20], int \
subsample: [0.4, 0.9], float \
colsample_bytree: [0.4, 0.9], float \
gamma: [0, 1], float \
reg_alpha: [0, 5], float \
reg_lambda: [0, 10], float


(this may take a while)

```python
# Run hyperparameter search for all fingerprints (this may take a while)
print("Running hyperparameter search for all fingerprints...")
all_results = {}

with warnings.catch_warnings():
    # Ignore useless warnings
    warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")
    
    for k, fp_name in enumerate(FINGERPRINT_CLASSES.keys()):
        print(f"[{k+1}/{len(FINGERPRINT_CLASSES)}] {fp_name}")
        searcher = FingerprintModelSearch(
            splits=splits,
            fp_name=fp_name,
            model_cls=LGBMRegressor,
            get_params=get_lgbm_params,
        )
        res = searcher.run(n_trials=64) # n_trials
        all_results[f"LGBM_{fp_name}"] = res
```

# Analyze results

```python
# Analyze results
df_results = pd.DataFrame({
    "study_name": list(all_results.keys()),
    "best_score": [v["best_score"] for v in all_results.values()],
})
df_results.sort_values(by="best_score", inplace=True)
print(df_results)
```

```python
# Refit best model on best fingerprint
best_study_name = df_results.iloc[0]["study_name"]
print(f"Best model: {best_study_name}")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")
    
    regressor, score = rerun_fit_score_from_study_results(
        results=all_results[best_study_name],
        splits=splits,
        fit_on_subset="train",
        score_on="val"
    )

print(f"Best model validation score: {score}")
```

```python
# Refit best model on avalon fingerprint
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")
    
    regressor, score = rerun_fit_score_from_study_results(
        results=all_results["LGBM_AvalonFingerprint_1024"],
        splits=splits,
        fit_on_subset="train",
        score_on="val"
    )

print(f"Best model on avalon fingerprint validation score: {score}")
```

# Conclusion


Best quality of fit on validation is obtained with tuning lgbm regressor on **MACCS keys**.

But for **reverse design** of molecules with higher logp we will choose **avalon fingerprint** since it is generally more structural.

However, decision tree based models are hard to interpret in terms of **applicability domain (AD)** of model.

Neural networks (and specifically MLP regressor) are more viable for determination of AD if we consider output of last hidden layer as a "latent space".

Further we will tune and train MLP regressor for better evaluation of AD.



