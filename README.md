### **README.md (AI generated for now)**


# Molecular LogP Prediction and Genetic Optimization Framework

This project provides a complete pipeline for:

- **Preparing molecular datasets** (SMILES → fingerprints)
- **Training machine-learning models** to predict LogP  
  (with focus on *LightGBM* and *PyTorch MLP*)
- **Estimating model applicability domain** using latent-space Mahalanobis distance
- **Running a genetic algorithm (GA)** to design molecules with increased LogP
- **Analyzing results** through plots and metrics

The project is fully reproducible and organized into modular Python packages (`chemdata`, `models`, `ga`), supported by Jupyter notebooks with end-to-end workflows.

---

## Project Motivation

The goal is to create a practical and educational pipeline for QSAR-style molecular modeling:

1. Load and analyze a LogP dataset  
2. Convert molecules to multiple fingerprint types  
3. Compare different representations via LightGBM hyperparameter search  
4. Train an MLP regressor and infer a latent feature space  
5. Quantify **applicability domain (AD)** using Mahalanobis distances  
6. Apply a **SELFIES-based genetic algorithm** to optimize molecules toward high LogP  
7. Visualize molecular evolution and prediction behavior

---

## Project Structure



.
├── data/                           ← dataset and split files
├── notebooks/                      ← step-by-step tutorials (md versions included)
│   ├── 01_Data_manipulations.*
│   ├── 02_LightGBM.*
│   └── 03_MLP_and_GA.*
├── src/
│   ├── chemdata/                   ← fingerprints + splitting utilities
│   ├── models/                     ← LightGBM search, MLP, metrics
│   ├── ga/                         ← Genetic algorithm engine
│   └── path_handling.py
├── tb_logs/                        ← TensorBoard logs for MLP training
├── tests/
│   └── test_mlp_forward.py
├── requirements.txt
└── README.md                       ← this file

````

---

## Downloading the Dataset

To reproduce the data used in the project, download the *Chemical Structure and LogP* dataset from Kaggle:

**Run the following commands in project root:**

```bash
mkdir -p ./data
````

```bash
curl -L -o ./data/archive_logp.zip \
https://www.kaggle.com/api/v1/datasets/download/matthewmasters/chemical-structure-and-logp
```

```bash
unzip ./data/archive_logp.zip -d ./data/
```

```bash
rm ./data/archive_logp.zip
```

After extraction, the expected file is:

```
./data/logP_dataset.csv
```

The notebooks automatically load this file.

---

## Installation

### 1. Create and activate environment

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure RDKit is available in your environment.
If installing RDKit via pip is not possible on your system, use conda:

```bash
conda install -c conda-forge rdkit
```

---

## Notebooks Overview

### **01_Data_manipulations**

* Loads and analyzes dataset (duplicates, atoms, Murcko scaffolds)
* Replaces dataset LogP with **RDKit Crippen MolLogP** due to inconsistencies
* Creates a **random train/val/test split**
* Computes and compares multiple fingerprints:

  * Morgan (standard & features)
  * MACCS keys
  * RDKit FP
  * AtomPair
  * Topological torsion
  * Avalon FP

---

### **02_LightGBM**

* Converts SMILES into all defined fingerprints
* Runs **Optuna hyperparameter search** for LightGBM on each fingerprint type
* Selects best fingerprint/model based on bootstrap-averaged MAE
* Visualizes prediction errors
* Conclusion:

  * *MACCS keys* gave the best test MAE
  * *Avalon fingerprints* selected for downstream molecular generation (more structural)

---

### **03_MLP_and_GA**

* Converts data into **Avalon fingerprints**
* Trains an MLP regressor with Optuna tuning (dropout, LR, weight decay)
* Tracks training with **TensorBoard**
* Computes latent vectors and fits:

  * mean vector
  * covariance
  * Mahalanobis distances
  * normality tests (CvM)
  * AD threshold (chi-square or empirical)
* Evaluates AD coverage on val/test sets
* Runs a **Genetic Algorithm** (SELFIES mutations) to maximize LogP:

  * AD-penalized fitness function
  * Hard/domain-based cutoffs
  * Population evolution & molecule filtering
  * Visualization of LogP shift and example molecules
* Shows distribution shift toward oleophilic molecules

---

## How to Run the Project

### Step 1 — Download the dataset

(See instructions above)

### Step 2 — Open the notebooks (recommended)

```bash
jupyter lab
```

Run notebooks in order:

1. `01_Data_manipulations`
2. `02_LightGBM`
3. `03_MLP_and_GA`

### Step 3 — Use the Python modules directly

Example: converting fingerprints

```python
from src.chemdata.fingerprints import AvalonFingerprint

fps = AvalonFingerprint(smiles_list).to_fps()
```

Training models or running GA can also be done programmatically.

---

## Tests

A minimal test is provided:

```
tests/test_mlp_forward.py
```

Run tests:

```bash
pytest
```

---

## Notes

* Original dataset LogP values were inconsistent with reputable computational estimates.
  Throughout the project **RDKit Crippen MolLogP** is used as the actual regression target.
* This is an **educational project**, focusing on:

  * model comparison,
  * latent space analysis,
  * ML-powered molecular design.

---

## References

* RDKit: [https://www.rdkit.org/](https://www.rdkit.org/)
* SELFIES: [https://github.com/aspuru-guzik-group/selfies](https://github.com/aspuru-guzik-group/selfies)
* LightGBM: [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
* Optuna: [https://optuna.org/](https://optuna.org/)
