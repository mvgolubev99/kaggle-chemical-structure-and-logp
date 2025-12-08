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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.chemdata.fingerprints import convert_smiles_to_fingerprints, FINGERPRINT_CLASSES
from src.chemdata import splits
from src.models.metrics import parity_plot
```

# Load data

```python
# Load data
data_path = resolve_path("./data/logP_dataset.csv")
data = pd.read_csv(data_path, names=["smiles", "logp"])
print(f"Loaded {len(data)} compounds")
print(data)
```

# Pre-split analysis of dataset

```python
# Display basic info
print(data.info())
```

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

def smiles_from_mol_from_smiles(smiles: str) -> str:
    """Convert SMILES to molecule and back to SMILES to normalize."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def count_atoms_from_smiles(smiles: str) -> int:
    """Count number of atoms in a molecule from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()
```

```python
# Check for duplicates
number_of_duplicates = data["smiles"].apply(smiles_from_mol_from_smiles).duplicated().sum()
print(f"Number of SMILES duplicates in dataset: {number_of_duplicates}")

# Number of atoms distribution
numbers_of_atoms = data["smiles"].apply(count_atoms_from_smiles).values

print("\nnumber of atoms")
print(f"mean:\t{numbers_of_atoms.mean():.1f}")
print(f"std:\t{numbers_of_atoms.std():.1f}")
print(f"min:\t{numbers_of_atoms.min():.1f}")
print(f"max:\t{numbers_of_atoms.max():.1f}")

# Murcko Scaffolds
n_empty_scaffolds = (data['smiles'].apply(MurckoScaffoldSmilesFromSmiles) == '').sum()

print(f"\nN of empty scaffolds:\t{n_empty_scaffolds}")
print(f"% of empty scaffolds:\t{n_empty_scaffolds/len(data)*100:.2f} %")
```

# Compare logp from dataset with rdkit crippen

```python
from rdkit.Chem.Crippen import MolLogP

def get_rdkit_logp(smi):
    mol = Chem.MolFromSmiles(smi)
    rdkit_logp = MolLogP(mol)
    return rdkit_logp

y_data = data["logp"].values
y_rdkit = data["smiles"].apply(get_rdkit_logp).values
```

```python
parity_plot(
    y_1=y_data,
    y_2=y_rdkit,
    title="LogP data\nvs\npredicted using rdkit.Chem.Crippen.MolLogP",
    xlabel="y_data",
    ylabel="y_rdkit"
)
```

# Replace logp in dataset with rdkit predictions


As we can see in previous python cell, there is strong inconsistency between logp data from dataset and rdkit.Chem.Crippen.MolLogP predictions.

Since this project is just educational, we will use simple method for validation of predicts on new generated molecules. 

So we will replace logp from dataset with logp predicted using rdkit.Chem.Crippen.MolLogP 

```python
data["logp"] = y_rdkit
```

# Make/save or load random split

```python
# make random split info file if it doesnt exist
split_file = resolve_path("./data/random_split_info.json")

if not split_file.exists():

    train_ids, val_ids, test_ids = splits.create_random_splits(data_length=len(data), random_state=42)

    splits.save_split_info(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        split_file = split_file,
    )

# Load splits and plot target distributions

split_info = splits.load_split_info(split_file)
print(f"Train: {len(split_info['train'])}, Val: {len(split_info['val'])}, Test: {len(split_info['test'])}")

```

# Plot target distributions for every split

```python
def plot_target_distributions(target: pd.Series, **ids: np.ndarray) -> None:
    """
    Plot violin plots of target distribution across different subsets.
    
    Args:
        target: Target values series
        **ids: Keyword arguments mapping subset names to indices
    """
    plt.figure()
    data_vals = []
    labels = []
    for name, idx in ids.items():
        data_vals.append(target.iloc[idx].values)
        labels.append(name)

    plt.violinplot(data_vals, showmedians=True)
    plt.xticks(range(1, len(ids) + 1), labels=labels)
    plt.tight_layout()
    plt.title(f"target distribution in {', '.join(ids.keys())} subsets")
    plt.show()
```

```python
plot_target_distributions(
    target=data["logp"],
    train=split_info["train"],
    val=split_info["val"],
    test=split_info["test"]
)
```

# Convert train data to fingerprints (subsample for speed)

```python
# Convert to fingerprints (subset for speed)
n_samples = min(1000, len(data))
sample_data = data.iloc[split_info["train"]].sample(n=n_samples, random_state=42)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    X_data_train = convert_smiles_to_fingerprints(
        smiles=sample_data["smiles"].tolist(),
        fingerprint_classes=FINGERPRINT_CLASSES
    )

```

```python
# Analyze fingerprint properties
fp_stats = {}
for fp_name, fp_array in X_data_train.items():
    fp_stats[fp_name] = {
        "shape": fp_array.shape,
        "sparsity": 1 - (fp_array.sum() / fp_array.size),
        "mean_bits": fp_array.mean(axis=0).mean(),
        "std_bits": fp_array.std(axis=0).mean()
    }

print("Fingerprint Statistics on train:\n")
for name, stats in sorted(fp_stats.items(), 
                          key=lambda x: x[1]["sparsity"]): # sorted by sparsity
    print(
        f"{name}:\n"
        f"shape: {stats['shape']}\t" 
        f"Mean_bits: {stats["mean_bits"]:4f},\t"
        f"Std_bits: {stats["std_bits"]:.4f}\t"
        f"Sparsity: {stats['sparsity']:.4f},\n"
    )
```

# Conclusion


This datataset does not have duplicates or missing values.

LogP from this dataset dramatically differs from rdkit.Chem.Crippen.MolLogP predictions from the same SMILES strings.
So we decided to replace original data with these predictions.   

This dataset is represented by relatively small molecules (mean number of atoms is ~6) and most of the molecules (~90%) yield murcko scaffold with empty SMILES string. This fact makes it impossible to split data using scaffold split. Alternative to scaffold split is butina split (or any split based on clustering by tanimoto similarity of fingerprints), but it is not guaranteed to work on small molecules such in that dataset. Of course, we can find fingerprint which provides more reasonable clustering, but that means indirect data leakage since we doing it on the whole data including test subset. 

So, **random splitting** would be used for this data. 

Fortunately, target distributions in subsets after random split seem to be generally similar and with close to normal (though statistical normality tests were not performed)

Also, as we can see after random splitting of data, Morgan fingerprint (which is considered as the standard choice for butina split) is very sparse which means it probably wouldn't have provided reasonable clusterisation.
