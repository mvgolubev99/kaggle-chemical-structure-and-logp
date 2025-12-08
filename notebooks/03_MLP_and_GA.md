# Paths and imports handling

```python
import sys
sys.path.insert(0, "..")

from src.path_handling import resolve_path
```

# Import necessary libraries

```python
# Import necessary libraries
import os
import sys
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem

from src.chemdata.fingerprints import AvalonFingerprint
from src.chemdata.splits import load_split_info
from src.models.torch_mlp import MLP, Pipeline, run_mlp_hp_search
from src.models.metrics import (
        collect_latents, 
        get_mean_cov_scipy, 
        mahalanobis_batch, 
        cvm_test_chi2, 
        chi2_threshold, 
        rejection_rate_scipy, 
        plot_mahalanobis_distributions,
        parity_plot,
    )
from src.ga.ga_engine import run_ga, predict_logp, plot_logp_comparison
```

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
```

# Set seeds

```python
# Set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
L.seed_everything(42)
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
from rdkit.Chem.Crippen import MolLogP

def get_rdkit_logp(smi):
    mol = Chem.MolFromSmiles(smi)
    rdkit_logp = MolLogP(mol)
    return rdkit_logp

data["logp"] = data["smiles"].apply(get_rdkit_logp).values
```

# Convert to Avalon fingerprints

```python
# Convert to Avalon fingerprints
print("Converting SMILES to Avalon fingerprints...")
X_fp = AvalonFingerprint(smiles=data["smiles"]).to_fps()
X_fp = torch.tensor(X_fp, dtype=torch.float32)
```

# Load splits

```python
# Load splits
split_info = load_split_info(
    resolve_path("./data/random_split_info.json")
)
print(f"Train: {len(split_info['train'])}, Val: {len(split_info['val'])}, Test: {len(split_info['test'])}")
```

# Prepare data

```python
# Prepare data
y = torch.tensor(data["logp"].values, dtype=torch.float32)
```

# Normalize target

```python
# Normalize target
y_train_mean = y[split_info["train"]].mean()
y_train_std = y[split_info["train"]].std()
eps = 1e-8

transform = lambda y: (y - y_train_mean) / (y_train_std + eps)
inv_transform = lambda y_scaled: y_scaled * (y_train_std + eps) + y_train_mean
```

# Create datasets

```python
# Create datasets
datasets = {
    "train": TensorDataset(X_fp[split_info["train"]], transform(y[split_info["train"]])),
    "val": TensorDataset(X_fp[split_info["val"]], transform(y[split_info["val"]])),
    "test": TensorDataset(X_fp[split_info["test"]], transform(y[split_info["test"]])),
}

train_dataloader = DataLoader(datasets["train"], batch_size=64, shuffle=True)
val_dataloader = DataLoader(datasets["val"], batch_size=256, shuffle=False)
test_dataloader = DataLoader(datasets["test"], batch_size=256, shuffle=False)
```

# Run hyperparameter search


Here we run hyperparameter search for MLPregressor with such architecture: 

```python
from torchsummary import summary

summary(MLP().to(DEVICE), input_size=(1024,), batch_size=64, device=DEVICE)
```

<!-- #region -->
Hyperparameters are:  

for MLP model itself:  
dropout_prob: [0.0, 0.1, 0.2], float  

and for optimizer (adam):  
weight_decay: [1e-6, 2.0], float, log scale  
learning_rate: [1e-4, 1e-2], float, log scale


For each run best model (monitoring by MAE metric on validation) is saved and used for comparison.

Seed is fixed before every run of neural network learning.
<!-- #endregion -->

(this will take a while...)

```python
# Run hyperparameter search
# uncomment if you can wait > 50 minutes for results

# print("Running hyperparameter search...")
# mlp_hp_search_results = run_mlp_hp_search(
    # train_dataloader=train_dataloader,
    # val_dataloader=val_dataloader,
    # inv_transform_y=inv_transform,
    # n_trials=64,
    # max_epochs=50,
    # study_name="mlp_hp_search",
# )
```

```python
# in case run_mlp_hp_search was n0t executed

if "mlp_hp_search_results" not in globals():
    # these results were obtained on my laptop with gpu gtx1650
    mlp_hp_search_results = {
        'study_name': 'mlp_hp_search',
        'best_params': {
            'dropout_prob': 0.0,
            'weight_decay': 0.00012314191193621172,
            'learning_rate': 0.005268912608192128,
        },
        'best_score': 0.17984221875667572,
    }
    
```

```python
print("Best parameters:", mlp_hp_search_results["best_params"])
```

# Train final model

```python
# Train final model
print("Training final model...")
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

logger = TensorBoardLogger(resolve_path("./tb_logs"), name="best_MLP_model")
checkpoint_callback = ModelCheckpoint(
    monitor="val_mae",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="{epoch}-{val_mae:.4f}",
)

pipeline = Pipeline(
    model_cls=MLP,
    inv_transform_y=inv_transform,
    **mlp_hp_search_results["best_params"]
)

trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback],
)

trainer.fit(pipeline, train_dataloader, val_dataloader)
```

```python
from src.models.metrics import get_curves_and_metrics_from_tb_logs

get_curves_and_metrics_from_tb_logs(LOG_DIR=resolve_path("./tb_logs/best_MLP_model"))
```

# Load best model as pipeline for using encoder

```python
# load best model as pipeline for using encoder
best_pipeline = Pipeline.load_from_checkpoint(
    checkpoint_path=checkpoint_callback.best_model_path,
    model_cls=MLP,
    inv_transform_y=inv_transform,
)
best_pipeline.eval()
```

# Collect train latents

```python
# Collect latents
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_latents = collect_latents(best_pipeline, train_dataloader, device)
```

# Calculate statistics on train in latent space

```python
# Calculate statistics
mean, cov, cov_inv = get_mean_cov_scipy(train_latents)
df = train_latents.shape[1]
```

# Calculate Mahalanobis distances for train in latent space

```python
# Mahalanobis distances
train_md = mahalanobis_batch(train_latents, mean, cov_inv)
```

```python
train_md.mean()
```

# CvM test for normality of latent space

```python
# CvM test
stat, pvalue, passed = cvm_test_chi2(train_md, df)
print(f"CvM statistic: {stat}")
print(f"p-value: {pvalue}")
print(f"Normality passed: {passed}")
```

# Threshold for applicability domain

```python
# Threshold
if passed:
    threshold = chi2_threshold(df, alpha=0.95)
else:
    threshold = np.percentile(train_md, 95)

print(f"Threshold: {threshold:.4f}")
```

# Rejection rate on validation

```python
val_rej = rejection_rate_scipy(
    pipeline=best_pipeline, 
    val_dataloader=val_dataloader, 
    mean=mean, 
    cov_inv=cov_inv, 
    threshold=threshold,
    device=device,
)

print(f"Rejection rate on validation: {val_rej:.4f}")
```

# Evaluate model on test

```python
# Evaluate model on test
print("Evaluating model...")
test_results = trainer.test(ckpt_path="best", dataloaders=test_dataloader)
```

```python
print(f"Test MAE: {test_results[0]['test_mae']}")
```

# Rejection rate on test

```python
test_rej = rejection_rate_scipy(
    pipeline=best_pipeline, 
    val_dataloader=test_dataloader, 
    mean=mean, 
    cov_inv=cov_inv, 
    threshold=threshold,
    device=device,
)

print(f"Rejection rate on test: {test_rej:.4f}")
```

# Plot mahalonobis distributions for train, val, test

```python
val_latents = collect_latents(best_pipeline, val_dataloader, device)
val_md = mahalanobis_batch(val_latents, mean, cov_inv)

test_latents = collect_latents(best_pipeline, test_dataloader, device)
test_md = mahalanobis_batch(test_latents, mean, cov_inv)
```

```python
plot_mahalanobis_distributions(train_md=train_md, val_md=val_md, test_md=test_md, log_scale=True)
```

# Run Genetic algorithm for maximizing LogP 

```python
# sample initial population
random.seed(42)
initial_population = random.sample(
    data["smiles"].iloc[split_info["train"]].tolist(),
    2000,
)
```

```python
# Calculate thresholds
d95 = threshold
k = 8 # adjusted_fitness(d_threshold) = fitness / k
alpha = np.log(k) / (d95**2)

print(f"95th percentile distance: {d95}")
print(f"Alpha parameter: {alpha}")
```

```python
# run genetic algorithm
random.seed(42)
history, final_population = run_ga(
    init_smiles_list=initial_population,
    pipeline=best_pipeline,
    inv_transform=inv_transform,
    mean=mean,
    cov_inv=cov_inv,
    d95=d95,
    alpha=alpha,
    device=device,
    pop_size=2000,
    n_generations=100,
    mutants_per_parent=5,
    elite_frac=0.2,
    hard_ad_cut_every=20,
    maximize=True,
)
```

```python
# how much smiles from initial population survived in final population
intersection_smiles = list(
    set(final_population).intersection(
        set(data["smiles"].to_list())
    )
)

print(f"size of final_population: {len(final_population)}")
print(f"number of shared smiles in initial and final population: {len(intersection_smiles)}")
print("\nshared smiles:",*intersection_smiles,sep="\n")
```

```python
final_population = list(
    filter(lambda smi: smi not in intersection_smiles, final_population)
)

print(f"initial population smiles removed from final population.")
print(f"size of final population after filtration: {len(final_population)}")
```

```python
# calculate logp
initial_logp = []
for smi in initial_population:
    pred = predict_logp(smi, best_pipeline, inv_transform, device)
    if pred is not None:
        initial_logp.append(pred)

final_logp = []
for smi in final_population:
    pred = predict_logp(smi, best_pipeline, inv_transform, device)
    if pred is not None:
        final_logp.append(pred)
```

```python
plot_logp_comparison(initial_logp, final_logp)
```

```python
from IPython.display import display
from rdkit.Chem.Crippen import MolLogP

random.seed(42)
idx_final_population_smiles_to_draw = random.sample(
    [idx for idx in range(len(final_population))],
    8,
)

for idx in idx_final_population_smiles_to_draw:
    smi = final_population[idx]
    m = Chem.MolFromSmiles(smi)
    mol_img = Chem.Draw.MolToImage(m)

    rdkit_logp = MolLogP(m)

    print(f"SMILES: {smi}")
    print(f"Predicted LogP: {final_logp[idx]:.2f}")
    print(f"Crippen rdkit LogP: {rdkit_logp:.2f}")

    display(mol_img)
```

```python
final_logp_rdkit = [MolLogP(Chem.MolFromSmiles(smi)) for smi in final_population]
```

```python
parity_plot(
    y_1=np.array(final_logp), 
    y_2=np.array(final_logp_rdkit), 
    title="predicted LogP \nvs\npredicted using rdkit.Chem.Crippen.MolLogP\non final population",
    xlabel="predicted LogP",
    ylabel="predicted with rdkit",
)
```

# Conclusion


We have tuned and fitted MLP regressor for predicting logp (which was calculated using rdkit.Chem.Crippen.MolLogP).

After that, applicability domain threshold for mahalonobis distance in latent space was established for MLP regressor based on train subset.

Both val and test subsets are generally inside applicability domain of MLP model (with rejection rate ~ 0.05).
Though train latent space didnt passed normality test which may be due to sparsity of fingerprints. We can see that distribution of machalonobis distances is very narrow.

After running genetic algorithm we can see distribution shift for logp predicted by MLP regressor.

In addition to that, we can look at random examples from final population and can see there generally oleophillic molecules. Definitely our model learned that length of carbon chains correlates with LogP.

Though, parity plot between MLP Regressor predictions and rdkit.Chem.Crippen.MolLogP predictions on final population display poor consistency. But main trend can be recognized.   
