import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from scipy.spatial.distance import mahalanobis
from scipy.stats import cramervonmises, chi2


def collect_latents(pipeline, dataloader, device) -> np.ndarray:
    """
    Collect latent representations from a pipeline.
    
    Args:
        pipeline: Trained pipeline with encode method
        dataloader: DataLoader for the data
        device: Device to run inference on
        
    Returns:
        Array of latent representations
    """
    pipeline.eval()
    latents = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            latents.append(pipeline.encode(x).cpu().numpy())
    return np.vstack(latents)


def get_mean_cov_scipy(latents: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate mean, covariance, and inverse covariance of latent space.
    
    Args:
        latents: Array of latent representations
        eps: Regularization epsilon
        
    Returns:
        Tuple of (mean, cov, cov_inv)
    """
    mean = latents.mean(axis=0)
    cov = np.cov(latents - mean, rowvar=False)
    cov += eps * np.eye(cov.shape[0])  # regularization
    cov_inv = np.linalg.inv(cov)
    return mean, cov, cov_inv


def mahalanobis_batch(latents: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """
    Calculate Mahalanobis distances for a batch of latent vectors.
    
    Args:
        latents: Array of latent representations
        mean: Mean of training latents
        cov_inv: Inverse covariance matrix
        
    Returns:
        Array of squared Mahalanobis distances
    """
    return np.array([
        mahalanobis(x, mean, cov_inv)**2   # squared distance
        for x in latents
    ])


def cvm_test_chi2(distances: np.ndarray, df: int, alpha: float = 0.95) -> Tuple[float, float, bool]:
    """
    Perform Cramér-von Mises test for normality against chi-squared distribution.
    
    Args:
        distances: Array of Mahalanobis distances
        df: Degrees of freedom
        alpha: Significance level
        
    Returns:
        Tuple of (statistic, p-value, passed)
    """
    res = cramervonmises(distances, chi2(df=df).cdf)
    return res.statistic, res.pvalue, (res.pvalue > (1 - alpha))


def chi2_threshold(df: int, alpha: float = 0.95) -> float:
    """
    Calculate chi-squared threshold for given degrees of freedom and alpha.
    
    Args:
        df: Degrees of freedom
        alpha: Significance level
        
    Returns:
        Chi-squared threshold
    """
    return chi2.ppf(alpha, df)


def rejection_rate_scipy(pipeline, val_dataloader, mean: np.ndarray, cov_inv: np.ndarray, 
                        threshold: float, device) -> float:
    """
    Calculate rejection rate based on Mahalanobis distance threshold.
    
    Args:
        pipeline: Trained pipeline
        val_dataloader: Validation DataLoader
        mean: Mean of training latents
        cov_inv: Inverse covariance matrix
        threshold: Distance threshold
        device: Device for inference
        
    Returns:
        Rejection rate as float
    """
    val_latents = collect_latents(pipeline, val_dataloader, device)
    distances = mahalanobis_batch(val_latents, mean, cov_inv)
    return np.mean(distances > threshold)


def plot_mahalanobis_distributions(train_md: np.ndarray, val_md: np.ndarray, test_md: np.ndarray, 
                                 log_scale: bool = True) -> None:
    """
    Plot violin plots of Mahalanobis distances.
    
    Args:
        train_md: Training Mahalanobis distances
        val_md: Validation Mahalanobis distances
        test_md: Test Mahalanobis distances
        log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(8, 5))

    sns.violinplot(
        data=[train_md, val_md, test_md],
        inner="quartile",
        palette="pastel"
    )
    plt.xticks(ticks=[0, 1, 2], labels=["train", "val", "test"])
    if log_scale:
        plt.yscale("log")
    plt.ylabel("Mahalanobis distance")
    plt.title("Violin plot of Mahalanobis distances in latent space\n($\\mu$ and $\\Sigma$ from train latent space)")
    plt.show()


def compute_logp_list(smiles_list: list, pipeline, inv_transform, device) -> list:
    """
    Compute logP values for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        pipeline: Trained pipeline
        inv_transform: Inverse transform function
        device: Device for inference
        
    Returns:
        List of predicted logP values
    """
    values = []
    for smi in smiles_list:
        try:
            from src.chemdata.fingerprints import AvalonFingerprint
            from rdkit import Chem
            from rdkit.Avalon import pyAvalonTools
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=1024)
            arr = np.array(fp, dtype=np.float32)
            fp_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
            
            pred_scaled = pipeline(fp_tensor.unsqueeze(0)).cpu().numpy().ravel()[0]
            values.append(float(inv_transform(pred_scaled)))
        except Exception as e:
            pass
    return values


def get_curves_and_metrics_from_tb_logs(LOG_DIR):
    import matplotlib.pyplot as plt
    from tbparse import SummaryReader
    from pathlib import Path

    # Find the latest version folder
    log_path = Path(LOG_DIR)
    versions = [v for v in log_path.iterdir() if v.is_dir() and v.name.startswith("version_")]
    if not versions:
        print(f"No version folders found in {LOG_DIR}")
        exit()

    # Get the latest version (highest number)
    latest_version = max(versions, key=lambda x: int(x.name.split("_")[-1]))
    print(f"Reading logs from: {latest_version}")

    # Read TensorBoard logs using tbparse
    reader = SummaryReader(str(latest_version))
    df = reader.scalars  # Get all scalar data as pandas DataFrame

    # Filter for our metrics
    train_loss = df[df['tag'] == 'train_loss']
    val_loss = df[df['tag'] == 'val_loss']
    train_mae = df[df['tag'] == 'train_mae']
    val_mae = df[df['tag'] == 'val_mae']

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Loss curves
    ax1.plot(train_loss['step'], train_loss['value'], label='Train Loss', linewidth=2)
    ax1.plot(val_loss['step'], val_loss['value'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('step')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot MAE curves
    ax2.plot(train_mae['step'], train_mae['value'], label='Train MAE', linewidth=2)
    ax2.plot(val_mae['step'], val_mae['value'], label='Val MAE', linewidth=2)
    ax2.set_xlabel('step')
    ax2.set_ylabel('MAE')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Save the plot if needed
    # plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')

    # Print final values
    print("\nFinal metrics:")
    print(f"Train Loss: {train_loss['value'].iloc[-1]:.4f}")
    print(f"Val Loss: {val_loss['value'].iloc[-1]:.4f}")
    print(f"Train MAE: {train_mae['value'].iloc[-1]:.4f}")
    print(f"Val MAE: {val_mae['value'].iloc[-1]:.4f}")


def parity_plot(
        y_1, 
        y_2, 
        title=None,
        xlabel="y_1",
        ylabel="y_2",
    ):

    from sklearn.metrics import (
            mean_absolute_error, 
            r2_score, 
            root_mean_squared_error,
        )

    line_ = [min(y_1.min(), y_2.min()), max(y_1.max(), y_2.max())]

    mae = mean_absolute_error(y_1, y_2)
    r2 = r2_score(y_1, y_2)
    rmse = root_mean_squared_error(y_1, y_2)

    metrics_output = (
            f"MAE = {mae:.4f}\n"
            f"R2 = {r2:.4f}\n"
            f"RMSE = {rmse:.4f}\n"
        )

    fig, ax = plt.subplots()
    plt.scatter(y_1, y_2)
    plt.plot(line_, line_, color = 'k', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(
        x=0.05, 
        y=0.95, 
        s=metrics_output, 
        transform=ax.transAxes,
        va='top',
    )
    plt.title(title)
    plt.show()

    print(metrics_output)