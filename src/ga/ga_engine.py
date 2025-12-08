import selfies as sf
from rdkit import Chem

import numpy as np
import torch
import random

from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

from src.models.torch_mlp import Pipeline


def smiles_to_fp_tensor(smiles: str, device, n_bits: int = 1024):
    """
    Convert SMILES to fingerprint tensor.
    
    Args:
        smiles: SMILES string
        device: PyTorch device
        n_bits: Number of fingerprint bits
        
    Returns:
        Fingerprint tensor or None if invalid
    """
    from rdkit.Avalon import pyAvalonTools
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
    arr = np.array(fp, dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_logp(smiles: str, pipeline: Pipeline, inv_transform, device) -> Optional[float]:
    """
    Predict logP for a SMILES string.
    
    Args:
        smiles: SMILES string
        pipeline: Trained pipeline
        inv_transform: Inverse transform function
        device: PyTorch device
        
    Returns:
        Predicted logP or None if invalid
    """
    fp = smiles_to_fp_tensor(smiles, device)
    if fp is None:
        return None

    pred_scaled = pipeline(fp.unsqueeze(0)).cpu().numpy().ravel()[0]
    return float(inv_transform(pred_scaled))


def safe_encode(smiles: str) -> Optional[str]:
    """
    Safely encode SMILES to SELFIES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        SELFIES string or None if encoding fails
    """
    try:
        return sf.encoder(smiles)
    except:
        return None


@torch.no_grad()
def latent_mahalanobis(smiles: str, pipeline: Pipeline, mean: np.ndarray, 
                      cov_inv: np.ndarray, device) -> float:
    """
    Calculate Mahalanobis distance in latent space.
    
    Args:
        smiles: SMILES string
        pipeline: Trained pipeline
        mean: Mean of training latents
        cov_inv: Inverse covariance matrix
        device: PyTorch device
        
    Returns:
        Mahalanobis distance or infinity if invalid
    """
    fp = smiles_to_fp_tensor(smiles, device)
    if fp is None:
        return np.inf

    latent = pipeline.encode(fp.unsqueeze(0)).cpu().numpy().ravel()
    diff = latent - mean
    return float(diff @ cov_inv @ diff)


def adjusted_fitness(pred: float, dist: float, alpha: float) -> float:
    """
    Calculate adjusted fitness combining prediction and distance.
    
    Args:
        pred: Predicted logP
        dist: Mahalanobis distance
        alpha: Adjustment parameter
        
    Returns:
        Adjusted fitness
    """
    return pred * np.exp(-alpha * dist * dist)


SELFIES_ALPHABET = list(sf.get_semantic_robust_alphabet())


def mutate_selfies(selfies_string: str, num_changes: int = 1) -> str:
    """
    Mutate a SELFIES string.
    
    Args:
        selfies_string: Input SELFIES string
        num_changes: Number of mutations
        
    Returns:
        Mutated SELFIES string
    """
    tokens = list(sf.split_selfies(selfies_string))
    if len(tokens) == 0:
        return selfies_string

    for _ in range(num_changes):
        op = random.choice(["replace", "insert", "delete"])

        if op == "replace":
            idx = random.randrange(len(tokens))
            tokens[idx] = random.choice(SELFIES_ALPHABET)

        elif op == "insert":
            idx = random.randrange(len(tokens)+1)
            tokens.insert(idx, random.choice(SELFIES_ALPHABET))

        elif op == "delete" and len(tokens) > 1:
            idx = random.randrange(len(tokens))
            tokens.pop(idx)

    # Normalize through RDKit to get valid molecules
    smiles = sf.decoder("".join(tokens))
    new_s = safe_encode(smiles)
    return new_s


def is_chemically_reasonable(smiles: str) -> bool:
    """
    Check if a SMILES string represents a chemically reasonable molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if chemically reasonable
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False

    # Full RDKit sanitization (validity + valences)
    try:
        Chem.SanitizeMol(mol)
    except:
        return False

    # 1. No radicals
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            return False

    # 2. Total charge = 0
    if Chem.GetFormalCharge(mol) != 0:
        return False

    # 3. No atoms with local charge (if you want to exclude even internal ions)
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return False

    # 4. No atoms like [O-], [N+], [C-], [C+], etc.
    prohibited = {"+", "-"}
    for atom in mol.GetAtoms():
        symbol = atom.GetSmarts()
        if any(ch in symbol for ch in prohibited):
            return False

    return True


def generate_mutants(selfies_src: str, n_mutants: int) -> List[str]:
    """
    Generate mutants from a source SELFIES string.
    
    Args:
        selfies_src: Source SELFIES string
        n_mutants: Number of mutants to generate
        
    Returns:
        List of mutant SMILES strings
    """
    mutants = []
    for _ in range(n_mutants):

        r = random.random()
        if r < 0.80:
            k = 1
        elif r < 0.95:
            k = random.randint(2, 3)
        else:
            k = random.randint(4, 8)

        mutated = mutate_selfies(selfies_src, k)
        smi = sf.decoder(mutated) if mutated is not None else None

        if smi is None:
            continue

        # RDKit check + chemical filter
        if Chem.MolFromSmiles(smi) is not None and is_chemically_reasonable(smi):
            mutants.append(smi)

    return mutants


def molecule_fitness(smiles: str, pipeline: Pipeline, inv_transform, mean: np.ndarray, 
                    cov_inv: np.ndarray, device, alpha: float) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Calculate molecule fitness.
    
    Args:
        smiles: SMILES string
        pipeline: Trained pipeline
        inv_transform: Inverse transform function
        mean: Mean of training latents
        cov_inv: Inverse covariance matrix
        device: PyTorch device
        alpha: Adjustment parameter
        
    Returns:
        Tuple of (fitness, prediction, distance)
    """
    pred = predict_logp(smiles, pipeline, inv_transform, device)
    if pred is None:
        return -1e9, None, None

    dist = latent_mahalanobis(smiles, pipeline, mean, cov_inv, device)
    fit = adjusted_fitness(pred, dist, alpha)
    return fit, pred, dist


class ModelBundle:
    """Bundle of model and related parameters for GA."""
    
    def __init__(self, pipeline: Pipeline, inv_transform, mean: np.ndarray, 
                 cov_inv: np.ndarray, alpha: float, d95: float, device):
        """
        Initialize model bundle.
        
        Args:
            pipeline: Trained pipeline
            inv_transform: Inverse transform function
            mean: Mean of training latents
            cov_inv: Inverse covariance matrix
            alpha: Adjustment parameter
            d95: 95th percentile distance
            device: PyTorch device
        """
        self.pipeline = pipeline
        self.inv_transform = inv_transform
        self.mean = mean
        self.cov_inv = cov_inv
        self.alpha = alpha
        self.d95 = d95
        self.device = device

    @torch.no_grad()
    def predict_logp(self, smiles: str) -> Optional[float]:
        """Predict logP for a SMILES string."""
        fp = smiles_to_fp_tensor(smiles, self.device)
        if fp is None:
            return None
        pred_scaled = self.pipeline(fp.unsqueeze(0)).cpu().numpy().ravel()[0]
        return float(self.inv_transform(pred_scaled))

    @torch.no_grad()
    def mahalanobis(self, smiles: str) -> float:
        """Calculate Mahalanobis distance."""
        fp = smiles_to_fp_tensor(smiles, self.device)
        if fp is None:
            return np.inf
        latent = self.pipeline.encode(fp.unsqueeze(0)).cpu().numpy().ravel()
        diff = latent - self.mean
        return float(diff @ self.cov_inv @ diff)

    def fitness(self, smiles: str) -> Tuple[float, Optional[float], Optional[float]]:
        """Calculate fitness."""
        pred = self.predict_logp(smiles)
        if pred is None:
            return -1e9, None, None

        dist = self.mahalanobis(smiles)
        fit = pred * np.exp(-self.alpha * dist * dist)
        return fit, pred, dist


class GAEngine:
    """Genetic Algorithm engine."""
    
    def __init__(
        self,
        model: ModelBundle,
        pop_size: int = 300,
        mutants_per_parent: int = 5,
        elite_frac: float = 0.2,
        hard_ad_cut_every: int = 10,
        maximize: bool = True
    ):
        """
        Initialize GA engine.
        
        Args:
            model: ModelBundle instance
            pop_size: Population size
            mutants_per_parent: Number of mutants per parent
            elite_frac: Fraction of elite individuals
            hard_ad_cut_every: Hard AD cut frequency
            maximize: Whether to maximize fitness
        """
        self.model = model
        self.pop_size = pop_size
        self.mutants_per_parent = mutants_per_parent
        self.elite_frac = elite_frac
        self.hard_ad_cut_every = hard_ad_cut_every
        self.maximize = maximize

    def run(self, init_population: List[str], generations: int = 100) -> List[Tuple]:
        """
        Run GA optimization.
        
        Args:
            init_population: Initial population of SMILES
            generations: Number of generations
            
        Returns:
            History of best individuals
        """
        population = init_population[:]
        history = []

        for gen in range(generations):

            # --- score current pop ---
            scored = []
            for smi in population:
                if not is_chemically_reasonable(smi):
                    continue
                fit, pred, dist = self.model.fitness(smi)
                scored.append((fit, smi, pred, dist))

            scored.sort(key=lambda x: x[0], reverse=self.maximize)
            best = scored[0]
            history.append(best)

            print(f"\n=== GEN {gen} ===")
            print(f"Best adj={best[0]:.4f} | pred={best[2]:.3f} | d={best[3]:.2f}")
            print(f"SMILES: {best[1]}")

            # --- elite selection ---
            elite_n = max(1, int(self.elite_frac * self.pop_size))
            elites = [x[1] for x in scored[:elite_n]]

            # --- mutation ---
            new_candidates = []
            for smi in elites:
                sfie = sf.encoder(smi)
                new_candidates += generate_mutants(sfie, self.mutants_per_parent)

            # --- hard AD cut ---
            if gen % self.hard_ad_cut_every == 0:
                new_candidates = [
                    s for s in new_candidates
                    if self.model.mahalanobis(s) <= self.model.d95
                ]

            new_candidates = list(set(new_candidates))
            if not new_candidates:
                print("No valid candidates left. Terminating.")
                break

            # --- score new candidates ---
            scored_new = []
            for smi in new_candidates:
                fit, pred, dist = self.model.fitness(smi)
                scored_new.append((fit, smi, pred, dist))

            scored_new.sort(key=lambda x: x[0], reverse=self.maximize)
            population = [x[1] for x in scored_new[:self.pop_size]]

        return history


def run_ga(
    init_smiles_list: List[str],
    pipeline: Pipeline,
    inv_transform,
    mean: np.ndarray,
    cov_inv: np.ndarray,
    d95: float,
    alpha: float,
    device,
    pop_size: int = 300,
    n_generations: int = 100,
    mutants_per_parent: int = 5,
    elite_frac: float = 0.2,
    hard_ad_cut_every: int = 10,
    maximize: bool = True
) -> Tuple[List[Tuple], List[str]]:
    """
    Run GA optimization.
    
    Args:
        init_smiles_list: Initial population
        pipeline: Trained pipeline
        inv_transform: Inverse transform function
        mean: Mean of training latents
        cov_inv: Inverse covariance matrix
        d95: 95th percentile distance
        alpha: Adjustment parameter
        device: PyTorch device
        pop_size: Population size
        n_generations: Number of generations
        mutants_per_parent: Mutants per parent
        elite_frac: Elite fraction
        hard_ad_cut_every: Hard AD cut frequency
        maximize: Whether to maximize
        
    Returns:
        Tuple of (history, final_population)
    """
    population = init_smiles_list[:]
    history = []

    for gen in range(n_generations):

        # --- SCORE CURRENT POP ---
        scored = []
        for smi in population:
            fit, pred, dist = molecule_fitness(
                smi, pipeline, inv_transform,
                mean, cov_inv, device, alpha
            )
            scored.append((fit, smi, pred, dist))

        scored.sort(key=lambda x: x[0], reverse=maximize)

        best = scored[0]
        history.append(best)

        print(f"\n=== GEN {gen} ===")
        print(f"Best adj fitness={best[0]:.4f} | pred={best[2]:.3f} | d={best[3]:.2f}")
        print(f"SMILES: {best[1]}")

        # --- ELITE SELECTION ---
        elite_n = max(1, int(elite_frac * pop_size))
        elites = [x[1] for x in scored[:elite_n]]

        # --- MUTATION ---
        new_candidates = []
        for smi in elites:
            sfie = safe_encode(smi)
            if sfie is None:
                continue

            new_candidates += generate_mutants(sfie, mutants_per_parent)

        # --- HARD AD CUT ---
        if gen % hard_ad_cut_every == 0:
            new_candidates = [
                s for s in new_candidates
                if latent_mahalanobis(s, pipeline, mean, cov_inv, device) <= d95
            ]

        new_candidates = list(set(new_candidates))
        if len(new_candidates) == 0:
            print("No valid candidates! Stopping.")
            break

        # --- SCORE NEW CANDIDATES ---
        scored_new = []
        for smi in new_candidates:
            fit, pred, dist = molecule_fitness(
                smi, pipeline, inv_transform,
                mean, cov_inv, device, alpha
            )
            scored_new.append((fit, smi, pred, dist))

        scored_new.sort(key=lambda x: x[0], reverse=maximize)
        population = [x[1] for x in scored_new[:pop_size]]

    return history, population


def plot_logp_comparison(initial_logp: List[float], final_logp: List[float]) -> None:
    """
    Plot logP distribution comparison.
    
    Args:
        initial_logp: Initial population logP values
        final_logp: Final population logP values
    """
    plt.figure(figsize=(10, 5))
    plt.hist(initial_logp, bins=40, alpha=0.6, label="Initial population", density=True)
    plt.hist(final_logp, bins=40, alpha=0.6, label="Final population", density=True)
    plt.xlabel("logP")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribution of predicted logP\nInitial vs Final GA population")
    plt.show()