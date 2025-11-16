from pathlib import Path

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------

class BaseFingerprintRepresenation:
    def __init__(self, smiles, *args, **kwargs):
        self.smiles = self._collect_smiles(smiles)
        self.args = args
        self.kwargs = kwargs
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]

    def _collect_smiles(self, smiles):
        if isinstance(smiles, list):
            pass
        elif hasattr(smiles, 'to_list'):
            smiles = smiles.to_list()
        else:
            smiles = list(smiles)
        return smiles

    def mol_to_fp(self, mol):
        return np.zeros(1)

    def to_fps(self):
        fps = []
        for mol in self.mols:
            fp = self.mol_to_fp(mol)
            fps.append(fp)
        self.fps = np.array(fps)
        return self.fps

    def save_fps(self, savepath):
        if not hasattr(self, 'fps'):
            raise AttributeError(
                "There is no 'fps' attribute! Nothing to save."
            )
        if not isinstance(self.fps, np.ndarray):
            raise TypeError(
                f"fps attribute is of wrong type: {type(self.fps)}, "
                "but 'np.ndarray' is expected"
            )
        if savepath is None:
            raise ValueError("path should not be None!")
        np.save(savepath, self.fps)

        if Path(savepath).is_file():
            print(f"\nfingerprints saved as \'{savepath}\'")

# ------------------------------------------------------------------
# Morgan (ECFP)
# ------------------------------------------------------------------

class MorganFingerprint(BaseFingerprintRepresenation):
    def mol_to_fp(self, mol):
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.kwargs.get('radius', 2), nBits=self.kwargs.get('nBits', 2048), useFeatures=False,)
        gen = GetMorganGenerator(
            radius=self.kwargs.get('radius', 2),
            fpSize=self.kwargs.get('fpSize', 2048),
        )
        fp = gen.GetFingerprint(mol)
        return np.array(fp, dtype=int)

# ------------------------------------------------------------------
# Features Morgan (FCFP)
# ------------------------------------------------------------------

class FeaturesMorganFingerprint(BaseFingerprintRepresenation):
    def mol_to_fp(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.kwargs.get('radius', 2),
            nBits=self.kwargs.get('nBits', 2048),
            useFeatures=True,
        )
        return np.array(fp, dtype=int)

# ------------------------------------------------------------------
# MACCS keys
# ------------------------------------------------------------------

class MACCSKeysFingerprint(BaseFingerprintRepresenation):
    def mol_to_fp(self, mol):
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=int)

# ------------------------------------------------------------------
# RDKit fingerprint
# ------------------------------------------------------------------

class RDKitFingerprint(BaseFingerprintRepresenation):
    def mol_to_fp(self, mol):
        fp = Chem.RDKFingerprint(
            mol,
            minPath=self.kwargs.get('minPath', 1),
            maxPath=self.kwargs.get('maxPath', 7),
            fpSize=self.kwargs.get('fpSize', 2048),
            useHs=self.kwargs.get('useHs', True),
            branchedPaths=self.kwargs.get('branchedPaths', True)
        )
        return np.array(fp, dtype=int)

# ------------------------------------------------------------------
# Atom Pair (hashed)
# ------------------------------------------------------------------

class AtomPairFingerprint(BaseFingerprintRepresenation):
    def mol_to_fp(self, mol):
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol,
            nBits=self.kwargs.get('nBits', 2048)
        )
        return np.array(fp, dtype=int)

# ------------------------------------------------------------------
# Topological Torsion (hashed)
# ------------------------------------------------------------------

class TopologicalTorsionFingerprint(BaseFingerprintRepresenation):
    def mol_to_fp(self, mol):
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol,
            nBits=self.kwargs.get('nBits', 2048)
        )
        return np.array(fp, dtype=int)

# ------------------------------------------------------------------
# Avalon fingerprint
# ------------------------------------------------------------------

class AvalonFingerprint(BaseFingerprintRepresenation):
    def mol_to_fp(self, mol):
        fp = pyAvalonTools.GetAvalonFP(
            mol,
            nBits=self.kwargs.get('nBits', 1024)
        )
        return np.array(fp, dtype=int)


# register all classes
FINGERPRINT_CLASSES = {
    "MorganFingerprint_2048": MorganFingerprint,
    'FeaturesMorganFingerprint_2048': FeaturesMorganFingerprint,
    "MACCSKeysFingerprint_167": MACCSKeysFingerprint,
    "RDKitFingerprint_2048": RDKitFingerprint,
    "AtomPairFingerprint_2048": AtomPairFingerprint,
    "TopologicalTorsionFingerprint_2048": TopologicalTorsionFingerprint,
    "AvalonFingerprint_1024": AvalonFingerprint,
}

def get_fingerprint_class(name: str):
    """Returns fingerprint class by its name"""
    name = name.lower()
    if name not in FINGERPRINT_CLASSES:
        raise ValueError(f"Unknown fingerprint type: {name}")
    return FINGERPRINT_CLASSES[name]


def _test_fingerprints():
    # Test molecule (benzene)
    test_smiles = ["c1ccccc1"]

    print("Testing mol_to_fp method in all fingerprint classes...\n")

    for name, fp_class in FINGERPRINT_CLASSES.items():
        print(f"Testing: {name}")
        try:
            # Create an instance of the fingerprint class
            fp_instance = fp_class(test_smiles)

            # Generate fingerprints
            fps = fp_instance.to_fps()

            # Check that the result is a numpy array
            if not isinstance(fps, np.ndarray):
                print(f"{name}: result is not np.ndarray, but {type(fps)}")
            elif fps.ndim != 2:
                print(f"{name}: result is np.ndarray but has unusual shape: {fps.shape}")
            else:
                print(f"{name}: OK, type {type(fps)}, shape {fps.shape}")
        except Exception as e:
            print(f"{name}:error during calculation — {type(e).__name__}: {e}")
        print()

def convert_data_from_smiles_to_fingerprints(
        smiles,
        folder_to_save, # ./data
        fingerprint_classes, # FINGERPRINT_CLASSES
        replace=False,
    ):
    print("Converting data from smiles to all fingerprints...")

    folder_to_save = Path(folder_to_save)
    if not folder_to_save.is_dir():
        raise FileNotFoundError(f"no such directory: {folder_to_save}")

    print(f"\nsaving fingerprints to folder: \'{folder_to_save}'\n")

    for k, (fp_name, fp_class) in enumerate(fingerprint_classes.items()):
        print(
            f"[{k+1}/{len(fingerprint_classes)}]\n"
            f"fp_name={fp_name}",
            end = ' ')

        savepath = folder_to_save / f"{fp_name}_X_data.npy"

        if savepath.is_file() and not replace:
            print(f"\nskipping {fp_name}, \'{savepath}\' already exists\n")
            continue

        fp_instance = fp_class(smiles)
        fp_instance.to_fps()
        fp_instance.save_fps(savepath=savepath)

        print(f"fps.shape={fp_instance.fps.shape}")

if __name__=='__main__':

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Disable RDKit logging to keep output clean

    import warnings
    # RDKit swears to usage of deprecated GetMorganFingerprintAsBitVect
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # _test_fingerprints()

    f = './data/logP_dataset.csv'
    data = pd.read_csv(f, names = ['smiles', 'logp'])

    convert_data_from_smiles_to_fingerprints(
        smiles=data['smiles'],
        folder_to_save='./data',
        fingerprint_classes=FINGERPRINT_CLASSES,
    )

    