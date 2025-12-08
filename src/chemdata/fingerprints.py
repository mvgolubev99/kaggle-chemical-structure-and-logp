import warnings


import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm


class BaseFingerprint:
    def __init__(self, smiles, *args, **kwargs):
        self.smiles = smiles
        self.args = args
        self.kwargs = kwargs
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]

    def mol_to_fp(self, mol):
        raise NotImplementedError

    def to_fps(self):
        fps = []
        print(f"converting data to {type(self).__name__}...")
        for mol in tqdm(self.mols):
            fp = self.mol_to_fp(mol)
            fps.append(fp)
        self.fps = np.array(fps)
        return self.fps


class MorganFingerprint(BaseFingerprint):
    def mol_to_fp(self, mol):
        if mol is None:
            # Return zero fingerprint for invalid molecules
            return np.zeros(self.kwargs.get('fpSize', 2048), dtype=int)
        
        gen = GetMorganGenerator(
            radius=self.kwargs.get('radius', 2),
            fpSize=self.kwargs.get('fpSize', 2048),
        )
        fp = gen.GetFingerprint(mol)
        return np.array(fp, dtype=int)


class FeaturesMorganFingerprint(BaseFingerprint):
    def mol_to_fp(self, mol):
        if mol is None:
            # Return zero fingerprint for invalid molecules
            return np.zeros(self.kwargs.get('fpSize', 2048), dtype=int)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.kwargs.get('radius', 2),
            nBits=self.kwargs.get('nBits', 2048),
            useFeatures=True,
        )
        return np.array(fp, dtype=int)


class MACCSKeysFingerprint(BaseFingerprint):
    def mol_to_fp(self, mol):
        if mol is None:
            # Return zero fingerprint for invalid molecules
            return np.zeros(167, dtype=int)
        
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=int)


class RDKitFingerprint(BaseFingerprint):
    def mol_to_fp(self, mol):
        if mol is None:
            # Return zero fingerprint for invalid molecules
            return np.zeros(self.kwargs.get('fpSize', 2048), dtype=int)
        
        fp = Chem.RDKFingerprint(
            mol,
            minPath=self.kwargs.get('minPath', 1),
            maxPath=self.kwargs.get('maxPath', 7),
            fpSize=self.kwargs.get('fpSize', 2048),
            useHs=self.kwargs.get('useHs', True),
            branchedPaths=self.kwargs.get('branchedPaths', True)
        )
        return np.array(fp, dtype=int)


class AtomPairFingerprint(BaseFingerprint):
    def mol_to_fp(self, mol):
        if mol is None:
            # Return zero fingerprint for invalid molecules
            return np.zeros(self.kwargs.get('fpSize', 2048), dtype=int)
        
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol,
            nBits=self.kwargs.get('nBits', 2048)
        )
        return np.array(fp, dtype=int)


class TopologicalTorsionFingerprint(BaseFingerprint):
    def mol_to_fp(self, mol):
        if mol is None:
            # Return zero fingerprint for invalid molecules
            return np.zeros(self.kwargs.get('fpSize', 2048), dtype=int)
        
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol,
            nBits=self.kwargs.get('nBits', 2048)
        )
        return np.array(fp, dtype=int)


class AvalonFingerprint(BaseFingerprint):
    def mol_to_fp(self, mol):
        if mol is None:
            # Return zero fingerprint for invalid molecules
            return np.zeros(self.kwargs.get('fpSize', 1024), dtype=int)
        
        fp = pyAvalonTools.GetAvalonFP(
            mol,
            nBits=self.kwargs.get('nBits', 1024)
        )
        return np.array(fp, dtype=int)


# register all classes
FINGERPRINT_CLASSES = {
    "MorganFingerprint_2048": MorganFingerprint,
    "FeaturesMorganFingerprint_2048": FeaturesMorganFingerprint,
    "MACCSKeysFingerprint_167": MACCSKeysFingerprint,
    "RDKitFingerprint_2048": RDKitFingerprint,
    "AtomPairFingerprint_2048": AtomPairFingerprint,
    "TopologicalTorsionFingerprint_2048": TopologicalTorsionFingerprint,
    "AvalonFingerprint_1024": AvalonFingerprint,
}


def convert_smiles_to_fingerprints(
        smiles,
        fingerprint_classes=FINGERPRINT_CLASSES,
    ):
    RDLogger.DisableLog('rdApp.*')  # Disable RDKit logging

    X_data = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for k, (fp_name, fp_cls) in enumerate(fingerprint_classes.items()):
            print(f"[{k+1}/{len(fingerprint_classes)}] {fp_name}")
            X_data[fp_name] = fp_cls(smiles=smiles).to_fps()

    return X_data