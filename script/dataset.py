import torch
from torch import FloatTensor, BoolTensor
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Mol
from typing import List, Tuple, Optional
from utils import feature


class MolGraphDataset(Dataset) :
    def __init__(
        self,
        smiles: List[str],
        label:List[int],
        max_atoms: int,
        setup: bool = False
        ) :
        # set setup to True when the size of dataset is small.
        super(MolGraphDataset, self).__init__()
        self.max_atoms = max_atoms
        self.smiles = []
        self.label = label
        self.V = []
        self.A = []
        self.setup = setup 
        if setup :
            for s in smiles :
                mol = Chem.MolFromSmiles(s)
                v, adj = self.get_features(mol)
                self.V.append(v)
                self.A.append(adj)
        else :
            self.smiles = smiles
    def __len__(self) :
        return len(self.smiles)

    def __getitem__(self, idx: int) :
        sample = dict()
        if self.setup :
            sample['V'] = self.V[idx]
            sample['A'] = self.A[idx]
            sample['Y'] = self.label[idx]
        else :
            smiles = self.smiles[idx]
            mol = Chem.MolFromSmiles(smiles)
            sample['V'], sample['A'] = self.get_features(mol)
            sample['Y'] = self.label[idx]
        return sample
    
    def get_features(self, mol: Mol) -> Tuple[FloatTensor, BoolTensor]:
        return feature.get_atom_features(mol, self.max_atoms), feature.get_adj(mol, self.max_atoms)
