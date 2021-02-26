from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
import torch
from torch import FloatTensor, BoolTensor
from typing import Union, Tuple, Optional, List
from .atom_feature import atom_features, NUM_ATOM_FEATURES

def get_atom_features(mol: Union[Mol,str],
                      max_atoms: int = 50,
                      device: Union[str, torch.device, None] = None) -> FloatTensor :
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    af = torch.zeros((max_atoms, NUM_ATOM_FEATURES), dtype=torch.float)
    for idx, atom in enumerate(mol.GetAtoms()) :
        af[idx, :] = torch.Tensor(atom_features(atom))
    return af.to(device)

def get_adj(mol: Union[Mol,str],
            max_atoms: int = 50,
            device: Optional[Union[str, torch.device]] = None) -> BoolTensor :
    adj = GetAdjacencyMatrix(mol) + np.eye(mol.GetNumAtoms())
    padded_adj = np.zeros((max_atoms, max_atoms), dtype='b')
    n_atom = len(adj)
    padded_adj[:n_atom, :n_atom] = adj
    return torch.from_numpy(padded_adj).to(device)
