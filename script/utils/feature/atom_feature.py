from rdkit import Chem
from rdkit.Chem import Atom
from typing import List, Union

# cited from https://github.com/chemprop/chemprop.git

__all__ = ['NUM_ATOM_FEATURES', 'atom_features']

others=-100
ATOM_FEATURES = {
    'period': [0, 1, 2, 3, 4, 5],
    'group': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'degree': [0, 1, 2, 3, 4, 5, others],
    'valence' : [0, 1, 2, 3, 4, 5, 6, 7, 8, others],
    'formal_charge': [-1, -2, 1, 2, 0, others],
    'num_Hs': [0, 1, 2, 3, 4, others],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        others
    ]
}

NUM_ATOM_FEATURES = sum(len(choices) for choices in ATOM_FEATURES.values()) + 3
# 6 + 9 + 7 + 10 + 6 + 6 + 6 + 1 + 2 = 53

electronegativity = { 
   1 : 2.20, # H
   3 : 0.98, # Li
   4 : 1.57, # Be
   5 : 2.04, # B
   6 : 2.55, # C
   7 : 3.04, # N
   8 : 3.44, # O
   9 : 3.98, # F
   11: 0.93, # Na
   12: 1.31, # Mg
   13: 1.61, # Al
   14: 1.90, # Si
   15: 2.19, # P
   16: 2.59, # S
   17: 3.16, # Cl
   19: 0.82, # K
   20: 1.00, # Ca
   21: 1.36, # Sc
   22: 1.54, # Ti
   23: 1.63, # V
   24: 1.66, # Cr
   25: 1.55, # Mn
   26: 1.83, # Fe
   27: 1.88, # Co
   28: 1.91, # Ni
   29: 1.90, # Cu 
   30: 1.65, # Zn
   31: 1.81, # Ga
   32: 2.01, # Ge
   33: 2.18, # As
   34: 2.55, # Se
   35: 2.96, # Br
   36: 3.00, # Kr
   37: 0.82, # Rb
   38: 0.95, # Sr
   39: 1.22, # Y
   40: 1.33, # Zr
   41: 1.60, # Nb
   42: 2.16, # Mo
   43: 1.90, # Tc
   44: 2.20, # Ru
   45: 2.28, # Rh
   46: 2.20, # Pd
   47: 1.93, # Ag
   48: 1.69, # Cd
   49: 1.78, # In
   50: 1.96, # Sn
   51: 2.05, # Sb
   52: 2.10, # Te
   53: 2.66, # I
   54: 2.60, # Xe
}

def atom_features(atom: Atom) -> List[Union[int, float]]:
    atomic_num = atom.GetAtomicNum()
    period, group = _get_periodic_feature(atomic_num)
    degree = atom.GetTotalDegree()
    valence = atom.GetTotalValence()
    formal_charge = atom.GetFormalCharge()
    num_Hs = atom.GetTotalNumHs()
    aromatics = atom.GetIsAromatic()
    hybridization = atom.GetHybridization()
    mass = atom.GetMass()
    en = electronegativity[atomic_num]
    features = _onek_encoding_unk(period, ATOM_FEATURES['period']) + \
               _onek_encoding_unk(group, ATOM_FEATURES['group']) + \
               _onek_encoding_unk(degree, ATOM_FEATURES['degree']) + \
               _onek_encoding_unk(valence, ATOM_FEATURES['valence']) + \
               _onek_encoding_unk(formal_charge, ATOM_FEATURES['formal_charge']) + \
               _onek_encoding_unk(num_Hs, ATOM_FEATURES['num_Hs']) + \
               _onek_encoding_unk(hybridization, ATOM_FEATURES['hybridization']) + \
               [1 if aromatics else 0] + \
               [mass * 0.01, en * 0.2] #scaled to ablut the same range as other features
    return features
               
_periodic_table = Chem.GetPeriodicTable()
def _get_periodic_feature(atomic_num: int) :
    periodic_list = [0, 2, 10, 18, 36, 54]
    for i in range(len(periodic_list)) :
        if periodic_list[i] >= atomic_num :
            period = i
            group = _periodic_table.GetNOuterElecs(atomic_num)
    return period, group

def _onek_encoding_unk(value: int, choices: List[int]) -> List[int] :
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding
