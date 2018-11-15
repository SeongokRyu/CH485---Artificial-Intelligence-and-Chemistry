import numpy as np
from utils import *
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

molecules = np.load("./inputs/molecules.npy")
char = np.load("./inputs/char.npy")
N = 200000

logP_list = []
for i in range(N):
    smi = convert_to_smiles(molecules[i], char)
    mol = Chem.MolFromSmiles(smi)
    logP = MolLogP(mol)
    logP_list.append(logP)

np.save("./inputs/logP.npy", np.asarray(logP_list))
