from rdkit import Chem
import numpy as np

def read_data(filename):
    f = open('../tox21/'+filename + '.smiles', 'r')
    contents = f.readlines()

    smiles = []
    labels = []
    for i in contents:
        smi = i.split()[0]
        label = int(i.split()[2].strip())

        smiles.append(smi)
        labels.append(label)

    num_total = len(smiles)
    rand_int = np.random.randint(num_total, size=(num_total,))
    
    return np.asarray(smiles)[rand_int], np.asarray(labels)[rand_int]
