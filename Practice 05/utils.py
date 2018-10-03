import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA

def read_data(filename):
    f = open(filename + '.smiles', 'r')
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

def read_ZINC(num_mol):
    f = open('ZINC.smiles', 'r')
    contents = f.readlines()

    smi = []
    fps = []
    logP = []
    tpsa = []
    for i in range(num_mol):
        smi = contents[i].strip()
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m,2)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        fps.append(arr)
        logP.append(MolLogP(m))
        tpsa.append(CalcTPSA(m))

    fps = np.asarray(fps).astype(float)
    logP = np.asarray(logP).astype(float)
    tpsa = np.asarray(tpsa).astype(float)

    return fps, logP, tpsa

def read_ZINC_smiles(num_mol):
    f = open('ZINC.smiles', 'r')
    contents = f.readlines()

    smi_list = []
    logP_list = []
    tpsa_list = []
    for i in range(num_mol):
        smi = contents[i].strip()
        m = Chem.MolFromSmiles(smi)
        smi_list.append(smi)
        logP_list.append(MolLogP(m))
        tpsa_list.append(CalcTPSA(m))

    logP_list = np.asarray(logP_list).astype(float)
    tpsa_list = np.asarray(tpsa_list).astype(float)

    return smi_list, logP_list, tpsa_list

def smiles_to_onehot(smi_list):
    def smiles_to_vector(smiles, vocab, max_length):
        while len(smiles)<max_length:
            smiles +=" "
        return [vocab.index(str(x)) for x in smiles]

    vocab = np.load('./vocab.npy')
    smi_total = []
    for smi in smi_list:
        smi_onehot = smiles_to_vector(smi, list(vocab), 120)
        smi_total.append(smi_onehot)
    return np.asarray(smi_total)
