import sys
sys.path.insert(0, './model')
sys.path.insert(0, './utils')
from convVAE import convVAE
from utils import *
import numpy as np
import os
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import tensorflow as tf

modelName = 'convVAE_re'

molecules = np.load("./inputs/molecules.npy")
char = np.load("./inputs/char.npy")
vocab_size = len(char)
seq_length = molecules.shape[1]
batch_size = 100
latent_dim = 512

model = convVAE(seq_length=seq_length,
                vocab_size=vocab_size,
                batch_size=batch_size,
                latent_dim=latent_dim,
                stddev=1.0,
                mean=0.0)
model.restore("./save/"+modelName+".ckpt-40000")

molecules_batch = molecules[-batch_size:]

retLatent = []
latent_vector = model.get_latent_vector(molecules_batch)

for vec in latent_vector:
    retLatent.append(vec)

retLatent = np.asarray(retLatent)
np.save('./latent/convVAE_200.npy', retLatent)
### Original Molecule
s_origin = []
for j in range(len(molecules_batch)):
    s_j = convert_to_smiles(molecules_batch[j], char)
    s_origin.append(s_j)

perturb = 0.1
num_samplings = 10
for i_latent in range(num_samplings):
    s_list = []
    s_list.append(s_origin[i_latent])

    ### Generated Molecules by VAE
    s_z = np.random.normal(0.0, 1.0, [batch_size, latent_dim])*perturb + latent_vector[i_latent]
    mol_gen = model.generate_molecule(s_z)
    for i in range(len(mol_gen)):
        for j in range(100):
            s = stochastic_convert_to_smiles(mol_gen[i], char)
            try:    
                m = Chem.MolFromSmiles(s)
                if m:
                    if s in s_list:
                        pass
                    else:
                        s_list.append(s)
                        break
            except:
                pass

    print (len(s_list))
    for i in s_list:
        print (s_list[i])

    mol_list = [Chem.MolFromSmiles(s) for s in s_list]
    img = Draw.MolsToGridImage([mol for mol in mol_list], molsPerRow=5)
    #del mol_list[0]
    img.save('./figures/convVAE_'+str(i_latent)+'_'+str(perturb)+'.png')
