import sys
sys.path.insert(0, './model')
sys.path.insert(0, './utils')
from convVAE_jointly import convVAE_jointly
from utils import *
import numpy as np
import os
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import tensorflow as tf

modelName = 'convVAE_re'

N = 20000
molecules = np.load("./inputs/molecules.npy")
char = np.load("./inputs/char.npy")
vocab_size = len(char)
seq_length = molecules.shape[1]
batch_size = 100
latent_dim = 512

model = convVAE_jointly(seq_length=seq_length,
                        vocab_size=vocab_size,
                        batch_size=batch_size,
                        latent_dim=latent_dim,
                        stddev=1.0,
                        mean=0.0)
model.restore("./save/"+modelName+".ckpt-40000")

num_batches = N//batch_size
retLatent = []
for i in range(num_batches):
    molecules_batch = molecules[i*batch_size:(i+1)*batch_size]
    latent_vector = model.get_latent_vector(molecules_batch)
    for vec in latent_vector:
        retLatent.append(vec)

retLatent = np.asarray(retLatent)
np.save('./latent/convVAE_200_jointly_all.npy', retLatent)
