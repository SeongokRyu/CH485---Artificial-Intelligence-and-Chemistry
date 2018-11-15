import numpy as np
from utils import *
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
import matplotlib.pyplot as plt
plt.switch_backend('agg')

latent_all = np.load('./latent/convVAE_200_jointly_all.npy')
N = latent_all.shape[0]
logP_all = np.load('../practice09/inputs/logP.npy')[:N]

fig=plt.figure()
dim = 2
Z = PCA(n_components = dim).fit(latent_all).transform(latent_all)
Z = np.transpose(Z)
cmap=plt.get_cmap('rainbow')
cax = plt.scatter(Z[0], Z[1], c=logP_all, s=3, cmap=cmap)
plt.colorbar(cax)
plt.tight_layout()
plt.savefig('./figures/convVAE_jointly_PCA.png')
