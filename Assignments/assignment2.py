import numpy as np
from utils import read_data
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

tox_type = 'nr-ahr'
smiles, label = read_data(tox_type)

# 1. Get molecular fingerprints of each molecules 

# 2. Split the dataset to training set and test set

# 3. Train a SVM classification model

# 4. Validate the trained model using test-set
