import h5py
import numpy as np
from rdkit import Chem

def convert_to_smiles(vector, char):
    list_char = char.tolist()
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()

def stochastic_convert_to_smiles(vector, char):
    list_char = char.tolist()
    s = ""
    for i in range(len(vector)):
        prob = vector[i].tolist()
        norm0 = sum(prob)
        prob = [i/norm0 for i in prob]
        index = np.random.choice(len(list_char), 1, p=prob)
        s+=list_char[index[0]]
    return s

def one_hot_array(i, n):
    return list(map(int, [ix == i for ix in range(n)]))

def one_hot_index(vec, charset):
    return list(map(charset.index, vec))

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return data_train, data_test, charset
    else:
        return data_test, charset

def encode_smiles(smiles, model, charset):
    cropped = list(smiles.ljust(120))
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))])
    latent = model.encoder.predict(preprocessed)
    return latent

def smiles_to_onehot(smiles, charset):
    cropped = list(smiles.ljust(120))
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))])
    return preprocessed

def smiles_to_vector(smiles, vocab, max_length):
    while len(smiles)<max_length:
        smiles +=" "
    return [vocab.index(str(x)) for x in smiles]

def decode_latent_molecule(latent, model, charset, latent_dim):
    decoded = model.decoder.predict(latent.reshape(1, latent_dim)).argmax(axis=2)[0]
    smiles = decode_smiles_from_indexes(decoded, charset)
    return smiles

def interpolate(source_smiles, dest_smiles, steps, charset, model, latent_dim):
    source_latent = encode_smiles(source_smiles, model, charset)
    dest_latent = encode_smiles(dest_smiles, model, charset)
    step = (dest_latent - source_latent) / float(steps)
    results = []
    for i in range(steps):
        item = source_latent + (step * i)        
        decoded = decode_latent_molecule(item, model, charset, latent_dim)
        results.append(decoded)
    return results

def get_unique_mols(mol_list):
    inchi_keys = [Chem.InchiToInchiKey(Chem.MolToInchi(m)) for m in mol_list]
    u, indices = np.unique(inchi_keys, return_index=True)
    unique_mols = [[mol_list[i], inchi_keys[i]] for i in indices]
    return unique_mols

def accuracy(arr1, arr2):
    total = len(arr1)
    count=0
    for i in range(len(arr1)):
        if np.array_equal(arr1[i], arr2[i]):
            count+=1

    return float(count/float(total)), np.sum(arr1==arr2)/arr1.size

def convertToGraph(molecules, char, k):
    # Convert One-hot format to Smiles and then convert once again to graph
    # 1. One-hot To Smiles
    smiles_list = []
    for i in molecules:
        smiles_list.append( convert_to_smiles(i, char) )

    # 2. smiles To Graph
    #adj_label, adj_norm, features = smilesToGraph(smiles_list)
    adj, features = smilesToGraph(smiles_list, k)
    return adj, features
    #return adj_norm, features
    #return adj_label, adj_norm, features

def smilesToGraph(smiles_list, k):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50
    for i in smiles_list:
        # Mol
        iMol = Chem.MolFromSmiles(i)
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 62))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:62] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            #iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            #iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), k))
            """

            # Adj-norm-preprocessing
            iAdjNorm = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdjNormTmp = preprocess_graph_chem(iAdjTmp)
            iAdjNorm[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjNormTmp
            adj_norm.append(iAdjNorm)
            """

    #adj = adj
    #adj_norm = adj_norm
    #features = features
    #adj = adj_k(np.asarray(adj), k)
    features = np.asarray(features)
    #adj_norm = np.asarray(adj_norm)

    return adj, features
    #return adj, adj_norm, features
    
def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (44, 6, 5, 6, 1)

def preprocess_graph_chem(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized               # tild-A in original paper

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def convertFeatures(features):
    # Convert atomic features of a single molecule.
    retVal = []
    for singleFeature in features:
        # Atom type
        iType = np.argmax(singleFeature[0:44])
        # atom degree
        iDegree = np.argmax(singleFeature[44:50])
        # atom implict numH
        iNumH = np.argmax(singleFeature[50:55])
        # atom implicit Valence
        iValence = np.argmax(singleFeature[55:61])
        # isAromatic    
        iAromatic = round(singleFeature[-1])
        atomFeature = [iType, iDegree, iNumH, iValence, iAromatic]
        retVal.append( atomFeature )

    return np.asarray(retVal)

def convert_to_onehot(vector, depth):
    a = np.zeros((len(vector), depth))
    a[np.arange(len(vector)), vector] = 1
    return a

def adj_k(adj, k):

    ret = adj
    for i in range(0, k-1):
        adj = np.dot(ret, adj)  

    return convertAdj(ret)

def convertAdj(adj):

    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype(int)
    d = c.reshape((dim, dim))

    return d

def construct_feed_dict(adj, features, molecules, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['adj'] : adj})
    feed_dict.update({placeholders['features'] : features})
    feed_dict.update({placeholders['molecules'] : molecules})
    return feed_dict
