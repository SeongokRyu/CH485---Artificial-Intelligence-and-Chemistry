import sys
from convVAE import convVAE
from utils import *
import numpy as np
import os
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
latent_dim = 512
batch_size = 100
molecules = np.load("./inputs/molecules.npy")
char = np.load("./inputs/char.npy")

N_train = 80000
N_val = 20000
train_molecules = molecules[0:N_train]
val_molecules = molecules[N_train:N_train+N_val]
num_batches=int((N_train+N_val)/batch_size)

seq_length = molecules.shape[1]
vocab_size = len(char)
model = convVAE(seq_length=seq_length,
                vocab_size=vocab_size,
                batch_size=batch_size,
                latent_dim=latent_dim,
                stddev=1.0,
                mean=0.0)

num_epochs    = 50
save_every    = 5000
learning_rate = 0.001
decay_rate    = 0.9

total_iter = 0
for epoch in range(num_epochs):
    # Learning rate scheduling 
    model.assign_lr(learning_rate * (decay_rate ** epoch))
    print ("Start training")
    st = time.time()

    num_train = train_molecules.shape[0]//batch_size
    for i in range(num_train):
        total_iter += 1
        mol_batch = train_molecules[i*batch_size:(i+1)*batch_size]
        cost = model.train(mol_batch)

    num_val = val_molecules.shape[0]//batch_size
    for i in range(num_val):
        mol_batch = val_molecules[i*batch_size:(i+1)*batch_size]
        Y, cost = model.test(mol_batch)
        if( i % 50 == 0):
            print ("test_iter : ", i, ", epoch : ", epoch, " ", cost)
            accuracy1_, accuracy2_ = accuracy(mol_batch, Y)
            print ("accuracy : ", accuracy1_, accuracy2_)
            for i in range (10):
                s1_1 = convert_to_smiles(mol_batch[i,:],char)
                s1_2 = convert_to_smiles(Y[i,:], char)
                print (s1_1+"\n"+s1_2)

    et = time.time()
    print ("time for epoch", epoch, ":", et-st)
    st = time.time()

    if total_iter % save_every == 0:
        ckpt_path = 'save/convVAE_re.ckpt'
        model.save(ckpt_path, total_iter)
        # Save network! 

