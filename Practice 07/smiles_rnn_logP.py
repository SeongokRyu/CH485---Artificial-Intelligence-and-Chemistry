import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import read_ZINC_smiles, smiles_to_onehot, smiles_to_onehot2
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys

# python smiles_rnn_logP.py 1 128 0.01 0.001
print (len(sys.argv))
num_layer = 1
hidden_dim = 256
init_lr = 0.01
reg_scale = 0.001

if( len(sys.argv) == 5 ):
    num_layer = int(sys.argv[1])
    hidden_dim = int(sys.argv[2])
    init_lr = float(sys.argv[3])
    reg_scale = float(sys.argv[4])

model_name = 'smiles_rnn_logP_' + str(num_layer) + '_' + str(hidden_dim) + '_' + str(init_lr) + '_' + str(reg_scale)
print (model_name)

#1. Prepare data - X : fingerprint, Y : logP
# and split to (training:validation:test) set
smi_list, logP_total, tpsa_total = read_ZINC_smiles(50000)
smi_total, len_total = smiles_to_onehot2(smi_list)

num_train = 30000
num_validation = 10000
num_test = 10000

smi_train = smi_total[0:num_train]
logP_train = logP_total[0:num_train]
len_train = len_total[0:num_train]
smi_validation = smi_total[num_train:(num_train+num_validation)]
logP_validation = logP_total[num_train:(num_train+num_validation)]
len_validation = len_total[num_train:(num_train+num_validation)]
smi_test = smi_total[(num_train+num_validation):]
logP_test = logP_total[(num_train+num_validation):]
len_test = len_total[(num_train+num_validation):]

#2. Construct a neural network

# RNN Layer
def RNNLayer(h, encoder, seq_len):
    _, state = tf.nn.dynamic_rnn(encoder, h, dtype=tf.float32, scope='encode', sequence_length=seq_len)
    return state[0]

dim = smi_train.shape[1]
X = tf.placeholder(tf.float32, shape=[None, dim])
Y = tf.placeholder(tf.float32, shape=[None, ])
seq_len = tf.placeholder(tf.int32, shape=[None])
is_training = tf.placeholder(tf.bool, shape=())

X = tf.cast(X, tf.int32)
h = tf.one_hot(X, 31)
regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

### RNN Layer
cell = []
cell.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
encoder = tf.nn.rnn_cell.MultiRNNCell(cell)
h = RNNLayer(h, encoder, seq_len)

#h = tf.layers.flatten(h)
h = tf.layers.dense(h, 
                    units=hidden_dim, 
                    use_bias=True, 
                    activation=tf.nn.relu, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)

h = tf.layers.dense(h, 
                    units=hidden_dim, 
                    use_bias=True, 
                    activation=tf.nn.tanh, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
Y_pred = tf.layers.dense(h, 
                         units=1, 
                         use_bias=True, 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         kernel_regularizer=regularizer,
                         bias_regularizer=regularizer)

#3. Set a loss function, in this case we will use a MSE-loss (l2-norm)
Y_pred = tf.reshape(Y_pred, shape=[-1,])
Y_pred = tf.cast(Y_pred, tf.float32)
Y = tf.cast(Y, tf.float32)
reg_loss = tf.losses.get_regularization_loss()
loss = tf.reduce_mean( (Y_pred - Y)**2 )  + reg_loss

#4. Set an optimizer
lr = tf.Variable(0.0, trainable = False)  # learning rate
opt = tf.train.AdamOptimizer(lr).minimize(loss)
#opt = tf.train.GradientDescentOptimizer(lr).minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

#5. Training & validation
batch_size = 100
epoch_size = 100
decay_rate = 0.95
batch_train = int(num_train/batch_size)
batch_validation = int(num_validation/batch_size)
batch_test = int(num_test/batch_size)

total_iter = 0
for t in range(epoch_size):

    pred_train = []
    sess.run(tf.assign( lr, init_lr*( decay_rate**t ) ))
    for i in range(batch_train):
        total_iter += 1
        X_batch = smi_train[i*batch_size:(i+1)*batch_size]
        Y_batch = logP_train[i*batch_size:(i+1)*batch_size]
        L_batch = len_train[i*batch_size:(i+1)*batch_size]
        _opt, _Y, _loss = sess.run([opt, Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch, seq_len : L_batch, is_training : True})
        pred_train.append(_Y.flatten())
        #print("Epoch :", t, "\t batch:", i, "Loss :", _loss, "\t Training")
    pred_train = np.concatenate(pred_train, axis=0)
    error = (logP_train-pred_train)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    stdv = np.std(error)

    print ("MAE :", mae, "RMSE :", rmse, "Std :", stdv, "\t Training, \t Epoch :", t)

    pred_validation = []
    for i in range(batch_validation):
        X_batch = smi_validation[i*batch_size:(i+1)*batch_size]
        Y_batch = logP_validation[i*batch_size:(i+1)*batch_size]
        L_batch = len_validation[i*batch_size:(i+1)*batch_size]
        _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch, seq_len : L_batch, is_training : False})
        #print("Epoch :", t, "\t batch:", i, "Loss :", _loss, "\t validation")
        pred_validation.append(_Y.flatten())

    pred_validation = np.concatenate(pred_validation, axis=0)
    error = (logP_validation-pred_validation)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    stdv = np.std(error)

    print ("MAE :", mae, "RMSE :", rmse, "Std :", stdv, "\t Validation, \t Epoch :", t)

    ### save model
    ckpt_path = 'save/'+model_name+'.ckpt'
    saver.save(sess, ckpt_path, global_step=total_iter)



#6. Test
pred_test = []
for i in range(batch_test):
    X_batch = smi_test[i*batch_size:(i+1)*batch_size]
    Y_batch = logP_test[i*batch_size:(i+1)*batch_size]
    L_batch = len_test[i*batch_size:(i+1)*batch_size]
    _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch, seq_len : L_batch, is_training : False})
    #print("Epoch :", t, "\t batch:", i, "Loss :", _loss, "\t validation")
    pred_test.append(_Y.flatten())

pred_test = np.concatenate(pred_test, axis=0)
error = (logP_test-pred_test)
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error**2))
stdv = np.std(error)

print ("MSE :", mae, "RMSE :", rmse, "Std :", stdv, "\t Test")

plt.figure()
plt.scatter(logP_test, pred_test, s=3)
plt.xlabel('logP - Truth', fontsize=15)
plt.ylabel('logP - Prediction', fontsize=15)
x = np.arange(-4,6)
plt.plot(x,x,c='black')
plt.tight_layout()
plt.savefig('./figures/'+model_name+'_results.png')
