import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import read_ZINC_smiles, smiles_to_onehot, convert_to_graph
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys
import time

# execution) python gcn_logP.py 3 64 256 0.001 gsc

# Default option
num_layer = 3
hidden_dim1 = 64
hidden_dim2 = 256
init_lr = 0.001
using_sc = 'gsc' # 'sc, 'gsc, 'no'

if( len(sys.argv) == 6 ):
    # Note that sys.argv[0] is gcn_logP.py
    num_layer = int(sys.argv[1])
    hidden_dim1 = int(sys.argv[2])
    hidden_dim2 = int(sys.argv[3])
    init_lr = float(sys.argv[4])
    using_sc = sys.argv[5]              # 'sc, 'gsc, 'no'

model_name = 'gcn_logP_' + str(num_layer) + '_' + str(hidden_dim1) + '_' + str(hidden_dim2) + '_' + str(init_lr) + '_' + using_sc

#1. Prepare data - X : fingerprint, Y : logP
# and split to (training:validation:test) set
smi_total, logP_total, tpsa_total = read_ZINC_smiles(50000)

num_train = 30000
num_validation = 10000
num_test = 10000

smi_train = smi_total[0:num_train]
logP_train = logP_total[0:num_train]
smi_validation = smi_total[num_train:(num_train+num_validation)]
logP_validation = logP_total[num_train:(num_train+num_validation)]
smi_test = smi_total[(num_train+num_validation):]
logP_test = logP_total[(num_train+num_validation):]

#2. Construct a neural network

def skip_connection(input_X, new_X, act):
    # Skip-connection, H^(l+1)_sc = H^(l) + H^(l+1)
    inp_dim = int(input_X.get_shape()[2])
    out_dim = int(new_X.get_shape()[2])

    if(inp_dim != out_dim):
        output_X = act(new_X + tf.layers.dense(input_X, units=out_dim, use_bias=False))

    else:
        output_X = act(new_X + input_X)

    return output_X        

def gated_skip_connection(input_X, new_X, act):
    # Skip-connection, H^(l+1)_gsc = z*H^(l) + (1-z)*H^(l+1)
    inp_dim = int(input_X.get_shape()[2])
    out_dim = int(new_X.get_shape()[2])

    def get_gate_coefficient(input_X, new_X, out_dim):
        X1 = tf.layers.dense(input_X, units=out_dim, use_bias=True)
        X2 = tf.layers.dense(new_X, units=out_dim, use_bias=True)
        gate_coefficient = tf.nn.sigmoid(X1 + X2)

        return gate_coefficient

    if(inp_dim != out_dim):
        input_X = tf.layers.dense(input_X, units=out_dim, use_bias=False)

    gate_coefficient = get_gate_coefficient(input_X, new_X, out_dim)
    output_X = tf.multiply(new_X, gate_coefficient) + tf.multiply(input_X, 1.0-gate_coefficient)

    return output_X

def graph_convolution(input_X, input_A, hidden_dim, act, using_sc):
    # Graph Convolution, H^(l+1) = A{H^(l)W^(l)+b^(l))
    output_X = tf.layers.dense(input_X,
                               units=hidden_dim, 
                               use_bias=True,
                               activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    output_X = tf.matmul(input_A, output_X)

    if( using_sc == 'sc' ):
        output_X = skip_connection(input_X, output_X, act)

    elif( using_sc == 'gsc' ):
        output_X = gated_skip_connection(input_X, output_X, act)

    elif( using_sc == 'no' ):
        output_X = act(output_X)

    else:
        output_X = gated_skip_connection(input_X, output_X)

    return output_X

# Readout
def readout(input_X, hidden_dim, act):
    # Readout, Z = sum_{v in G} NN(H^(L)_v)
    output_Z = tf.layers.dense(input_X, 
                               units=hidden_dim, 
                               use_bias=True,
                               activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    output_Z = tf.reduce_sum(output_Z, axis=1)
    output = act(output_Z)

    return output_Z


num_atoms=50
num_features=58
X = tf.placeholder(tf.float64, shape=[None, num_atoms, num_features])
A = tf.placeholder(tf.float64, shape=[None, num_atoms, num_atoms])
Y = tf.placeholder(tf.float64, shape=[None, ])
is_training = tf.placeholder(tf.bool, shape=())

h = X
# Graph convolution layers
for i in range(num_layer):
    h = graph_convolution(h,
                          A, 
                          hidden_dim1, 
                          tf.nn.relu,
                          using_sc)

# Readout layer
h = readout(h, hidden_dim2, tf.nn.sigmoid)

# Predictor composed of MLPs(multi-layer perceptron)
h = tf.layers.dense(h, 
                    units=hidden_dim2, 
                    use_bias=True, 
                    activation=tf.nn.relu, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
h = tf.layers.dense(h, 
                    units=hidden_dim2, 
                    use_bias=True, 
                    activation=tf.nn.tanh, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
Y_pred = tf.layers.dense(h, 
                         units=1, 
                         use_bias=True, 
                         kernel_initializer=tf.contrib.layers.xavier_initializer())

#3. Set a loss function, in this case we will use a MSE-loss (l2-norm)
Y_pred = tf.reshape(Y_pred, shape=[-1,])
Y_pred = tf.cast(Y_pred, tf.float64)
Y = tf.cast(Y, tf.float64)
loss = tf.reduce_mean( (Y_pred - Y)**2 )

#4. Set an optimizer
lr = tf.Variable(0.0, trainable = False)  # learning rate
opt = tf.train.AdamOptimizer(lr).minimize(loss) # Note that we use the Adam optimizer in this practice.
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
total_time = 0.0
for t in range(epoch_size):

    pred_train = []
    sess.run(tf.assign( lr, init_lr*( decay_rate**t ) ))
    st = time.time()
    for i in range(batch_train):
        total_iter += 1
        smi_batch = smi_train[i*batch_size:(i+1)*batch_size]
        X_batch, A_batch = convert_to_graph(smi_batch)
        Y_batch = logP_train[i*batch_size:(i+1)*batch_size]
        _opt, _Y, _loss = sess.run([opt, Y_pred, loss], feed_dict = {X : X_batch, A : A_batch, Y : Y_batch, is_training : True})
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
        smi_batch = smi_validation[i*batch_size:(i+1)*batch_size]
        X_batch, A_batch = convert_to_graph(smi_batch)
        Y_batch = logP_validation[i*batch_size:(i+1)*batch_size]
        _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, A : A_batch, Y : Y_batch, is_training : False})
        #print("Epoch :", t, "\t batch:", i, "Loss :", _loss, "\t validation")
        pred_validation.append(_Y.flatten())

    pred_validation = np.concatenate(pred_validation, axis=0)
    error = (logP_validation-pred_validation)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    stdv = np.std(error)

    et = time.time()
    print ("MAE :", mae, "RMSE :", rmse, "Std :", stdv, "\t Validation, \t Epoch :", t, "\t Time per epoch", (et-st))
    total_time += (et-st)

    ### save model
    ckpt_path = 'save/'+model_name+'.ckpt'
    saver.save(sess, ckpt_path, global_step=total_iter)



#6. Test
pred_test = []
for i in range(batch_test):
    smi_batch = smi_test[i*batch_size:(i+1)*batch_size]
    X_batch, A_batch = convert_to_graph(smi_batch)
    Y_batch = logP_test[i*batch_size:(i+1)*batch_size]
    _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, A : A_batch, Y : Y_batch, is_training : False})
    pred_test.append(_Y.flatten())

pred_test = np.concatenate(pred_test, axis=0)
error = (logP_test-pred_test)
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error**2))
stdv = np.std(error)

print ("MSE :", mae, "RMSE :", rmse, "Std :", stdv, "\t Test", "\t Total time :", total_time)

plt.figure()
plt.scatter(logP_test, pred_test, s=3)
plt.xlabel('logP - Truth', fontsize=15)
plt.ylabel('logP - Prediction', fontsize=15)
x = np.arange(-4,6)
plt.plot(x,x,c='black')
plt.tight_layout()
plt.savefig('./figures/'+model_name+'_results.png')
