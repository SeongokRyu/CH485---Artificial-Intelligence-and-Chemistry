import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import read_ZINC
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


#1. Prepare data - X : fingerprint, Y : logP
# and split to (training:validation:test) set
fps_total, logP_total, tpsa_total = read_ZINC(60000)

num_train = 40000
num_validation = 10000
num_test = 10000

fps_train = fps_total[0:num_train]
logP_train = logP_total[0:num_train]
fps_validation = fps_total[num_train:(num_train+num_validation)]
logP_validation = logP_total[num_train:(num_train+num_validation)]
fps_test = fps_total[(num_train+num_validation):]
logP_test = logP_total[(num_train+num_validation):]

#2. Construct a neural network
X = tf.placeholder(tf.float64, shape=[None, 2048])
Y = tf.placeholder(tf.float64, shape=[None, ])

h1 = tf.layers.dense(X, units=512, use_bias=True, activation=tf.nn.relu)
h2 = tf.layers.dense(h1, units=512, use_bias=True, activation=tf.nn.tanh)
Y_pred = tf.layers.dense(h2, units=1, use_bias=True)

#3. Set a loss function, in this case we will use a MSE-loss (l2-norm)
Y_pred = tf.layers.flatten(Y_pred)
loss = tf.reduce_mean( (Y_pred - Y)**2 ) 

#4. Set an optimizer
lr = tf.Variable(0.0, trainable = False)  # learning rate
opt = tf.train.AdamOptimizer(lr).minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#5. Training & validation
batch_size = 100
epoch_size = 100
decay_rate = 0.95
batch_train = int(num_train/batch_size)
batch_validation = int(num_validation/batch_size)
batch_test = int(num_test/batch_size)

init_lr = 0.001
for t in range(epoch_size):

    pred_train = []
    sess.run(tf.assign( lr, init_lr*( decay_rate**t ) ))
    for i in range(batch_train):
        X_batch = fps_train[i*batch_size:(i+1)*batch_size]
        Y_batch = logP_train[i*batch_size:(i+1)*batch_size]
        _opt, _Y, _loss = sess.run([opt, Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch})
        pred_train.append(_Y.flatten())
        #print("Epoch :", t, "\t batch:", i, "Loss :", _loss, "\t Training")
    pred_train = np.concatenate(pred_train, axis=0)
    error = (logP_train-pred_train)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    stdv = np.std(error)

    print ("MSE :", mae, "RMSE :", rmse, "Std :", stdv, "\t Training, \t Epoch :", t)

    pred_validation = []
    for i in range(batch_validation):
        X_batch = fps_validation[i*batch_size:(i+1)*batch_size]
        Y_batch = logP_validation[i*batch_size:(i+1)*batch_size]
        _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch})
        #print("Epoch :", t, "\t batch:", i, "Loss :", _loss, "\t validation")
        pred_validation.append(_Y.flatten())

    pred_validation = np.concatenate(pred_validation, axis=0)
    error = (logP_validation-pred_validation)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    stdv = np.std(error)

    print ("MSE :", mae, "RMSE :", rmse, "Std :", stdv, "\t Validation, \t Epoch :", t)


#6. Test
pred_test = []
for i in range(batch_test):
    X_batch = fps_test[i*batch_size:(i+1)*batch_size]
    Y_batch = logP_test[i*batch_size:(i+1)*batch_size]
    _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch})
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
plt.xlabel('logP - Truth')
plt.ylabel('logP - Prediction')
plt.savefig('logP_mlp.png')
