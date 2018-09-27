import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import read_ZINC
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys

# python mlp_logP.py 3 512 0.01 0.3 0.01
num_layer = int(sys.argv[1])
hidden_dim = int(sys.argv[2])
init_lr = float(sys.argv[3])
drop_rate = float(sys.argv[4])
reg_scale = float(sys.argv[5])

model_name = 'mlp_logP_' + str(num_layer) + '_' + str(hidden_dim) + '_' + str(init_lr) + '_' + str(drop_rate) + '_' + str(reg_scale)

#1. Prepare data - X : fingerprint, Y : logP
# and split to (training:validation:test) set
fps_total, logP_total, tpsa_total = read_ZINC(50000)

num_train = 30000
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
is_training = tf.placeholder(tf.bool, shape=())

h = X
regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
for i in range(num_layer-1):
    h = tf.layers.dense(h, 
                        units=hidden_dim, 
                        use_bias=True, 
                        activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer)
    h = tf.layers.dropout(h, 
                          rate=drop_rate, 
                          training=is_training)
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
reg_loss = tf.losses.get_regularization_loss()
loss = tf.reduce_mean( (Y_pred - Y)**2 )  + reg_loss

#4. Set an optimizer
lr = tf.Variable(0.0, trainable = False)  # learning rate
opt = tf.train.GradientDescentOptimizer(lr).minimize(loss)
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
        X_batch = fps_train[i*batch_size:(i+1)*batch_size]
        Y_batch = logP_train[i*batch_size:(i+1)*batch_size]
        _opt, _Y, _loss = sess.run([opt, Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch, is_training : True})
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
        X_batch = fps_validation[i*batch_size:(i+1)*batch_size]
        Y_batch = logP_validation[i*batch_size:(i+1)*batch_size]
        _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch, is_training : False})
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
    X_batch = fps_test[i*batch_size:(i+1)*batch_size]
    Y_batch = logP_test[i*batch_size:(i+1)*batch_size]
    _Y, _loss = sess.run([Y_pred, loss], feed_dict = {X : X_batch, Y : Y_batch, is_training : False})
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
plt.savefig('./figures/'+model_name+'_results.png')
