import tensorflow as tf
class convVAE():
    def __init__(self,
                 seq_length,
                 vocab_size,
                 batch_size=20,
                 latent_dim=200,
                 stddev = 1.0,
                 mean = 0.0
                  ):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.mean = mean
        self.stddev = 1.0
        self.X = tf.placeholder(shape=(None, seq_length), dtype=tf.float32)
        self._create_network()


    def _create_network(self):
        self.X = tf.cast(self.X, tf.int32)
        self.X_onehot = tf.one_hot(self.X, self.vocab_size)
        self.X_onehot = tf.cast(self.X_onehot, tf.float32)
        self.encoder_dim = [512,256,256,128] #number of filter for encoder
        self.decoder_dim = [128,256,256,512] #number of filter for decoder
        self.eps = tf.random_normal([self.batch_size, self.latent_dim], stddev=self.stddev, mean=self.mean)

        self.z, z_mean, z_logvar = self.conv_encoder(self.X_onehot)
        self._X = self.conv_decoder(self.z)

        # Loss
        loss1, loss2 = self.cal_loss( self.X_onehot, self._X, z_mean, z_logvar)
        cost1 = tf.reduce_mean(loss1)*self.seq_length
        cost2 = tf.reduce_mean(loss2)
        self.cost = tf.cast(cost1,tf.float32)+tf.cast(cost2,tf.float32)
        self.mol_pred = tf.argmax(self._X, axis=2)
        
        # Configure optimizatier
        self.lr = tf.Variable(0.0, trainable=False)
        optimizer    = tf.train.AdamOptimizer(self.lr)
        self.opt = optimizer.minimize(self.cost)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print ("Network Ready")

        return

    def conv_encoder(self, X):
        # input : X, molecule
        # output : z, latent vector

        output = X
        for i in range(len(self.encoder_dim)):
            output = tf.layers.conv1d(X, 
                                      filters=self.encoder_dim[i], 
                                      kernel_size=9, 
                                      use_bias=True, 
                                      activation=tf.nn.relu, 
                                      padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                      bias_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.layers.flatten(output)
        output = tf.layers.dense(output, 
                                 units=self.latent_dim, 
                                 use_bias=True, 
                                 activation=tf.nn.sigmoid, 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                 bias_initializer=tf.contrib.layers.xavier_initializer())
        z_mean = tf.layers.dense(output, 
                                 units=self.latent_dim, 
                                 use_bias=True, 
                                 activation=None, 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                 bias_initializer=tf.contrib.layers.xavier_initializer())
        z_logvar = tf.layers.dense(output, 
                                 units=self.latent_dim, 
                                 use_bias=True, 
                                 activation=None, 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                 bias_initializer=tf.contrib.layers.xavier_initializer())
        output = z_mean+tf.exp(z_logvar/2.0)*self.eps
        return output, z_mean, z_logvar

    def conv_decoder(self, z):
        # input : z, latent vector
        # output : _X, decoded molecule
        output = tf.layers.dense(z,
                                 units=self.encoder_dim[-1]*self.seq_length,
                                 use_bias=True,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                 bias_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.reshape(output, [self.batch_size, self.seq_length, -1])
        for i in range(len(self.decoder_dim)):
            output = tf.layers.conv1d(output,
                                      filters=self.decoder_dim[i],
                                      kernel_size=9,
                                      activation=tf.nn.relu, 
                                      padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                      bias_initializer=tf.contrib.layers.xavier_initializer())
            print (output)
        output = tf.layers.dense(output,
                                 units=self.vocab_size,
                                 use_bias=True,
                                 activation=tf.nn.softmax,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                 bias_initializer=tf.contrib.layers.xavier_initializer())
        print (output)
        return output

    def cal_reconstr_loss(self, X, Y):
        X = tf.reshape(X, [self.batch_size, -1])
        Y = tf.reshape(Y, [self.batch_size, -1])
        reconstr_loss = -(X*tf.log(1e-8+Y))
        #reconstr_loss = -(X*tf.log(1e-30+Y)+(1-X)*tf.log(1e-30+1-Y))
        return reconstr_loss

    def cal_latent_loss(X, Y, self, z_mean, z_log_sigma_sq):
        latent_loss = -0.5*(1+z_log_sigma_sq-tf.square(z_mean)-tf.exp(z_log_sigma_sq))
        return latent_loss

    def cal_reconstr_loss_by_rmse(self, X, Y):
        X = tf.reshape(X, [self.batch_size, -1])
        Y = tf.reshape(Y, [self.batch_size, -1])
        reconstr_loss = tf.reduce_mean(tf.pow((X-Y),2))
        return reconstr_loss

    def cal_loss(self, X, Y, z_mean, z_log_sigma_sq):
        return self.cal_reconstr_loss(X, Y), self.cal_latent_loss(X, Y, z_mean, z_log_sigma_sq)

    def get_output(self):
        return self.opt, self.cost

    def train(self, train_molecule):
        opt, cost = self.sess.run([self.opt, self.cost], feed_dict = {self.X:train_molecule})
        return cost

    def test(self, test_molecule):
        opt, Y, cost = self.sess.run([self.opt, self.mol_pred, self.cost], feed_dict = {self.X : test_molecule})
        return Y, cost

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step = global_step)
        print("model saved to '%s'" % (ckpt_path))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate ))

    def get_latent_vector(self, test_molecule):
        return self.sess.run(self.z, feed_dict = {self.X : test_molecule})

    def generate_molecule(self, s_z):
        return self.sess.run(self._X, feed_dict={self.z : s_z}) 

