# Reference: https://arxiv.org/abs/1602.07360
#            https://github.com/Tandon-A/SqueezeNet
import tensorflow as tf


def fire_sim(inputs, sq_struct, exp_struct, name='fire', is_training = False):
#Sample sq_struct: (64, 1, 1)
#Sample exp_struct: ((64, 1),(64, 3)) //stride = 1 for all layers to allow concatenating
	w_init = tf.truncated_normal_initializer(mean=0.0, stddev=(1.0/int(inputs.shape[2])))
	with tf.variable_scope(name):
		sq_f, sq_k, sq_s = sq_struct
		sq = tf.layers.conv2d(inputs, filters=sq_f, kernel_size=sq_k, 
				      strides=sq_s, padding='SAME', kernel_initializer=w_init)
		expand = []
		for es in exp_struct:
			k_exp = tf.layers.conv2d(sq, filters=es[0], kernel_size=es[1], strides=1, padding='SAME', 
						kernel_initializer=w_init)
			k_batch = tf.layers.batch_normalization(k_exp, training = is_training)
			k_relu = tf.nn.relu(k_exp)
			expand.append(k_relu)
		return tf.concat(expand, axis=3)

class Net():
	def __init__(self, input_shape, out_classes, lr, is_training):
		self.lr = tf.placeholder(tf.float32, name='lr')
		self.out_classes = out_classes
		if len(input_shape) == 3:
			self.inputs = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], input_shape[2]))
		else:
			self.inputs = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1]))
		self.labels = tf.placeholder(tf.float32, shape=(None, self.out_classes))
		self.is_training = tf.placeholder(tf.bool)
		self.loss = self.model_loss(self.inputs, self.labels, self.is_training)
		self.opt = self.optimize(self.loss, self.lr)
	def model(self, inputs, is_training, reuse=False):
		with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(inputs, filters = 96, kernel_size = 7, strides = 2, padding = 'SAME', activation='relu')
			pool1 = tf.layers.max_pooling2d(conv1, pool_size = 3, strides = 2)

			fire_sim1 = fire_sim(pool1, (16, 1, 1), ((64,1),(64,3)), name = 'fire_sim1', is_training = is_training)
			fire_sim2 = fire_sim(fire_sim1, (16, 1, 1), ((64,1),(64,3)), name = 'fire_sim2')
			fire_sim3 = fire_sim(fire_sim2, (32, 1, 1), ((128,1),(128,3)), name = 'fire_sim3', is_training = is_training)
			# bypass_23 = tf.add(fire2, fire3, name='bypass_23')
			
			pool2 = tf.layers.max_pooling2d(fire_sim3, pool_size=3, strides=2, name='pool2')

			fire_sim4 = fire_sim(pool2, (32, 1, 1), ((128,1),(128,3)), name = 'fire_sim4', is_training = is_training)
			fire_sim5 = fire_sim(fire_sim4, (48, 1, 1), ((192,1),(192,3)), name = 'fire_sim5', is_training = is_training)
			fire_sim6 = fire_sim(fire_sim5, (48, 1, 1), ((192,1),(192,3)), name = 'fire_sim6', is_training = is_training)
			fire_sim7 = fire_sim(fire_sim6, (64, 1, 1), ((256,1),(256,3)), name = 'fire_sim7', is_training = is_training)

			pool3 = tf.layers.max_pooling2d(fire_sim7, pool_size=3, strides=2, name='pool3')

			fire_sim8 = fire_sim(pool3, (64, 1, 1), ((256,1),(256,3)), name = 'fire_sim8', is_training = is_training)
			drop = tf.layers.dropout(fire_sim8, rate = 0.5, training = is_training)

			conv2 = tf.layers.conv2d(drop, filters = 200, kernel_size = 1, strides = 1, padding = 'SAME', activation='relu')
			avg_pool = tf.layers.average_pooling2d(conv2, pool_size=13, strides=1, name='pool_end')
			avg_pool = tf.layers.flatten(avg_pool)
			logits = tf.layers.dense(avg_pool, self.out_classes)
			return logits
	def model_loss(self, inputs, label, is_training):
		logits = self.model(inputs, is_training)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label))
		return loss
	def predict(self, inputs, is_training):
		logits = self.model(inputs, is_training)
		prediction = tf.nn.softmax(logits)
		return prediction
	def optimize(self, loss, lr):
		opt = tf.train.AdamOptimizer(lr).minimize(loss)
		return opt


