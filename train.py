# The code is written with the following references: https://github.com/udacity/deep-learning
import os
import sys

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_files
from sklearn.utils import shuffle
from PIL import Image

# Import the neural network architecture defined in model.py
from model import Net

data_dir = 'data/train/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

def load_dataset(path):
	"""
	Sample Usage: train_files, train_targets = load_dataset(train_dir)
	"""
	data = load_files(path)
	files = np.array(data['filenames'])
	targets = tf.keras.utils.to_categorical(np.array(data['target']))
	return files, targets

def get_minibatch(x, y, batch_size = 64):
	"""
	Sample Usage: for x, y in get_minibatch(val_inputs_, val_targets): ##Do stuff with x, y
	"""
	start = 0
	x, y = shuffle(x, y)
	while True:
		idx = start
		start += batch_size
		if start > x.shape[0]:
			break
		yield x[idx:idx+batch_size], y[idx:idx+batch_size]

def acc(pred, labels):
	"""
	Returns prediction accuracy given the predicted probability and the label
	Usage: accuracy = acc(pred, y)
	"""
	equals = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
	acc = tf.reduce_mean(tf.cast(equals, tf.float32))
	return acc

def train(net, lr, epochs, out_classes, inputs_, targets, val_inputs_, val_targets, batch_size = 64, load_file = False, save_freq = save_freq):
	counter = 0
	save_file = './model.ckpt'
	with tf.Session() as sess:
		if load_file:
			saver.restore(sess, save_file)
		else:
			sess.run(tf.global_variables_initializer())
		for e in range(epochs):
			running_loss = 0
			batch_count = 0
			for x, y in get_minibatch(inputs_, targets, batch_size = batch_size):
				counter += 1
				feed_dict = {net.inputs:x, net.labels:y, net.lr: lr, net.is_training: True}
				_, loss = sess.run([net.opt, net.loss], feed_dict = feed_dict)
				running_loss += loss
				batch_count += 1
				print('\rCurrent Batch: {}/{} Loss: {:4f}'.format(batch_count, inputs_.shape[0]//batch_size, 
										loss), end='')
				sys.stdout.flush()
				if counter % save_freq == 0:
					saver.save(sess, 'checkponits/i{}.ckpt'.format(counter))
					print('step: {}, saved file'.format(counter))
			running_loss /= (inputs_.shape[0]/batch_size)
			val_loss, accuracy = 0, 0
			for x, y in get_minibatch(val_inputs_, val_targets, batch_size = batch_size):
				feed_dict = {net.inputs:x, net.labels:y, net.is_training : False}
				val_loss += sess.run(net.loss, feed_dict = feed_dict)
				accuracy += sess.run(acc(net.predict(net.inputs, net.is_training), y), feed_dict=feed_dict)
			val_loss /= (val_inputs_.shape[0]/batch_size)
			accuracy /= (val_inputs_.shape[0]/batch_size)
			print('\rEpoch: {}, Running Loss: {:4f}, Val Loss: {:4f}, Val Acc: {:3f}'.format(e+1, running_loss, 
				val_loss, accuracy))
			sys.stdout.flush()
		saver.save(sess, 'checkpoints/i{}.ckpt'.format(counter))

train_dir = 'data/train/'
#test_dir = 'data/test/'
val_dir = 'data/valid'

train_files, train_targets = load_dataset(train_dir)
val_files, val_targets = load_dataset(val_dir)
#test_files, test_targets = load_dataset(test_dir)

height = 224
width = 224
train_x = np.array([np.array(Image.open(train_file).resize(
		    (height, width), Image.ANTIALIAS)) for train_file in train_files])/255.0
val_x = np.array([np.array(Image.open(val_file).resize(
		    (height, width), Image.ANTIALIAS)) for val_file in val_files])/255.0

# Train the model
model = Net(train_x.shape[1:], train_targets.shape[1], lr=0.01, is_training=True)
saver = tf.train.Saver(max_to_keep=100)
train(model, 0.00001, 1000, train_targets.shape[1], train_x, train_targets, val_x, val_targets, batch_size = 64, load_file = True, save_freq = 500)

