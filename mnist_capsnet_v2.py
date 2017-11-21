#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist_capsnet.py

import os
import argparse
import tensorflow as tf
from capsLayer import *
"""
MNIST CapsNet example.
"""

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset

IMAGE_SIZE = 28
BATCH_SIZE = 128

# Grab from cfg.py
m_plus 			= 0.9
m_minus 		= 0.9
lambda_val 		= 0.5
iter_routing 	= 3
mask_with_y 	= True
epsilon 		= 1e-9
regularization_scale = 0.392



class Model(ModelDesc):
	def _get_inputs(self):
		"""
		Define all the inputs (with type, shape, name) that
		the graph will need.
		"""
		return [InputDesc(tf.float32, 	(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE), 'input'),
				InputDesc(tf.int32, 	(BATCH_SIZE,), 'label')]

	def _build_graph(self, inputs):
		"""This function should build the model which takes the input variables
		and define self.cost at the end"""

		# inputs contains a list of input variables defined above
		image, label = inputs
		print image.get_shape()
		print label.get_shape()
		X = image
		Y = tf.one_hot(label, depth=10, axis=1, dtype=tf.float32)
		# In tensorflow, inputs to convolution function are assumed to be
		# NHWC. Add a single channel here.
		X = tf.expand_dims(X, 3)

		X = X * 2 - 1   # center the pixels values at zero
		# First convolutional layer, return [batch_size, 20, 20, 256]
		with tf.variable_scope('Conv1_layer'):
			# Conv1, [batch_size, 20, 20, 256]
			conv1 = tf.layers.conv2d(X, filters=256,
						 kernel_size=9, strides=(1,1),
						 padding='VALID')
			# print conv1.get_shape()
			assert conv1.get_shape() == [BATCH_SIZE, 20, 20, 256]

		# Primary Capsules layer, return [batch_size, 1152, 8, 1]
		with tf.variable_scope('PrimaryCaps_layer'):
			primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
			caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
			# print caps1.get_shape()
			assert caps1.get_shape() == [BATCH_SIZE, 1152, 8, 1]

		# DigitCaps layer, return [batch_size, 10, 16, 1]
		with tf.variable_scope('DigitCaps_layer'):
			digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
			caps2 = digitCaps(caps1)

		# Decoder structure in Fig. 2
		# 1. Do masking, how:
		with tf.variable_scope('Masking'):
			# a). calc ||v_c||, then do softmax(||v_c||)
			# [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
			v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2),
							 axis=2, keep_dims=True) + epsilon)
			softmax_v = tf.nn.softmax(v_length, dim=1)
			assert softmax_v.get_shape() == [BATCH_SIZE, 10, 1, 1]

			# b). pick out the index of max softmax val of the 10 caps
			# [batch_size, 10, 1, 1] => [batch_size] (index)
			argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
			assert argmax_idx.get_shape() == [BATCH_SIZE, 1, 1]
			argmax_idx = tf.reshape(argmax_idx, shape=(BATCH_SIZE, ))

			# Method 1.
			if not mask_with_y:
				# c). indexing
				# It's not easy to understand the indexing process with argmax_idx
				# as we are 3-dim animal
				masked_v = []
				for batch_size in range(BATCH_SIZE):
					v = caps2[batch_size][argmax_idx[batch_size], :]
					masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

				masked_v = tf.concat(masked_v, axis=0)
				assert masked_v.get_shape() == [BATCH_SIZE, 1, 16, 1]
			# Method 2. masking with true label, default mode
			else:
				# masked_v = tf.matmul(tf.squeeze(caps2), tf.reshape(Y, (-1, 10, 1)), transpose_a=True)
				masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(Y, (-1, 10, 1)))
				v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)

		# 2. Reconstructe the MNIST images with 3 FC layers
		# [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
		with tf.variable_scope('Decoder'):
			vector_j = tf.reshape(masked_v, shape=(BATCH_SIZE, -1))
			fc1 = tf.layers.dense(inputs=vector_j, units=512)
			assert fc1.get_shape() == [BATCH_SIZE, 512]
			fc2 = tf.layers.dense(inputs=fc1, units=1024)
			assert fc2.get_shape() == [BATCH_SIZE, 1024]
			decoded = tf.layers.dense(inputs=fc2, units=784, activation=tf.tanh)


		# Build loss
		# 1. The margin loss

		# [batch_size, 10, 1, 1]
		# max_l = max(0, m_plus-||v_c||)^2
		max_l = tf.square(tf.maximum(0., m_plus - v_length))
		# max_r = max(0, ||v_c||-m_minus)^2
		max_r = tf.square(tf.maximum(0., v_length - m_minus))
		assert max_l.get_shape() == [BATCH_SIZE, 10, 1, 1]

		# reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
		max_l = tf.reshape(max_l, shape=(BATCH_SIZE, -1))
		max_r = tf.reshape(max_r, shape=(BATCH_SIZE, -1))

		# calc T_c: [batch_size, 10]
		# T_c = Y, is my understanding correct? Try it.
		T_c = Y
		# [batch_size, 10], element-wise multiply
		L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

		margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1), name='margin_loss')

		# 2. The reconstruction loss
		orgin = tf.reshape(X, shape=(BATCH_SIZE, -1))
		squared = tf.square(decoded - orgin)
		reconstruction_err = tf.reduce_mean(squared, name='reconstruction_err')

		# 3. Total loss
		# The paper uses sum of squared error as reconstruction error, but we
		# have used reduce_mean in `# 2 The reconstruction loss` to calculate
		# mean squared error. In order to keep in line with the paper,the
		# regularization scale should be 0.0005*784=0.392
		total_loss = tf.identity(margin_loss + regularization_scale * reconstruction_err, name='total_loss')
		self.cost = total_loss



		# summary
		summary.add_moving_summary(margin_loss)
		summary.add_moving_summary(reconstruction_err)
		summary.add_moving_summary(total_loss)
		recon_img = tf.reshape(decoded, shape=(BATCH_SIZE, 28, 28, 1))
		recon_img = 255*(recon_img+1.0)/2.0
		recon_img = tf.clip_by_value(recon_img, 0, 255)
		recon_img = tf.cast(recon_img, tf.uint8, name='recon_img')
		tf.summary.image('reconstruction_img', recon_img, max_outputs=50)
		# tf.summary.image('reconstruction_img', recon_img, max_outputs=50)

		correct_prediction = tf.equal(tf.to_int32(label), argmax_idx)
		batch_accuracy 	   = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
		# test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])

	def _get_optimizer(self):
		lr = tf.train.exponential_decay(
			learning_rate=1e-3,
			global_step=get_global_step_var(),
			decay_steps=468 * 10,
			decay_rate=0.3, staircase=True, name='learning_rate')
		# This will also put the summary in tensorboard, stat.json and print in terminal
		# but this time without moving average
		tf.summary.scalar('lr', lr)
		return tf.train.AdamOptimizer(lr)


def get_data():
	train = BatchData(dataset.Mnist('train'), BATCH_SIZE)
	test =  BatchData(dataset.Mnist('test'), BATCH_SIZE, remainder=False)
	train = PrefetchDataZMQ(train, BATCH_SIZE)
	test  = PrefetchDataZMQ(test, BATCH_SIZE)
	return train, test


def get_config():
	dataset_train, dataset_test = get_data()

	# How many iterations you want in each epoch.
	# This is the default value, don't actually need to set it in the config
	steps_per_epoch = dataset_train.size()

	# get the config which contains everything necessary in a training
	return TrainConfig(
		model=Model(),
		dataflow=dataset_train,  # the DataFlow instance for training
		callbacks=[
			ModelSaver(),   # save the model after every epoch
			MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
			InferenceRunner(    # run inference(for validation) after every epoch
				dataset_test,   # the DataFlow instance used for validation
				ScalarStats([
							 'margin_loss', 
							 'reconstruction_err', 
							 'total_loss'])),
		],
		steps_per_epoch=steps_per_epoch,
		max_epoch=100,
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', help='load model')
	args = parser.parse_args()
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	# automatically setup the directory train_log/mnist-convnet for logging
	logger.auto_set_dir()

	config = get_config()
	if args.load:
		config.session_init = SaverRestore(args.load)
	# SimpleTrainer is slow, this is just a demo.
	# You can use QueueInputTrainer instead
	# launch_train_with_config(config, SimpleTrainer())
	nr_gpu = len(args.gpu.split(','))
	trainer = QueueInputTrainer() if nr_gpu <= 1 \
		else SyncMultiGPUTrainerReplicated(nr_gpu)
	launch_train_with_config(config, trainer)
