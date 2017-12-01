#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist_capsnet.py

import os
import argparse
import numpy as np
import tensorflow as tf

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

np.random.seed(2017)
tf.set_random_seed(2017)
###################################################################################################
class Model(ModelDesc):
	def _get_inputs(self):
		"""
		Define all the inputs (with type, shape, name) that
		the graph will need.
		"""
		return [InputDesc(tf.float32, 	(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE), 'input'),
				InputDesc(tf.int32, 	(BATCH_SIZE,), 'label')]
	###############################################################################################			
	def _build_graph(self, inputs):
		"""This function should build the model which takes the input variables
		and define self.cost at the end"""

		### Input Images
		# inputs contains a list of input variables defined above
		image, label = inputs
		print image.get_shape()
		print label.get_shape()
		X = image
		Y = tf.one_hot(label, depth=10, axis=1, dtype=tf.float32)
		# In tensorflow, inputs to convolution function are assumed to be
		# NHWC. Add a single channel here.
		X = tf.expand_dims(X, 3)

		X = X/255.0   # Normalize between 0 and 1
		
		#
		### Primary Capsules
		#
		"""
		The first layer will be composed of 32 maps of 6x6 capsules each, 
		where each capsule will putput an 8D activation vector
		"""
		caps1_n_maps = 32
		caps1_n_caps = 6 * 6 * caps1_n_maps # 1152 primamry capsules
		caps1_n_dims = 8 

		# To compute their outputs, we first apply two regular convolutional layers
		conv1_params = {
			"filters"		:	256,
			"kernel_size"	:	9,
			"strides"		:	1,
			"padding"		:	"valid",
			"activation"	: 	tf.nn.relu,
		}

		conv2_params = {
			"filters"		: 	caps1_n_maps * caps1_n_dims,  # 32 * 8 = 256 convolutional filters
			"kernel_size"	:	9,
			"strides"		: 	2,
			"padding"		: 	"valid",
			"activation"	:	tf.nn.relu,
		}

		conv1 = tf.layers.conv2d(name="conv1", **conv1_params, inputs=X)
		conv2 = tf.layers.conv2d(name="conv2", **conv2_params, inputs=conv1)
		"""
		Note: since we used a kernel size of 9 and no padding, the image shrunk by 9-1=8 pixels
		28x28 to 20x20, 20x20 to 12x12
		and since we used a stride of 2 in the second convolutional layer,
		we end up with 6x6 feature maps (6x6 vector output)
		"""

		"""
		Next we reshape the output to get a bunch of 8D vectors representing the output of the 
		primary capsules. The output of conv2 is an array containing 32x8=256 feature maps for
		each instance, where each feature map is 6x6. So the shape of this output is (batch_size, 
		6, 6, 256).

		We can reshape to (batch_size, 6, 6, 32, 8) to divide 256 into 32 vectors of 8 dimension each.
		However, since the first capsule layer will be fully connected to the next capsule layer, 
		we can simply flatten the 6x6 grids. Equivalenly, we just need to reshape to (batch_size, 
		6x6x32, 8)
		"""
		caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")


		"""
		We need to squash these vectors. Let us define the squash function, based on the equation.
		The squash() function will squash all the vectors in the given array, along a given axis (by 
		default, the last axis).

		Caution, a nasty bug is waiting to bite you: the derivative of ||s|| is undefined when ||s|| = 0, 
		so we can not just use tf.norm(), or else. The solution is to compute the safe_norm
		"""
		def squash(s, axis=-1, epsilon=1e-7, name=None):
			with tf.name_scope(name, default_name='squash'):
				squared_norm 	= tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
				safe_norm 		= tf.sqrt(squared_norm+epsilon)
				squash_vector 	= squared_norm / (1.0 + squared_norm)
				unit_vector 	= s / safe_norm
				return squash_vector * unit_vector

		"""
		Now let us apply this function the get the ouput u_i of each primary capsule i
		"""
		caps1_out = squash(caps1_raw, name="caps1_output")


		self.cost = tf.identity(0., name='total_costs')
		summary.add_moving_summary(self.cost)
	###############################################################################################
	def _get_optimizer(self):
		lr = tf.get_variable('learning_rate', initializer=5e-3, trainable=False)
		return tf.train.AdamOptimizer(lr)


###################################################################################################
def get_data():
	train = BatchData(dataset.Mnist('train'), BATCH_SIZE)
	test  = BatchData(dataset.Mnist('test'),  BATCH_SIZE, remainder=False)
	# print(np.max(train))
	train = PrintData(train)
	test  = PrintData(test)
	train = PrefetchDataZMQ(train, BATCH_SIZE)
	test  = PrefetchDataZMQ(test,  BATCH_SIZE)
	return train, test

###################################################################################################
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

###################################################################################################
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
