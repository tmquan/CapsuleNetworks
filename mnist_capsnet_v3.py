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

		X = X/255.0   # center the pixels values at zero
		


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
