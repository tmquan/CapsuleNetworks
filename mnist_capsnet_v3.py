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
		y = label #
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

		conv1 = tf.layers.conv2d(inputs=X, 		name="conv1", **conv1_params)
		conv2 = tf.layers.conv2d(inputs=conv1, 	name="conv2", **conv2_params)
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
		caps1_output = squash(caps1_raw, name="caps1_output")


		#
		### Digit Capsules
		#
		"""
		To compute the output of the digit capsules, we must first compute the predicted output
		vectors (one for reach primary/digit capsule pair). 
		Then we can run the routing by agreement algorithm. 
		"""
		### Compute the predicted output vectors.
		# The digit capsule layer contains 10 capsules (one for each digit) of 16 dimension each
		caps2_n_caps = 10
		caps2_n_dims = 16

		"""
		For each capsule i in the first layer, we want to predict the output of every capsule j in 
		the second layer. For this, we will need a transformation matrix W_i (one for each pair of
		capsules (i, j)), then we can compute the predicted output u^j|i = W_ij * u_i .
		Since we want to transform an 8D vector into a 16D vector, each transformation W_ij must 
		have a shape (16x8). 

		We can use tf.matmul() to perform matrix-wise multiplication to compute u^j|i for every pair
		of capsules (i, j) 


		The shape of the first array is (1152, 10, 16, 8), and the shape of the second array is (1152, 10, 8, 1). 
		Note that the second array must contain 10 identical copies of the vectors $\mathbf{u}_1$ to $\mathbf{u}_{1152}$. 
		To create this array, we will use the handy tf.tile() function, which lets you create an array containing many copies of a base array, 
		tiled in any way you want.
		Oh, wait a second! We forgot one dimension: batch size. Say we feed 50 images to the capsule network, 
		it will make predictions for these 50 images simultaneously. So the shape of the first array must be 
		(50, 1152, 10, 16, 8), and the shape of the second array must be (50, 1152, 10, 8, 1). 
		The first layer capsules actually already output predictions for all 50 images, so the second array will be fine, 
		but for the first array, we will need to use tf.tile() to have 50 copies of the transformation matrices.

		Okay, let's start by creating a trainable variable of shape (1, 1152, 10, 16, 8) that will hold all the transformation matrices. 
		The first dimension of size 1 will make this array easy to tile. 
		We initialize this variable randomly using a normal distribution with a standard deviation to 0.01.
		"""
		init_sigma 	= 	0.01
		W_init 		= 	tf.random_normal(shape 	=	(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims), 
										 stddev	=	init_sigma, 
										 dtype	=	tf.float32, 
										 name 	= 	"W_init",
							)
		W = tf.get_variable(name="W", initializer=W_init)
		# W 		= 	tf.random_normal(shape 	=	(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims), 
		# 							 stddev	=	init_sigma, 
		# 							 dtype	=	tf.float32, 
		# 							 name 	= 	"W_init",
		# 				)

		###Now we can create the first array by repeating W once per instance:
		batch_size = tf.shape(X)[0]
		W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")


		"""
		As discussed earlier, we need to create an array of shape (batch size, 1152, 10, 8, 1), 
		containing the output of the first layer capsules, repeated 10 times 
		(once per digit, along the third dimension, which is axis=2). 
		The caps1_output array has a shape of (batch size, 1152, 8), 
		so we first need to expand it twice, to get an array of shape (batch size, 1152, 1, 8, 1), 
		then we can repeat it 10 times along the third dimension:
		"""
		caps1_output_expanded 	= tf.expand_dims(caps1_output, -1,
												name="caps1_output_expanded")
		caps1_output_tile 		= tf.expand_dims(caps1_output_expanded, 2,
												name="caps1_output_tile")
		caps1_output_tiled 		= tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
										 name="caps1_output_tiled")

		print(W_tiled)
		print(caps1_output_tiled)

		### Yes! Now, to get all the predicted output vectors $\hat{\mathbf{u}}_{j|i}$, 
		# we just need to multiply these two arrays using tf.matmul(), as explained earlier:
		caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
		print(caps2_predicted)

		# Perfect, for each instance in the batch (we don't know the batch size yet, hence the "?") 
		# and for each pair of first and second layer capsules (1152x10) we have a 16D predicted 
		# output column vector (16×1). We're ready to apply the routing by agreement algorithm!

		#
		# Routing by agreement
		#
		# First, let us initialize the raw routing weights b_ij to zero
		raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], 
						dtype	= 	tf.float32,
						name 	= 	"raw_weights")

		### Round 1
		#First, let's apply the softmax function to compute the routing weights, 
		# c_i = softmax(b_i) (equation (3) in the paper):
		routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

		# Compute the weighted sum
		weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
		weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")

		"""
		There are a couple important details to note here:
		
		To perform elementwise matrix multiplication (also called the Hadamard product, noted $\circ$), 
		we use the tf.multiply() function. It requires routing_weights and caps2_predicted to have the same rank, 
		which is why we added two extra dimensions of size 1 to routing_weights, earlier.
		
		The shape of routing_weights is (batch size, 1152, 10, 1, 1) 
		while the shape of caps2_predicted is (batch size, 1152, 10, 16, 1). 
		Since they don't match on the fourth dimension (1 vs 16), 
		tf.multiply() automatically broadcasts the routing_weights 16 times along that dimension. 
		"""

		# And finally, let us apply the squash function to get the outputs of the second layer 
		# capsules at the end of the first iteration of the routing by agreement algorithm, 
		# v_j = squash(s_j) :
		caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")
		print(caps2_output_round_1)


		### Round 2
		"""
		First, let's measure how close each predicted vector u^_j|i is to the actual output vector v_j 
		by computing their scalar product u^_j|i x v_j.
		Quick math reminder: if $\vec{a}$ and $\vec{b}$ are two vectors of equal length, 
		and $\mathbf{a}$ and $\mathbf{b}$ are their corresponding column vectors (i.e., matrices with a single column), 
		then $\mathbf{a}^T \mathbf{b}$ (i.e., the matrix multiplication of the transpose of $\mathbf{a}$, and $\mathbf{b}$) 
		is a 1x1 matrix containing the scalar product of the two vectors $\vec{a}\cdot\vec{b}$. 
		In Machine Learning, we generally represent vectors as column vectors, so when we talk about computing 
		the scalar product $\hat{\mathbf{u}}_{j|i} \cdot \mathbf{v}_j$, this actually means computing ${\hat{\mathbf{u}}_{j|i}}^T \mathbf{v}_j$.
		Since we need to compute the scalar product $\hat{\mathbf{u}}_{j|i} \cdot \mathbf{v}_j$ for each instance, and for each pair of first and second level capsules $(i, j)$, we will once again take advantage of the fact that tf.matmul() can multiply many matrices simultaneously. This will require playing around with tf.tile() to get all dimensions to match (except for the last 2), just like we did earlier. So let's look at the shape of caps2_predicted, which holds all the predicted output vectors $\hat{\mathbf{u}}_{j|i}$ for each instance and each pair of capsules:
		"""
		print(caps2_predicted) # u^_j|i
		print(caps2_output_round_1) # v_j

		# To get these shapes to match, we just need to tile the caps2_output_round_1 array 1152 times 
		# (once per primary capsule) along the second dimension:	
		caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],  name="caps2_output_round_1_tiled")

		# And now we are ready to call tf.matmul() (note that we must tell it to transpose the matrices in the first array, 
		# to get ${\hat{\mathbf{u}}_{j|i}}^T$ instead of $\hat{\mathbf{u}}_{j|i}$):
		agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
							  transpose_a=True, name="agreement")


		"""We can now update the raw routing weights $b_{i,j}$ by simply adding the scalar product 
		$\hat{\mathbf{u}}_{j|i} \cdot \mathbf{v}_j$ we just computed: 
		$b_{i,j} \gets b_{i,j} + \hat{\mathbf{u}}_{j|i} \cdot \mathbf{v}_j$ (see Procedure 1, step 7, in the paper).
		"""
		raw_weights_round_2 = tf.add(raw_weights, agreement, name="raw_weights_round_2")
		routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2, dim=2, name="routing_weights_round_2")
		weighted_predictions_round_2 = tf.multiply(routing_weights_round_2, caps2_predicted, name="weighted_predictions_round_2")
		weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2, axis=1, keep_dims=True, name="weighted_sum_round_2")
		caps2_output_round_2 = squash(weighted_sum_round_2, axis=-2, name="caps2_output_round_2")
		# We could go on for a few more rounds, by repeating exactly the same steps as in round 2
		caps2_output = caps2_output_round_2

		#
		# Estimated Class Probabilities (Length)
		#
		# The lengths of the output vectors represent the class probabilities, 
		# so we could just use tf.norm() to compute them, but as we saw when discussing the squash function, 
		# it would be risky, so instead let's create our own safe_norm() function:
		def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
			with tf.name_scope(name, default_name="safe_norm"):
				squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
				return tf.sqrt(squared_norm + epsilon)

		y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")		

		# To predict the class of each instance, we can just select the one with the highest estimated probability. 
		# To do this, let us start by finding its index using tf.argmax():		
		y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
		print(y_proba_argmax)

		# That's what we wanted: for each instance, we now have the index of the longest output vector. 
		# Let us get rid of the last two dimensions by using tf.squeeze() which removes dimensions of size 1. 
		# This gives us the capsule network's predicted class for each instance:
		y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
		print(y_pred)


		#
		# Compute the loss
		#
		"""
		Margin loss
		"""
		m_plus = 0.9
		m_minus = 0.1
		lambda_ = 0.5

		# Since y will contain the digit classes, from 0 to 9, to get $T_k$ for every instance and every class, 
		# we can just use the tf.one_hot() function:
		T = tf.one_hot(y, depth=caps2_n_caps, name="T")


		# Now let's compute the norm of the output vector for each output capsule and each instance. 
		# First, let's verify the shape of caps2_output:
		print caps2_output

		# The 16D output vectors are in the second to last dimension, 
		# so let's use the safe_norm() function with axis=-2:
		caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm")

		# Now let's compute $\max(0, m^{+} - \|\mathbf{v}_k\|)^2$, 
		# and reshape the result to get a simple matrix of shape (batch size, 10):
		present_error_raw 	= tf.square(tf.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")
		present_error 	 	= tf.reshape(present_error_raw, shape=(-1, 10), name="present_error")

		# Next let's compute $\max(0, \|\mathbf{v}_k\| - m^{-})^2$ and reshape it:
		absent_error_raw 	= tf.square(tf.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
		absent_error 		= tf.reshape(absent_error_raw, shape=(-1, 10), name="absent_error")


		# We are ready to compute the loss for each instance and each digit:
		L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")

		# Now we can sum the digit losses for each instance ($L_0 + L_1 + ... + L_9$), 
		# and compute the mean over all instances. This gives us the final margin loss:
		margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")



		"""
		Reconstruction
		"""
		# Now let's add a decoder network on top of the capsule network. 
		# It is a regular 3-layer fully connected neural network which will learn to reconstruct the input images 
		# based on the output of the capsule network. 
		# This will force the capsule network to preserve all the information required to reconstruct the digits, across the whole network. 
		# This constraint regularizes the model: it reduces the risk of overfitting the training set, and it helps generalize to new digits.
		

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
