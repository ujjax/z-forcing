from __future__ import print_function

import tensorflow as tf
import numpy as np

from cell import *

def log_prob_gaussian(x, mu, log_vars, mean=False):
    lp = - 0.5 * math.log(2 * math.pi) \
        - log_vars / 2 - (x - mu) ** 2 / (2 * torch.exp(log_vars))
    if mean:
        return torch.mean(lp, -1)
    return torch.sum(lp, -1)

def log_prob_bernoulli(x, mu):
    lp = x * torch.log(mu + 1e-5) + (1. - y) * torch.log(1. - mu + 1e-5)
    return lp



class Z_Forcing(object):
	def __init__(self):
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		self.rnn_dim = rnn_dim
		self.mlp_dim = mlp_dim
		self.z_dim = z_dim
		self.num_layers = num_layers
		self.dropout_keep_prob = dropout_keep_prob
		self.embedding_dropout = embedding_dropout
		self.cond_ln = cond_ln

		if output_type == 'bernoulli' or output_type == 'softmax':
			with tf.variable_scope('embedding_scope'):
				self.embedding = tf.get_variable(name = 'embedding', initializer = self._init_matrix([self.vocab_size,self.embedding_dim]))

			with tf.name_scope('embedding_dropout'):
				self.embedding_matrix = tf.nn.dropout(self.embedding, keep_prob=self.embedding_dropout, noise_shape=[self.vocab_size,1])

		self.backward_mode = LSTMCell(self.embedding_dim, self.rnn_dim, self.num_layers)

		self.forward_mode = LSTMCell(self.embedding_dim if cond_ln else self.embedding_dim + self.mlp_dim, self.rnn_dim, use_layernorm = cond_ln)

	

	self.fwd_out_mod = self._linear(rnn_dim, out_dim)
    self.bwd_out_mod = self._linear(rnn_dim, out_dim)

    def _aux_mod(self, inputs):
    	temp = self._linear(self.z_dim +self.rnn_dim, self.mlp_dim)
    	temp = self._LReLU(temp)
    	return temp = self._linear(self.mlp_dim, 2 * self.rnn_dim)


    def _gen_mod(self, inputs, cond_ln = self.cond_ln):
    	if cond_ln:
	    	temp = self._linear(self.z_dim, self.mlp_dim)
	    	temp = self._LReLU(temp)
	    	return temp = self._linear(self.mlp_dim, 8 * self.rnn_dim)

	    else:
	    	return self._linear(self.z_dim,self.mlp_dim)

	def _inf_mod(self, inputs):
		temp = self._linear(self.rnn_dim * 2, self.mlp_dim)
		temp = self._LReLU(temp)
		return temp = self._linear(mlp_dim, z_dim * 2)


	def _pri_mode(self, inputs):
		temp = self._linear(self.rnn_dim, self.mlp_dim)
		temp = self._LReLU(temp)
		temp = self._linear(mlp_dim, z_dim * 2)
		return temp

	def _LReLU(self, input_tensor ,slope = 1.0/3):
		return tf.clip_by_value(tf.leaky_relu(input_tensor, slope), -3. ,3.)


	def _linear(self, input_tensor, output_dim, scope=None):

		change_dimension = tf.shape(input_tensor)[-1]

		if len(tf.shape(input_tensor))>2:
			input_tensor = tf.reshape(input_tensor, [-1, change_dimension])

		if len(tf.shape(input_tensor))<=1:
			raise ValueError('Shape of input tensor should be greater than 1')

		with tf.variable_scope(scope or "linear"):
	        weight = tf.get_variable("weight", [change_dimension, output_dim], dtype=input_tensor.dtype)
	        bias = tf.get_variable("bias", [output_dim], dtype=input_tensor.dtype)

	    return tf.matmul(input_tensor, weight) + bias


	def _init_matrix(self, shape):
		return tf.random_normal(shape, stddev=0.1)


	def forward_pass(self, x_fwd, hidden, bwd_states=None, z_step=None):
		with tf.variable_scope('forward-pass'):
			self.x_fwd = tf.nn.embedding_lookup(self.embedding, x_fwd)
			n_steps = tf.shape(self.x_fwd)[0]
			states = [(hidden[0][0], hidden[1][0])]
			klds, zs, log_pz, log_qz, aux_cs = [], [], [], [], []

			




			assert (z_step is None) or (n_steps == 1)

			for step in n_steps:
				states_step = states[step]
				x_step = self.x_fwd[step]
				h_step, c_step = states_step[0], states_step[1]
            	r_step = eps[step]

            	pri_params = self.pri_mod(h_step)
	            pri_params = torch.clamp(pri_params, -8., 8.)
	            pri_mu, pri_logvar = torch.chunk(pri_params, 2, 1)


	


