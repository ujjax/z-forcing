from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

from lstm_cell import *

def log_prob_gaussian(x, mu, log_vars, mean=False):
    lp = - 0.5 * math.log(2 * math.pi) \
        - log_vars / 2 - (x - mu) ** 2 / (2 * tf.exp(log_vars))
    if mean:
        return tf.reduce_mean(lp, -1)
    return tf.reduce_sum(lp, -1)

def log_prob_bernoulli(x, mu):
    lp = x * tf.log(mu + 1e-5) + (1. - y) * tf.log(1. - mu + 1e-5)
    return lp


def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (tf.exp(logvar_left) / tf.exp(logvar_right)) +
                        ((mu_left - mu_right)**2.0 / tf.exp(logvar_right)) - 1.0)
    assert len(gauss_klds.size()) == 2
    return tf.reduce_sum(gauss_klds, 1)


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

		self.bwd_mod = LSTMCell(self.embedding_dim, self.rnn_dim, self.num_layers)
		if not cond_ln:
			self.fwd_mod = LSTMCell(self.rnn_dim)
		else:
			self.fwd_mod = LayerNormBasicLSTMCell(self.rnn_dim)
	

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
		temp = self._linear(inputs, self.mlp_dim)
		temp = self._LReLU(temp)
		temp = self._linear(temp, self.z_dim * 2)
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
	"""
	def reparametrize(self, mu, logvar, eps=None):
        std = logvar.mul(0.5).exp_()
        if eps is None:
            eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    """


	def fwd_pass(self, x_fwd, hidden, bwd_states=None, z_step=None):
		with tf.variable_scope('forward-pass'):
			self.x_fwd = tf.nn.embedding_lookup(self.embedding, x_fwd)
			n_steps = tf.shape(self.x_fwd)[0]
			states = [(hidden[0][0], hidden[1][0])]
			klds, zs, log_pz, log_qz, aux_cs = [], [], [], [], []

			eps = tf.Variable(tf.random_normal())




			assert (z_step is None) or (n_steps == 1)

			for step in n_steps:
				states_step = states[step]
				x_step = self.x_fwd[step]
				h_step, c_step = states_step[0], states_step[1]
            	r_step = eps[step]

            	pri_params = self._pri_mod(h_step)
	            pri_params = tf.clip_by_value(pri_params, -8., 8.)
	            pri_mu, pri_logvar = tf.split(pri_params, 2, axis = 1)

	            if bwd_states is not None:
	            	b_step = bwd_states[step]
	            	inf_params = self._inf_mod(tf.concat(h_step, b_step, axis =1))
	            	inf_params = tf.clip_by_value(inf_params, -8. , 8.)
	            	inf_mu, inf_logvar = tf.split(inf_params, 2, axis = 1)
	            	kld = gaussian_kld(inf_mu, inf_logvar, pri_mu, pri_logvar)
	            	z_step = self.reparametrize(inf_mu, inf_logvar, eps=r_step)
	                
	                if self.z_force:
	                    h_step_ = h_step * 0.
	                else:
	                    h_step_ = h_step

	                aux_params = self.aux_mod(tf.concat(h_step_, z_step, axis =1))
	                aux_params = tf.clip_by_value(aux_params, -8., 8.)
	                aux_mu, aux_logvar = tf.split(aux_params, 2, axis = 1)

	                b_step_ = tf.stop_gradient(b_step)

	                if self.use_l2:
	                	aux_step = tf.reduce_sum((b_step_ - tf.tanh(aux_mu)) ** 2.0, 1)
	                else:
	                	aux_step = -log_prob_gaussian(
                            b_step_, tf.tanh(aux_mu), aux_logvar, mean=False)

	            else:
	            	if z_step is None:
	            		z_step = self.reparametrize(pri_mu, pri_logvar, eps=r_step)

	            	aux_step = tf.reduce_sum(pri_mu * 0., -1)
	            	inf_mu, inf_logvar = pri_mu, pri_logvar
                	kld = aux_step
	
                i_step = self._gen_mod(z_step)
                
                if self.cond_ln:
                	i_step = tf.clip_by_value(i_step, -3, 3)
	                gain_hh, bias_hh = tf.split(i_step, 2, axis = 1)
	                gain_hh = 1. + gain_hh
	                h_new, c_new = self.fwd_mod(x_step, (h_step, c_step),
	                                            gain_hh=gain_hh, bias_hh=bias_hh)

	            else:
	            	h_new, c_new = self.fwd_mod(tf.concat(i_step, x_step, axis = 1),
                                            (h_step, c_step))

	            states.append((h_new, c_new))
	            klds.append(kld)
	            zs.append(z_step)
	            aux_cs.append(aux_step)
	            log_pz.append(log_prob_gaussian(z_step, pri_mu, pri_logvar))
            	log_qz.append(log_prob_gaussian(z_step, inf_mu, inf_logvar))

            klds = tf.stack(klds, 0)
	        aux_cs = tf.stack(aux_cs, 0)
	        log_pz = tf.stack(log_pz, 0)
	        log_qz = tf.stack(log_qz, 0)
	        zs = tf.stack(zs, 0)

	        outputs = [s[0] for s in states[1:]]
	        outputs = tf.stack(outputs, 0)
	        outputs = self.fwd_out_mod(outputs)
	        return outputs, states[1:], klds, aux_cs, zs, log_pz, log_qz

	    def infer(self, x, hidden):
	    	x_ = x[:-1]
	        y_ = x[1:]
	        bwd_states, bwd_outputs = self.bwd_pass(x_, y_, hidden)
	        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
	                x_, hidden, bwd_states=bwd_states)
	        return zs

	    def bwd_pass(self, x, y, hidden):
	        idx = np.arange(y.size(0))[::-1].tolist()
	        idx = torch.LongTensor(idx)
	        idx = Variable(idx).cuda()

	        # invert the targets and revert back
	        x_bwd = y.index_select(0, idx)
	        x_bwd = tf.concat(x_bwd, x[:1], axis = 0)
	        x_bwd = self.emb_mod(x_bwd)
	        states, _ = self.bwd_mod(x_bwd, hidden)
	        outputs = self.bwd_out_mod(states[:-1])
	        states = states.index_select(0, idx)
	        outputs = outputs.index_select(0, idx)
	        return states, outputs

	    def forward(self, x, y, x_mask, hidden, return_stats=False):
	        nsteps, nbatch = tf.shape(x)[0], tf.shape(x)[1]
	        bwd_states, bwd_outputs = self.bwd_pass(x, y, hidden)
	        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
	            x, hidden, bwd_states=bwd_states)
	        kld = tf.reduce_sum((klds * x_mask),axis=0)
	        log_pz = tf.reduce_sum(log_pz * x_mask , axis =0)
	        log_qz = tf.reduce_sum(log_qz * x_mask, axis =0)
	        aux_nll = tf.reduce_sum(aux_nll * x_mask, axis =0)
	        
	        if self.out_type == 'gaussian':
	            out_mu, out_logvar = tf.split(fwd_outputs, 2, axis = -1)
	            fwd_nll = -log_prob_gaussian(y, out_mu, out_logvar)
	            fwd_nll = tf.reduce_sum(fwd_nll * x_mask, axis =0)
	            out_mu, out_logvar = tf.split(bwd_outputs, 2, axis = -1)
	            bwd_nll = -log_prob_gaussian(x, out_mu, out_logvar)
	            bwd_nll = tf.reduce_sum(bwd_nll * x_mask, axis =0)

	        elif self.out_type == 'softmax':
	            fwd_out = tf.reshape(fwd_outputs,[nsteps * nbatch, self.out_dim])   
	            fwd_out = tf.nn.log_softmax(fwd_out)
	            y = tf.reshape(y,[-1, 1])
	            fwd_nll = tf.squeeze(torch.gather(fwd_out, y, axis =1), axis =1)
	            fwd_nll = tf.reshape(fwd_nll,[nsteps, nbatch])
	            fwd_nll = tf.reduce_sum(-(fwd_nll * x_mask),axis=0)
	            bwd_out = tf.reshape(bwd_outputs,[nsteps * nbatch, self.out_dim])
	            bwd_out = tf.nn.log_softmax(bwd_out)
	            x = tf.reshape(x, [-1, 1])
	            bwd_nll = tf.squeeze(torch.gather(bwd_out, x, axis =1), axis=1)
	            bwd_nll = tf.reshape(-bwd_nll,[nsteps, nbatch])
	            bwd_nll = tf.reduce_sum((bwd_nll * x_mask),axis =0)
	        
	        if return_stats:        
	            return fwd_nll, bwd_nll, aux_nll, kld, log_pz, log_qz

	        return tf.reduce_mean(fwd_nll), tf.reduce_mean(bwd_nll), tf.reduce_mean(aux_nll), tf.reduce_mean(kld_nll)



