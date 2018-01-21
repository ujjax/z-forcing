from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import *

def LSTMCell(hidden_dim, num_layers, cell_type = 'LSTM'):
	if cell_type == 'LSTM':
		cell = BasicLSTMCell(hidden_dim)

	else:
		raise ValueError('This cell type is not supported')

	if num_layers>1:
		cell = [cell]*num_layers
		cell = MultiRNNCell(cell)

	return cell







