from tensor
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense


class ValueFunction(object):
    '''Estimate the value function. Uses L-BFGS approach.'''

    def __init__(self, input_dim, session, cfg=None):

        self.session = session
        cfg = cfg if cfg else {}

        # Build the neural network
        self.inputs = tf.placeholder(dtype, [None, input_dim])
        layers = [(64, tf.nn.relu), (1, None)]
        cfg.setdefault('layers', layers)
        cfg.setdefault('gamma', 0.99)
        self.model = SimpleNet(self.inputs, cfg['layers'])

        # Use an LBFGS optimizer (borrowed from SCIPY).
        self.loss = self.model.square_loss
        self.outputs = self.model.outputs
        self.targets = tf.placeholder(dtype, [None, 1])
        self.optimizer = ScipyOptimizerInterface(loss, options={'maxiter':100})

    def fit(self, paths):
        '''Fit the observed returns.'''
        states, returns = [], []
        for path in paths:
            states.append(path['state_vectors'])
            returns.append(discount(path['rewards'], cfg['gamma']))
        feed = {self.inputs: states, self.targets: returns}
        self.optimizer(self.session, feed_dict=feed)

    def predict(self, paths):
        '''Predict baseline values for trajectories.'''
        for path in paths:
            feed = {self.inputs: path['state_vectors']}
            path['baseline'] = self.session.run(self.output, feed_dict=feed)
        return paths




