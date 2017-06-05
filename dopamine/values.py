import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np
from dopamine.utils import *
from dopamine.net import *
from ipdb import set_trace as debug


class ValueFunction(object):
    '''Estimate the value function. Uses L-BFGS approach.'''

    def __init__(self, input_dim, session, cfg=None):

        self.session = session
        cfg = cfg if cfg else {}
        self.cfg = cfg

        # Build the neural network
        self.inputs = tf.placeholder(dtype, [None, input_dim])
        layers = [(64, tf.nn.relu), (1, None)]
        cfg.setdefault('max_lbfgs_iters', 125)
        cfg.setdefault('layers', layers)
        cfg.setdefault('gamma', 0.99)
        self.model = SimpleNet(self.inputs, cfg['layers'])

        # Use an LBFGS optimizer (borrowed from SCIPY).
        self.loss = loss = self.model.square_loss
        self.outputs = self.model.outputs
        self.targets = self.model.targets
        maxiters = self.cfg['max_lbfgs_iters']
        self.optimizer = ScipyOptimizerInterface(loss, \
                options={'maxiter':maxiters})

        init = tf.global_variables_initializer()
        self.session.run(init)

    def fit(self, paths):
        '''Fit the observed returns.'''
        states, returns = [], []
        for path in paths:
            states.append(path['state_vectors'])
            returns.append(discount(path['rewards'], self.cfg['gamma']))
        states = np.vstack(states)
        returns = np.vstack(returns)
        feed = {self.inputs: states, self.targets: returns}
        self.optimizer.minimize(self.session, feed_dict=feed)

    def predict(self, paths):
        '''Predict baseline values for trajectories.'''
        for path in paths:
            feed = {self.inputs: path['state_vectors']}
            path['baseline'] = self.session.run(self.outputs, feed_dict=feed)
        return paths
