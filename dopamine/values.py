import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np
from dopamine.utils import *
from dopamine.net import *
from ipdb import set_trace as debug
from keras.layers import Dense
from keras import backend


class ValueFunction(object):
    '''Estimate the value function. Uses L-BFGS approach.'''

    def __init__(self, input_dim, session, cfg=None):

        self.session = session
        backend.set_session(session)
        cfg = cfg if cfg else {}
        self.cfg = cfg

        # Build the neural network via KERAS.
        layers = [(64, tf.nn.relu), (1, None)]
        cfg.setdefault('max_lbfgs_iters', 125)
        cfg.setdefault('gamma', 0.99)
        self.net = Sequential()
        self.net.add(Dense(input_shape=(input_dim,), units=64,\
                activation='relu'))
        self.net.add(Dense(1, activation='linear'))
        self.input = self.net.input
        self.output = self.net.output

        # Use an LBFGS optimizer (borrowed from SCIPY).
        self.target = tf.placeholder(dtype, [None, 1])
        self.loss = loss = tf.reduce_mean(tf.squared_difference(\
                self.output, self.target))
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
        feed = {self.input: states, self.target: returns}
        self.optimizer.minimize(self.session, feed_dict=feed)

    def predict(self, paths):
        '''Predict baseline values for trajectories.'''
        for path in paths:
            feed = {self.input: path['state_vectors']}
            path['baseline'] = self.session.run(self.output, feed_dict=feed)
        return paths
