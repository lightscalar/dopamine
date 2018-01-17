import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np
from dopamine.utils import *
from dopamine.net import *
from keras.models import load_model
from keras.layers import Dense
from keras import backend
from ipdb import set_trace as debug


class ValueFunction(object):
    '''Estimate the value function. Uses L-BFGS approach.'''

    def __init__(self, name, session, input_dim=None, load_model=False, cfg=None):

        self.name = name.replace(' ', '_')
        self.session = session
        backend.set_session(session)
        cfg = cfg if cfg else {}
        self.cfg = cfg

        # Build the neural network via KERAS.
        cfg.setdefault('max_lbfgs_iters', 125)
        cfg.setdefault('gamma', 0.995)
        cfg.setdefault('model_file', \
                'weights/values_{:s}.h5'.format(self.name))

        if load_model:
            # Load from disk.
            self.load_model()
        else:
            # Build from scratch.
            self.net = Sequential()
            self.net.add(Dense(input_shape=(input_dim,), units=128,\
                    activation='relu'))
            self.net.add(Dense(units=128, activation='relu'))
            self.net.add(Dense(1, activation='linear'))

        # Defines the ins/outs.
        self.input = self.net.input
        self.output = self.net.output

        # Use an LBFGS optimizer (borrowed from SCIPY).
        self.target = tf.placeholder(dtype, [None, 1])
        self.net.compile(optimizer='rmsprop', loss='mse')
        # self.loss = loss = tf.reduce_mean(tf.squared_difference(\
        #         self.output, self.target))
        # maxiters = self.cfg['max_lbfgs_iters']
        # self.
        # self.optimizer = ScipyOptimizerInterface(loss, \
        #         options={'maxiter':maxiters})

        # Initialize the variables.
        init = tf.global_variables_initializer()
        self.session.run(init)

    def fit(self, paths):
        '''Fit the observed returns.'''
        states, returns = [], []
        nb_paths = int(0.20 * len(paths))
        for path in paths[:nb_paths]:
            states.append(path['state_vectors'])
            returns.append(discount(path['rewards'], self.cfg['gamma']))
        states = np.vstack(states)
        returns = np.vstack(returns)
        feed = {self.input: states, self.target: returns}
        # self.optimizer.minimize(self.session, feed_dict=feed)
        self.net.fit(states, returns, batch_size=50, epochs=10)

    def predict(self, paths):
        '''Predict baseline values for trajectories.'''
        for path in paths:
            feed = {self.input: path['state_vectors']}
            path['baseline'] = self.session.run(self.output, feed_dict=feed)
        return paths

    def save_model(self):
        '''Save current weights/structure to specified file.'''
        self.net.save(self.cfg['model_file'])

    def load_model(self):
        '''Load previously saved weights.'''
        self.net = load_model(self.cfg['model_file'])
