import numpy as np
import tensorflow as tf
from dopamine.net import SimpleNet
from dopamine.utils import *
import pylab as plt


# Specify data type for the tensors.
dtype = 'float32'


class ValueFunction(object):
    '''Creates a simple neural network to estimate value function.'''


    def __init__(self, session, state_space_dimension):
        '''Only parameter is the state space dimension.'''

        self.session = session

        # Dimension of state space.
        self.D_in = state_space_dimension

        # Create placeholders for inputs (x) and outputs (y).
        self.x = tf.placeholder(dtype, [None,self.D_in])
        self.y = tf.placeholder(dtype, [None,1]) # scalar output

        # Create our neural network.
        layers = [(64, tf.nn.relu), (64, tf.nn.relu), (1, None)]
        self.net = SimpleNet(self.x, layers)


    def __call__(self, x):
        '''Execute the neural network on the inputs x.'''
        return self.net.predict(x, self.session)

    def fit(self, x, y):
        '''Fit network to the supplied data.'''
        pass

    @property
    def params(self):
        '''Return the trainable parameters.'''
        return self.net.vars

    @property
    def theta(self):
        '''Flattened version of all parameters.'''
        return self.get_flat()

    @property
    def D_theta(self):
        '''Return dimension of network parameter vector.'''
        return self.net.param_dim


if __name__ == '__main__':

    # Create a function to learn.
    def func(x):
        return np.atleast_2d(3 * x[:,0]**1 + 2 * x[:,1]**2).T
        return np.atleast_2d(np.sin(x[:,0])**2 + np.cos(2*np.pi*x[:,1]/0.6)).T

    # Sample data.
    M = 5000
    X_train = np.random.uniform(2*np.pi, size=(M,2))
    y_train = func(X_train)

    # Create our neural network.
    x = tf.placeholder(dtype, [None, 2])
    layers = [(64, tf.nn.relu), (1, None)]
    layers = [(32, tf.nn.relu), (1, None)]
    net = SimpleNet(x, layers)

    sess = tf.Session()

    # Initialize.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train the network.
    net.fit(sess, X_train, y_train, nb_itr=50000, batch_size=1000)

    plt.figure(100)
    plt.clf()
    plt.plot(net.predict(sess, X_train), y_train, 'o')
    plt.show()
    plt.title('Should be a straight line.')


