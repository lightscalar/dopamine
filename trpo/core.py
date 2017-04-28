import tensorflow as tf
import numpy as np
from lineworld import *


class Agent(object):
    '''An agent has a brain (the policy) and can take actions.'''

    def __init__(self, policy):
        pass

class StochasticPolicy(object):

    def __init__()



class TRPO(object):
    '''Trust Region Policy Optimizer.'''

    def __init__(self, env, config=None):
        '''Creates a TRPO instance.
        INPUTS
            env - object
                An environment simulation object.
        '''

        # We have an environment!
        self.env = env

        # Set up configuration.
        if (not config):
            config = {}

        # Set defaults for optimizer.
        config.setdefault('episodes_per_step', 100)
        config.setdefault('gamma', 0.99)
        config.setdefault('nb_neurons', 100)
        self.config = config

        # Create our tensorflow model.
        self._create_model()


    def _create_model(self):
        '''Creates a Tensorflow model.'''
        
        # Determine relevant dimensions.
        D = self.env.D
        A = self.env.nb_actions
        H = self.config['nb_neurons']
        layer_1_shape = (D, H)
        bias_1_shape = (D,1)
        xavier_init = 2/np.sqrt(D) 

        # Start very simple. Two layer RELU network, sigmoid output.
        self.W1 = tf.Variable(tf.random_normal(layer_shape, stddev=xavier_init))
        self.b = tf.Variable(tf.random_normal(bias_shape, stddev=xavier_init))

        # Declare placeholders for states, actions, and rewards.
        self.s = tf.placeholder('float', [None, D])
        self.a = tf.placeholder('float', [None, A])
        self.r = tf.placeholder('float', [None, 1])

        # Here is the policy!
        self.policy = tf.matmul(W,x) + self.b


    def predict(self, x):
        pass
        

if __name__ == '__main__':

    # Create a lineworld instance.
    env = LineWorld()

    # Create an instance of the TRPO optimizer.
    tr = TRPO(env)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(tr.W))

