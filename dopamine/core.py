import tensorflow as tf
import numpy as np
from agent import Agent
from lineworld import *
from ipdb import set_trace as debug


class TRPO(object):
    '''Trust Region Policy Optimizer.'''

    def __init__(self, env, policy, config=None):
        '''Creates a TRPO instance.
        INPUTS
            env - object
                An environment simulation object.
            policy - object
                The policy we're trying to optimize. Should be a tensorflow-
                based neural network object of some sort.
        '''

        # Here is the environment we'll be simulating.
        self.env = env

        # Set up configuration.
        if (not config):
            config = {}

        # Set defaults for TRPO optimizer.
        config.setdefault('episodes_per_step', 100)
        config.setdefault('gamma', 0.99)
        self.config = config

        # And here is the policy that we're trying to optimize.
        self.policy = policy

        # Current paths are empty.
        self.paths = []


    def rollout(self):
        '''Simulate the environment with respect to the given policy.'''

        for _k in range(self.config['episodes_per_step']):
            pass
            

    def learn(self):
        pass

        


if __name__ == '__main__':

    # Create a lineworld instance.
    env = LineWorld()

    # Create a placeholder for inputs to the model.
    x_ = tf.placeholder('float32', [None, env.D])

    # Create the policy model.
    model = [(64, tf.nn.relu), (64, tf.nn.relu), (1, tf.nn.sigmoid)]
    net = SimpleNet(model)

