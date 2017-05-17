'''Implements a variety of stochastic policies for policy gradients.'''
import numpy as np
import tensorflow as tf

# Specify the data type we're using.
dtype = tf.float32


class StochasticPolicy(object):
    '''Base class for all stochastic policies.'''

    def __init__(self):
        '''Configure the policy.'''
        pass


    def predict(self):
        '''Predicts action distribution give state.'''
        raise NotImplementedError


    def sample_action(self):
        '''Samples an action from the allowable action space.'''
        raise NotImplementedError


    @property
    def action_space(self):
        '''Returns an action space object.'''
        raise NotImplementedError


class BinaryPolicy(StochasticPolicy):
    '''Binary policy.''' 

    def __init__(self, nb_actions, config=None):
        '''Creates a neural network parameterized policy for non mutually
           exclusive binary actions.
        INPUTS
            nb_actions - int
                Number of binary actions available.
            config - dict
                Options. So many options.
        '''

        

