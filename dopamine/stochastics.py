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
        pass


class ProbabilityType(object):
    '''Base class for various probability densities/distributions.'''

    def sampled_variable(self):
        raise NotImplementedError

    def prob_variable(self):
        raise NotImplementedError

    def likelihood(self, a, prob):
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        raise NotImplementedError

    def kl(self, prob0, prob1):
        '''The Kullback-Leibler divergence.'''
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError


class DiagGaussian(ProbabilityType):
    '''Multivariate Gaussian policy characterized by a mean vector and a
       diagonal covariance matrix.
    '''

    def __init__(self, D):
        '''A Gaussian policy vector.
        INPUTS
            D - int
                The dimension of the multivariate Gaussian density.
        '''
        self.D = D



    def sample_action(self, N=1):
        '''Sample from our density.'''
        return np.random.multivariate_normal(self.mu, self.cov, N)


if __name__=='__main__':

    # Create a new stochastic policy.
    gauss = GaussianPolicy([5,5], [1,1])

    # Sample from this density!
    print(gauss())
