'''Implements a variety of stochastic policies for policy gradients.'''
import numpy as np
import tensorflow as tf


# Specify the data type we're using.
dtype = tf.float32


class PDF(object):
    '''Base class for various probability densities/distributions.'''

    def sampled_variable(self):
        '''Returns a placeholder that like the sampled random variable.'''
        raise NotImplementedError

    def parameter_vector(self):
        '''Returns placeholder like parameter vector for the distribution.'''
        raise NotImplementedError

    def likelihood(self, obs, param):
        '''Returns the likelihood of an observation, x, with respect to a
           distribution characterized by parameter vector param.
        '''
        raise NotImplementedError

    def loglikelihood(self, obs, prob):
        '''Returns the log likelihood of an observation a w.r.t. density
           characterized by probability prob.
        '''
        raise NotImplementedError

    def kl(self, param0, param1):
        '''The Kullback-Leibler divergence between two distributions.'''
        raise NotImplementedError

    def entropy(self, param):
        '''Returns entropy of the distribution.'''
        raise NotImplementedError

    def maxprob(self, param):
        '''Values corresponding to the maximum probability.'''
        raise NotImplementedError


class DiagGaussian(PDF):
    '''Multivariate Gaussian policy characterized by a D-dimensional mean
       vector and a diagonal covariance matrix.
    '''

    def __init__(self, D, stddev=None):
        '''A Gaussian policy vector.
        INPUTS
            D - int
                The dimension of the multivariate Gaussian density.
            stddev - array_like
                The standard deviation of the multivariate Gaussian
                distribution (diagonal of coariance). Defaults to ones of the
                same size as the mean parameter vector.
        '''
        self.D = int(D)
        self.stddev = sttdev if stddev else np.ones(D)

    @property
    def parameter_vector(self):
        '''Returns parameter vector placeholder.'''
        return tf.placeholder(dtype, [None, self.D])

    @property
    def sampled_variable(self):
        '''Returns placeholder for sampled variable.'''
        return tf.placeholder(dtype, [None, self.D])

    def loglikelihood(self, x, parameter_vector):
        '''Log likelihood of observations x given mean parameter vector.'''
        mu = parameter_vector[:,:]
        std = self.stddev * tf.ones_like(mu)
        return -0.5 * tf.reduce_sum(tf.square( (x-mu)/std ), axis=1)\
                -0.5 * tf.log(2*np.pi) * self.D\
                -tf.reduce_sum(tf.log(std), axis=1)

    def likelihood(self, x, parameter_vector):
        '''Likelihood of observation given parameter vector.'''
        return tf.exp(self.loglikelihood(x, parameter_vector))

    def kl(self, param_vector_a, param_vector_b):
        '''Compute Kullback-Leibler divergence between two densities.'''
        mu_a = param_vector_a
        mu_b = param_vector_b
        # std_a = self.stddev
        std_a = self.stddev * tf.ones_like(mu_a)
        # std_b = self.stddev
        std_b = self.stddev * tf.ones_like(mu_b)
        std_a2 = tf.square(std_a)
        mean_diff2 = tf.square(mu_a - mu_b)
        denom = 2 * tf.square(std_b)
        term_1 = tf.reduce_sum(tf.log(std_b/std_a), axis=1)
        term_2 = tf.reduce_sum( tf.divide(std_a2 + mean_diff2, denom), axis=1)
        term_3 = -0.5 * self.D
        return term_1 + term_2 + term_3

    def sample(self, param_vector):
        '''Sample a vector from this multivariate density function!'''
        mu = param_vector
        std = self.stddev * tf.ones_like(param_vector)
        M = tf.shape(param_vector)[0]
        return mu + std * tf.random_normal((M,self.D))

    def maxprob(self, param_vector):
        '''Return values with maximum probability (here, just the mean
           vector).'''
        return param_vector
