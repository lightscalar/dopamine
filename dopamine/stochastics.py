'''Implements a variety of stochastic policies for policy gradients.'''
import numpy as np
import tensorflow as tf

# Specify the data type we're using.
dtype = tf.float32


class StochasticPolicy(object):
    '''Base class for stochastic policies.'''

    def __init__(self, network, probability_type):
        '''Configure the policy.'''
        self.net = network
        self.prob = probability_type

    def predict(self, state)):
        '''Predicts action distribution give state.'''
        return self.network(state)

    def act(self, state):
        '''Predicts an action vector and takes an action.'''
        pass


    @property
    def action_space(self):
        '''Returns an action space object.'''
        raise NotImplementedError


class ProbabilityType(object):
    '''Base class for various probability densities/distributions.'''

    def sampled_variable(self):
        raise NotImplementedError

    def prob_variable(self):
        raise NotImplementedError

    def likelihood(self, a, prob):
        '''Returns the likelihood of an observation, x, with respect to a
           distribution characterized by vector prob.
        '''
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        '''Returns the log likelihood of an observation a w.r.t. density
           characterized by probability prob.
        '''
        raise NotImplementedError

    def kl(self, prob0, prob1):
        '''The Kullback-Leibler divergence.'''
        raise NotImplementedError

    def entropy(self, prob):
        '''Returns entropy of the distribution.'''
        raise NotImplementedError

    def maxprob(self, prob):
        '''Values corresponding to the maximum probability.'''
        raise NotImplementedError


class DiagGaussian(ProbabilityType):
    '''Multivariate Gaussian policy characterized by a D-dimensional mean
       vector and a diagonal covariance matrix.
    '''

    def __init__(self, D):
        '''A Gaussian policy vector.
        INPUTS
            D - int
                The dimension of the multivariate Gaussian density.
        '''
        self.D = int(D)

    def loglikelihood(self, x, parameter_vector):
        '''Log likelihood of observations x given parameter vector. In all that
           follows, the first D components of the parameter vector correspond
           to the mean vector of the density; the second D elements are the
           standard deviation of the corresponding mean elements.
        '''
        mu = parameter_vector[:, :self.D]
        std = parameter_vector[:, self.D:]
        return - 0.5 * tf.reduce_sum(tf.square( (x-mu)/std ),1) - \
                 0.5 * tf.log(2*np.pi)*self.D - tf.reduce_sum(tf.log(std), 1)

    def likelihood(self, x, parameter_vector):
        '''Likelihood of observation given parameter vector.'''
        return tf.exp(self.loglikelihood(x, parameter_vector))

    def kl(self, param_vector_a, param_vector_b):
        '''Compute Kullback-Leibler divergence between two densities.'''
        mu_a = param_vector_a[:, :self.D]
        mu_b = param_vector_b[:, :self.D]
        std_a = param_vector_a[:, self.D:]
        std_b = param_vector_b[:, self.D:]
        return tf.reduce_sum(tf.log(std_b/std_a), axis=1) + \
                 tf.reduce_sum((tf.square(std_a) + tf.square(mu_a - mu_b) /\
                 (2.0 * tf.square(std_b))), axis=1) - (0.5 * self.D)

    def differential_entropy(self, param_vector):
        '''Compute the differential entropy of the density.'''
        std = param_vector[:, self.D:]
        return tf.reduce_sum(tf.log(std),axis=1) + \
                0.5 * np.log(2*np.pi*np.e)*self.D

    def sample(self, param_vector):
        '''Sample a vector from this multivariate density function!'''
        mu = param_vector[:, :self.D]
        std = param_vector[:, self.D:]
        M = tf.shape(param_vector)[0]
        return mu + std * tf.random_normal((M,self.D))

    def maxprob(self, param_vector):
        '''Return values with maximum probability (here, just the mean
           vector).'''
        return param_vector[:, :self.D]


if __name__=='__main__':

    # Create a new stochastic policy.
    gauss = DiagGaussian(1)
    dtype = 'float32'

    # Sample from this density!
    _x = tf.placeholder(dtype, [None, 1])
    _param = tf.placeholder(dtype, [None, 2])

    x = [[1], [1]]
    param = [[15,0.2], [1, 0.6]]

    with tf.Session() as sess:
        _loglik = gauss.likelihood(_x, _param)
        _dS = gauss.differential_entropy(_param)
        _samples = gauss.sample(_param)
        loglik = sess.run(_loglik, feed_dict={_x: x, _param: param})
        dS = sess.run(_dS, feed_dict={_param: param})
        samples = sess.run(_samples, feed_dict={_param: param})

