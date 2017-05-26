'''Implements a variety of stochastic policies for policy gradients.'''
import numpy as np
import tensorflow as tf
from ipdb import set_trace as debug


# Specify the data type we're using.
dtype = tf.float32


class PDF(object):
    '''Base class for various probability densities/distributions.'''

    def sampled_var(self):
        raise NotImplementedError

    def param_vector(self):
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


class DiagGaussian(PDF):
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

    @property
    def parameter_vector(self):
        '''Returns parameter vector placeholder.'''
        return tf.placeholder(dtype, [None, 2*self.D])

    @property
    def sampled_variable(self):
        '''Returns placeholder for sampled variable.'''
        return tf.placeholder(dtype, [None, self.D])

    def loglikelihood(self, x, parameter_vector):
        '''Log likelihood of observations x given parameter vector. In all that
           follows, the first D components of the parameter vector correspond
           to the mean vector of the density; the second D elements are the
           standard deviation of the corresponding mean elements.
        '''
        mu = parameter_vector[:, :self.D]
        std = parameter_vector[:, self.D:]

        return -0.5 * tf.reduce_sum(tf.square( (x-mu)/std ), axis=1)\
                -0.5 * tf.log(2*np.pi) * self.D\
                -tf.reduce_sum(tf.log(std), axis=1)

    def likelihood(self, x, parameter_vector):
        '''Likelihood of observation given parameter vector.'''
        return tf.exp(self.loglikelihood(x, parameter_vector))

    def kl(self, param_vector_a, param_vector_b):
        '''Compute Kullback-Leibler divergence between two densities.'''
        mu_a = param_vector_a[:, :self.D]
        mu_b = param_vector_b[:, :self.D]
        std_a = param_vector_a[:, self.D:]
        std_b = param_vector_b[:, self.D:]
        std_a2 = tf.square(std_a)
        mean_diff2 = tf.square(mu_a - mu_b)
        denom = 2 * tf.square(std_b)
        term_1 = tf.reduce_sum(tf.log(std_b/std_a), axis=1)
        term_2 = tf.reduce_sum( tf.divide(std_a2 + mean_diff2, denom), axis=1)
        term_3 = -0.5 * self.D
        return term_1 + term_2 + term_3

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


def kl_numpy(a,b):
    '''Numpy implementation of KL for sanity check.'''
    a = np.array(a)
    b = np.array(b)
    d = int(a.shape[1]/2)
    mu_a = a[:,:d]
    std_a = a[:,d:]
    mu_b = b[:,:d]
    std_b = b[:,d:]
    std_a2 = np.square(std_a)
    mean_diff2 = np.square(mu_a - mu_b)
    denom = 2 * np.square(std_b)
    term_1 = np.sum(np.log(std_b/std_a), axis=1)
    term_2 = np.sum( np.divide(std_a2 + mean_diff2, denom), axis=1)
    term_3 = -0.5 * d
    return term_1 + term_2 + term_3


def loglike_np(x, a):
    '''Numpy implementation of KL for sanity check.'''
    a = np.array(a)
    d = int(a.shape[1]/2)
    mu = a[:,:d]
    std = a[:,d:]
    term1 = -0.5 * ((x - mu)**2/std**2).sum(axis=1)
    term2 = -0.5 * np.log(2*np.pi) * d
    term3 = -np.log(std).sum(axis=1)
    return term1 + term2 + term3


if __name__=='__main__':

    # Create a new stochastic policy.
    gauss = DiagGaussian(1)
    dtype = 'float32'

    # Sample from this density!
    obs = tf.placeholder(dtype, [None, 1])
    vector = tf.placeholder(dtype, [None, 2])
    vector_b = tf.placeholder(dtype, [None, 2])

    x = [[1], [1]]
    param = [[15,0.2], [1.0, 0.6]]
    param2 = [[11, 0.05,], [2.4, 0.4]]

    l1 = gauss.loglikelihood(obs, vector)
    kl = gauss.kl(vector, vector_b)

    with tf.Session() as sess:
        ll_tf = sess.run(l1, feed_dict={obs: x, vector: param})
        ll_np = loglike_np(x, param)
        kl_tf = sess.run(kl, feed_dict={vector: param, vector_b: param2})
        kl_np = kl_numpy(param, param2)


    # These should be more or less the same.
    print(ll_tf)
    print(ll_np)

    print(kl_tf)
    print(kl_np)
