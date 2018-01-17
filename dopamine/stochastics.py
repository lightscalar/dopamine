'''Implements a variety of stochastic policies for policy gradients.'''
import numpy as np
import tensorflow as tf
from dopamine.utils import slice_tensor
from ipdb import set_trace as debug


# Specify the data type we're using.
dtype = tf.float32


def categorical_sample(prob_nk):
    '''Sample from categorical distribution. Each row specifies the class
       probabilities.'''
    prob_nk = np.asarray(prob_nk)
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    return np.argmax(csprob_nk > np.random.rand(N,1), axis=1)


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
        self.stddev = stddev if stddev else np.ones(D)

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
        std_a = self.stddev * tf.ones_like(mu_a)
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
        return tf.clip_by_value(mu + std * tf.random_normal((M,self.D)), -1, 1)

    def maxprob(self, param_vector):
        '''Return values with maximum probability (here, just the mean
           vector).'''
        return param_vector


class Categorical(PDF):
    '''A categorical distribution.'''

    def __init__(self, D):
        '''Specify number of classes/categories in distribution..'''
        self.D = D

    @property
    def sampled_variable(self):
        '''Returns placeholder for sampled variable.'''
        return tf.placeholder(dtype, [None, 1])

    @property
    def parameter_vector(self):
        '''Returns parameter vector placeholder.'''
        return tf.placeholder(dtype, [None, self.D])

    def likelihood(self, a, prob):
        '''Likelihood of observation a, given distribution prob.'''
        # Best way to index into these arrays?
        N = tf.shape(prob)[0]
        idx_0 = tf.range(N)
        idx_1 = tf.squeeze(a)
        return slice_tensor(prob, idx_0, idx_1)

    def loglikelihood(self, a, prob):
        '''Log-likehood of observation, given distribution.'''
        return tf.log(self.likelihood(a, prob))

    def kl(self, prob0, prob1):
        '''Kullback-Liebler divergence between two distributions.'''
        return tf.reduce_sum(prob0 * tf.log(prob0/(prob1 + 1e-6)), axis=1)
        # return (prob0 * T.log(prob0/prob1)).sum(axis=1)

    def entropy(self, prob0):
        '''Returns entropy of distribution.'''
        return -tf.reduce_sum(prob0 * tf.log(prob0), axis=1)
        # return - (prob0 * T.log(prob0)).sum(axis=1)

    def sample(self, prob):
        '''Sample from the categorical distribution.'''
        N = tf.shape(prob)[0]
        dist = tf.contrib.distributions.Categorical(prob)
        return tf.reshape(dist.sample(), shape=(N,1))

    def maxprob(self, prob):
        'Maximum probability across distribution.'''
        return tf.argmax(prob, axis=1)


if __name__ == '__main__':

    pdf = Categorical(4)
    params = pdf.parameter_vector()
    sample = pdf.sample(params)
    dist = np.atleast_2d([0.2, 0.4, 0.1, 0.3])
    # dist = np.atleast_2d([0.25, 0.25, 0.25, 0.25])
    dist_2 = np.atleast_2d([0.2, 0.35, 0.1, 0.3])
    p1 = pdf.parameter_vector()
    p2 = pdf.parameter_vector()
    kl = pdf.kl(p1, p2)
    entropy = pdf.entropy(p1)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        pdf = Categorical(4)
        out = sess.run(sample, feed_dict={params: dist})
        kl_div = sess.run(kl, feed_dict={p1: dist, p2: dist_2})
        ent = sess.run(entropy, feed_dict={p1:dist})
        pmax = sess.run(pdf.maxprob(p1), feed_dict={p1:dist})


