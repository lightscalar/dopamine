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


class GaussianPolicy(StochasticPolicy):
    '''Multivariate Gaussian policy characterized by a mean vector and a vector
       of log standard deviations.
    '''

    def __init__(self, action_vector):
        '''A Gaussian policy vector.
        INPUTS
            action_vector - array_like
                Array specifying the mean of the Gaussian process (first N 
                elements), as well as the the log standard deviation of the
                diagonal covariance matrix.
        '''

        # Vector must consist of mean followed by log standard deviation.
        assert np.mod(len(action_vector),2) == 0, \
                'Action vector must be of even length'
        n = int(len(action_vector)/2)
        self.mu = action_vector[0:n]
        log_std = action_vector[n:]
        self.std = np.exp(log_std)
        self.cov = np.diag(self.std)


    def __call__(self,N=1):
        '''Alias sample action for convenience.'''
        return self.sample_action(N)


    def sample_action(self, N=1):
        '''Sample from our density.'''
        return np.random.multivariate_normal(self.mu, self.cov, N)


if __name__=='__main__':

    # Create a new stochastic policy.
    gauss = GaussianPolicy([5,5], [1,1])

    # Sample from this density!
    print(gauss())
