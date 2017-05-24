import tensorflow as tf
import numpy as np
from dopamine.agent import Agent
from dopamine.lineworld import *
from dopamine.net import SimpleNet
from dopamine.utils import *
from ipdb import set_trace as debug


class TRPO(object):
    '''Trust Region Policy Optimizer.'''

    def __init__(self, env, policy, config=None):
        '''Creates a TRPO instance.
        INPUTS
            env - object
                An environment simulation object.
            policy - object
                A stochastic policy object of some sort.
        '''

        # Create a tensorflow session.
        self.sess = tf.Session()

        # Here is the environment we'll be simulating.
        self.env = env

        # Set up configuration.
        if (not config):
            config = {}

        # Set defaults for TRPO optimizer.
        config.setdefault('episodes_per_step', 100)
        config.setdefault('gamma', 0.99)
        self.config = config

        # Current paths are empty.
        self.paths = []

        # And here is the policy that we're trying to optimize.
        self.policy = policy

        # Define variables of interest.
        probability = policy.prob
        network_params = policy.params

        # Action vector is the [mean, std] of the Gaussian action density.
        action_vector = policy.output
        action_vector_old = probability.parameter_vector

        # Observation is a placeholder for the states that we supply to policy.
        observations = policy.input

        # The action_taken placeholder holds the actual actions sampled from
        # the policy. It takes its shape from the probability density.
        action_taken = probability.sampled_var

        # The number of observations.
        N = observations.shape[0]

        # Compute expected KL divergence (but exclude first argument from
        # gradient).
        action_vector_fixed = tf.stop_gradient(action_vector)
        kl_first_fixed = probability.kl(action_vector_fixed, action_vector)
        expected_kl = tf.reduce_mean(kl_first_fixed)

        # Now we compute the gradients of the expected KL divergence.
        grads = tf.gradients(expected_kl, network_params)

        # Placeholder for tangent vector in the network's parameter space.
        self.flat_tangent = tf.placeholder(dtype, [None])

        # Set up the computation of the Fisher Vector product!
        tangents = make_tangents(self.flat_tangent, network_params)

        # The gradient/vector product.
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]

        # Take gradient of GVP and flatten the result to obtain the Fisher-
        # vector product.
        self.fvp = flat_gradient(gvp, network_params)

        # Create objects to convert flat to expanded parameters, & vice/versa.
        self.get_flat = GetFlat(self.sess, network_params)
        self.set_from_flat = SetFromFlat(self.sess, network_params)

        # Use another SimpleNet to model our value function.
        tf.paths = tf.placeholder(dtype, [None, env.D])
        vf_layers = [(64, tf.nn.relu), (64, tf.nn.relu), (1, None)]
        self.value_function = SimpleNet(paths, vf_layers)


    def rollout(self):
        '''Simulate the environment with respect to the given policy.'''

        paths = []
        for _k in range(self.config['episodes_per_step']):

            pass


    def learn(self):
        '''Learn to control an agent in an environmentl.'''
        for itr in range(1):

            paths = self.rollout()


if __name__ == '__main__':

    # Create a lineworld instance.
    env = LineWorld()
    nb_actions = env.nb_actions

    # Create a placeholder for inputs to the model.
    x_ = tf.placeholder('float32', [None, env.D])

    # Create the policy model.
    layer_config = [(64, tf.nn.relu), (64, tf.nn.relu), (2*nb_actions, None)]
    net = SimpleNet(x_, layer_config)
    policy = StochasticPolicy(net, DiagGaussian(nb_actions))
    tr = TRPO(env, policy)

