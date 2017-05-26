import tensorflow as tf
import numpy as np
from dopamine.agent import Agent
from dopamine.lineworld import *
from dopamine.net import SimpleNet
from dopamine.utils import *
from ipdb import set_trace as debug


class TRPOAgent(object):
    '''Trust Region Policy Optimizer.'''

    def __init__(self, env, policy, pdf, cfg=None):
        '''Creates a TRPO instance.
        INPUTS
            env - object
                An environment simulation object.
            policy - object
                A neural network that takes in state space vectors and returns
                an action parameter vector.
            pdf - object
                A probability density/distribution; must provide a .sample
                method that takes in an action parameter vector.
        '''

        # Create a tensorflow session.
        self.session = tf.Session()

        # Here is the environment we'll be simulating, and pdf.
        self.env = env
        self.pdf = pdf

        # Set up configuration (if None is passed, fill empty dict).
        cfg = cfg if cfg else {}

        # Set defaults for TRPO optimizer.
        cfg.setdefault('episodes_per_step', 2)
        cfg.setdefault('gamma', 0.99)
        cfg.setdefault('lambda', 0.96)
        self.cfg = cfg

        # And here is the policy that we're trying to optimize.
        self.policy = policy

        # Define variables of interest.
        network_params = policy.params

        # Action vector is the [mean, std] of the Gaussian action density.
        self.state_vectors = state_vectors = policy.inputs
        action_vector = policy.outputs
        action_vector_old = self.pdf.parameter_vector
        self.action_taken = self.pdf.sample(action_vector)

        # The action_taken placeholder holds the actual actions sampled from
        # the policy. It takes its shape from the probability density.
        action_taken = pdf.sampled_var

        # The number of observations.
        N = state_vectors.shape[0]

        # Compute expected KL divergence (but exclude first argument from
        # gradient computations).
        action_vector_fixed = tf.stop_gradient(action_vector)
        kl_first_fixed = pdf.kl(action_vector_fixed, action_vector)
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
        self.get_flat = GetFlat(self.session, network_params)
        self.set_from_flat = SetFromFlat(self.session, network_params)

        # Use another SimpleNet to model our value function.
        self.states = tf.placeholder(dtype, [None, env.D])
        vf_layers = [(64, tf.nn.relu), (64, tf.nn.relu), (1, None)]
        self.value_function = SimpleNet(self.states, vf_layers)
        self.vf = self.value_function.outputs

        # Paths will contain (state, reward, action) tuples for trajectories.
        self.paths = []

        # Initialize all of our variables.
        init = tf.global_variables_initializer()
        self.session.run(init)

    def compute_deltas(self, path):
        '''Computes the deltas given observed states, rewards, and current
        value function.'''
        states = path['states']
        rewards = path['rewards']
        values = self.session.run(self.vf, feed_dict={self.states: states)
        return values

    def simulate(self):
        '''Simulate the environment with respect to the given policy.'''

        # Initialize these things!
        states, actions, rewards, paths = [], [], [], []

        for itr in range(self.cfg['episodes_per_step']):

            print('Simulating episode {:d}'.format(itr))

            # Initialize the current state of the system.
            state = self.env.reset()
            done = False
            k=0

            while (not done):

                print(k)
                # Estimate the action based on current policy!
                action = self.act(state).tolist()

                # Now take a step!
                state, reward, done = self.env.step(action)

                # Store these things for later.
                states.append(state)
                actions.append(action[0])
                rewards.append(reward)
                k+=1

            print('And we are done!')

            # Assemble the full path information.
            path = {'states': np.vstack(states),
                    'rewards': np.vstack(rewards),
                    'actions': np.vstack(actions)}

            paths.append(path)

        return paths

    def act(self, state):
        '''Take an action, given an observed state.'''
        return self.action_taken.eval(session=self.session,\
                feed_dict={self.state_vectors: state})

    def learn(self):
        '''Learn to control an agent in an environment.'''

        for _ in range(1):

            # 1. Simulate paths using current policy.
            paths = self.simulate()

            # 2. Generalized Advantage Estimation.

        return paths




if __name__ == '__main__':

    # Create a lineworld instance.
    env = LineWorld()
    nb_actions = env.nb_actions

    # Create a placeholder for inputs to the policy.
    state_vector = tf.placeholder('float32', [None, env.D])

    # Create the policy model.
    layer_config = [(64, tf.nn.relu), (64, tf.nn.relu), (2*nb_actions, None)]
    policy = SimpleNet(state_vector, layer_config)
    pdf = DiagGaussian(nb_actions)

    # So we have our three necessary objects! Let's create a TRPO agent.
    agent = TRPOAgent(env, policy, pdf)
    paths = agent.learn()

    # policy = StochasticPolicy(net, DiagGaussian(nb_actions))
    # tr = TRPO(env, policy)

    # Test our actions.
    # dg = DiagGaussian(1)
    # action_vector = policy.outputs
    # action_taken = dg.sample(action_vector)

    # with tf.Session() as sess:

    #     # Run this guy.
    #     init = tf.global_variables_initializer()
    #     sess.run(init)

    #     for k in range(1000):
    #         # print(sess.run(action_vector, feed_dict={state: env.reset()}))
    #         fd={state_vector: env.reset()}
    #         print(action_taken.eval(session=sess, feed_dict=fd))





