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
        cfg.setdefault('cg_damping', 0.1)
        self.cfg = cfg

        # And here is the policy that we're trying to optimize.
        self.policy = policy

        # Define variables of interest.
        network_params = policy.params

        # Action vector is the [mean, std] of the Gaussian action density.
        self.state_vectors = state_vectors = policy.inputs
        self.action_vectors = action_vectors = policy.outputs
        self.action_vectors_old = action_vectors_old = \
                self.pdf.parameter_vector
        self.actions_taken = actions_taken = self.pdf.sample(action_vectors)
        self.advantages = advantages = tf.placeholder(dtype, [None])

        # Compute the surrogate loss function.
        logp = self.pdf.loglikelihood(actions_taken, action_vectors)
        logp_old = self.pdf.loglikelihood(actions_taken, action_vectors_old)
        log_ratio = logp/logp_old
        self.loss = loss = -tf.reduce_mean( log_ratio * self.advantages )
        self.policy_gradient = flat_gradient(self.loss, network_params)

        # The number of observations.
        N = state_vectors.shape[0]

        # Compute expected KL divergence (but exclude first argument from
        # gradient computations).
        action_vectors_fixed = tf.stop_gradient(action_vectors)
        kl_first_fixed = pdf.kl(action_vectors_fixed, action_vectors)
        self.expected_kl = expected_kl = tf.reduce_mean(kl_first_fixed)

        # Now we compute the gradients of the expected KL divergence.
        self.grads = grads = tf.gradients(expected_kl, network_params,\
                name='gradients')

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
        states = path['state_vectors']
        rewards = path['rewards']
        # Estimate the state values using the neural network.
        values = self.session.run(self.vf, feed_dict={self.states: states})
        # Now compute the delta.
        delta_t = np.zeros_like(values)
        delta_t[:-1] = -values[:-1] + rewards[:-1] + self.cfg['gamma'] *\
                values[1:]
        delta_t[-1] = rewards[-1] # Last element value is just its reward
        return delta_t

    def compute_advantages(self, path):
        '''Computes advantages, give delta estimates.'''
        deltas = self.compute_deltas(path)
        gl = (self.cfg['gamma'] * self.cfg['lambda'])
        path['advantages'] = discounted_sum(deltas, gl)
        return path

    def simulate(self):
        '''Simulate the environment with respect to the given policy.'''

        # Initialize these things!
        paths = []

        for itr in range(self.cfg['episodes_per_step']):

            print('Simulating episode {:d}'.format(itr))
            states, actions, action_vectors, rewards = [], [], [], []

            # Initialize the current state of the system.
            state = self.env.reset()
            done = False

            while (not done): # keep simulating until episode terminates.

                # Estimate the action based on current policy!
                action_vector, action = self.act(state)
                action_vector = action_vector.tolist()
                action = action.tolist()

                # Now take a step!
                state, reward, done = self.env.step(action)

                # Store these things for later.
                states.append(state)
                actions.append(action[0])
                action_vectors.append(action_vector[0])
                rewards.append(reward)

            # Assemble all useful path information.
            path = {'state_vectors': np.vstack(states),
                    'rewards': np.vstack(rewards),
                    'action_vectors': np.vstack(action_vectors),
                    'actions': np.vstack(actions)}
            paths.append(path)
        return paths

    def act(self, state):
        '''Take an action, given an observed state.'''
        return self.session.run([self.action_vectors, self.actions_taken], \
                feed_dict={self.state_vectors: state})


    def learn(self):
        '''Learn to control an agent in an environment.'''

        for _ in range(1):

            # 1. Simulate paths using current policy --------------------------
            paths = self.simulate()

            # 2. Generalized Advantage Estimation -----------------------------
            for path in paths:
                path = self.compute_advantages(path)

            # 2b. Assemble necessary data -------------------------------------
            state_vectors = np.concatenate([path['state_vectors'] for \
                    path in paths])
            advantages = np.concatenate([path['advantages'] for path in paths])
            actions_taken = np.concatenate([path['actions'] for path in paths])
            action_vectors = np.concatenate([path['action_vectors'] for \
                    path in paths])

            # 3. TRPO update of policy ----------------------------------------

            # Normalize the advantages.
            advantages -= advantages.mean()
            advantages /= (advantages.std() + 1e-8)

            # Load up results for update.
            feed = {self.action_vectors_old: action_vectors,
                    self.actions_taken: actions_taken,
                    self.state_vectors: state_vectors,
                    self.advantages: advantages.flatten(),
                    self.action_vectors_old: action_vectors}

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.session.run(self.fvp, feed) + cfg['cg_damping'] * p

        return advantages




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






