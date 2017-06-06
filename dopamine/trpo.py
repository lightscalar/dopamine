import tensorflow as tf
import numpy as np
import pylab as plt
from dopamine.agent import Agent
from dopamine.lineworld import *
from dopamine.net import SimpleNet
from dopamine.utils import *
from dopamine.values import *
from ipdb import set_trace as debug
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
import logging
logging.basicConfig(level=logging.INFO)


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
        backend.set_session(self.session)
        self.info = logging.info

        # Here is the environment we'll be simulating, and pdf.
        self.env = env
        self.pdf = pdf
        self.pf = None

        # Set up configuration (if None is passed, fill empty dict).
        cfg = cfg if cfg else {}

        # Set defaults for TRPO optimizer.
        cfg.setdefault('episodes_per_step', 20)
        cfg.setdefault('gamma', 0.995)
        cfg.setdefault('lambda', 0.96)
        cfg.setdefault('cg_damping', 0.1)
        cfg.setdefault('epsilon', 0.01)
        cfg.setdefault('weight_file', 'weights_policy.dat')
        cfg.setdefault('iterations_per_save', 10)
        cfg.setdefault('load_weights', False)
        cfg.setdefault('make_plots', True)
        self.cfg = cfg

        # And here is the policy that we're trying to optimize.
        self.policy = policy

        # Define variables of interest.
        self.network_params = network_params = policy.trainable_weights

        # Action vector is the [mean, std] of the Gaussian action density.
        self.state_vectors = state_vectors = policy.input
        self.action_vectors = action_vectors = policy.output
        self.action_vectors_old = action_vectors_old = \
                self.pdf.parameter_vector
        self.actions_taken = actions_taken = self.pdf.sample(action_vectors)
        self.advantages = advantages = tf.placeholder(dtype, [None])

        # Compute the surrogate loss function.
        self.logp = logp = self.pdf.loglikelihood(actions_taken, action_vectors)
        self.logp_old = logp_old = self.pdf.loglikelihood(actions_taken,\
                action_vectors_old)
        self.loss = -tf.reduce_mean(tf.exp(logp - logp_old) * advantages)
        self.policy_gradient = flat_gradient(self.loss, network_params)

        # Compute expected KL divergence (but exclude first argument from
        # gradient computations).
        action_vectors_fixed = tf.stop_gradient(action_vectors)
        kl_first_fixed = pdf.kl(action_vectors_fixed, action_vectors)
        self.expected_kl = expected_kl = tf.reduce_mean(kl_first_fixed)

        self.kl_oldnew = tf.reduce_mean(pdf.kl(action_vectors,\
                action_vectors_old))

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

        # Estimate vvalue function using another neural network.
        self.vf = ValueFunction(env.D, self.session)

        # Initialize all of our variables.
        init = tf.global_variables_initializer()
        self.session.run(init)

        # Load previous training parameters.
        if cfg['load_weights']:
            self.policy.load_weights(self.cfg['weight_file'])
            weights = self.policy.get_weights()
            theta = np.concatenate([np.reshape(x, np.prod(x.shape)) \
                    for x in weights])
            self.set_from_flat(theta)

    def save_weights(self):
        '''Save weights for policy and value function.'''
        self.info('> Saving weights.')
        self.policy.save_weights(self.cfg['weight_file'])
        self.vf.save_weights()


    def compute_advantages(self, paths):
        '''Computes advantages, give delta estimates.'''
        gamma = self.cfg['gamma']
        gl = (self.cfg['gamma'] * self.cfg['lambda'])
        for path in paths:
            path["returns"] = discount(path["rewards"], gamma)
            b = path['baseline']
            b1 = np.append(b, 0)
            deltas = path["rewards"].flatten() + gamma*b1[1:] - b1[:-1]
            path['advantages'] = discount(deltas, gl)

    def simulate(self):
        '''Simulate the environment with respect to the given policy.'''

        # Initialize these things!
        paths = []

        self.info('> Simulating episodes.')
        for itr in range(self.cfg['episodes_per_step']):

            states, actions, action_vectors, rewards = [], [], [], []

            # Initialize the current state of the system.
            state = self.env.reset()
            done = False
            position = []

            while (not done): # keep simulating until episode terminates.

                # Estimate the action based on current policy!
                x = 1.0*state[0][0]
                position.append(x)

                states.append(state)
                action_vector, action = self.act(state)
                action_vector = action_vector.tolist()
                action = action.tolist()

                # Now take a step!
                state, reward, done = self.env.step(action)

                # Filter the rewards.
                # reward = reward_filter(reward)
                rewards.append(reward)

                # Store these things for later.
                actions.append(action[0])
                action_vectors.append(action_vector[0])

            # Assemble all useful path information.
            path = {'state_vectors': np.vstack(states),
                    'rewards': np.vstack(rewards),
                    'action_vectors': np.vstack(action_vectors),
                    'actions': np.vstack(actions),
                    'position': np.vstack(position)}
            paths.append(path)
        return paths

    def act(self, state):
        '''Take an action, given an observed state.'''
        return self.session.run([self.action_vectors, self.actions_taken], \
                feed_dict={self.state_vectors: state})

    def learn(self):
        '''Learn to control an agent in an environment.'''

        for _itr in range(9000):

            # 1. Simulate paths using current policy --------------------------
            paths = self.simulate()
            self.info(paths[0]['state_vectors'][0])

            # 2. Generalized Advantage Estimation -----------------------------
            self.vf.predict(paths)
            self.compute_advantages(paths)

            # 2b. Assemble necessary data -------------------------------------
            state_vectors = np.concatenate([path['state_vectors'] for \
                    path in paths])
            advantages = np.concatenate([path['advantages'] for path in paths])
            actions_taken = np.concatenate([path['actions'] for path in paths])
            action_vectors = np.concatenate([path['action_vectors'] for \
                    path in paths])
            returns = np.concatenate([path['returns'] for path in paths])

            # 3. TRPO update of policy ----------------------------------------
            theta_previous = 1*self.get_flat()

            # Normalize the advantages.
            advantages -= advantages.mean()
            advantages /= (advantages.std() + 1e-8)

            # Load up dict for the big update.
            feed = {self.action_vectors_old: action_vectors,
                    self.actions_taken: actions_taken,
                    self.state_vectors: state_vectors,
                    self.advantages: advantages.flatten()}

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.session.run(self.fvp, feed) + \
                        self.cfg['cg_damping'] * p

            # Compute the current gradient (the g in Fx = g).
            g = self.session.run(self.policy_gradient, feed_dict=feed)

            # Use conjugate gradient to find natural gradient direction.
            natural_direction = conjugate_gradient(fisher_vector_product, -g)

            # Determine the maximum allowable step size.
            quadratic_term = 0.5 * natural_direction.dot(\
                    fisher_vector_product(natural_direction))
            lagrange_multiplier = np.sqrt(quadratic_term/self.cfg['epsilon'])
            full_step = natural_direction/lagrange_multiplier
            expected_improvement_rate = -g.dot(natural_direction)/\
                    lagrange_multiplier

            # Now line search to update theta.
            def surrogate_loss(theta):
                self.set_from_flat(theta)
                return self.session.run(self.loss, feed_dict=feed)

            # Use a linesearch to take largest useful step.
            success, theta_new = linesearch(surrogate_loss, theta_previous,\
                    full_step, expected_improvement_rate)

            self.info('Iteration: {:d}'.format(_itr))
            self.info('> Fitting the value function.')
            self.vf.fit(paths)

            # Compute the new KL divergence.
            kl = self.session.run(self.kl_oldnew, feed_dict=feed)

            if kl > 2*self.cfg['epsilon']: # No big steps!
                self.set_from_flat(theta_previous) # assigns old theta.
            else:
                self.info ('> Updating theta.')
                self.set_from_flat(theta_new)

            mean_rewards = np.array(
                [path["rewards"].mean() for path in paths])
            self.info('> KL divergence: {:.3f}'.format(kl))
            self.info('> Mean Reward: {:.4f}'.format(mean_rewards.mean()))
            self.info('> Surrogate Loss: {:.4}'.format(\
                    surrogate_loss(theta_new)))

            if np.mod(_itr, self.cfg['iterations_per_save']) == 0:
                self.info('Saving policy and value function weights.')
                self.save_weights()

            if self.cfg['make_plots']:
                if np.mod(_itr,1) == 0:
                    plt.figure(100);
                    plt.clf()
                    for path in paths:
                        plt.plot(path['state_vectors'][:,0], 'red')
                        plt.ylim([-20, 20])
                    plt.plot(plt.xlim(), [5,5])
                    plt.show()
                    plt.grid(True)
                    plt.title('Iteration {:d}'.format(_itr))
                    plt.pause(0.05)
        return advantages


if __name__ == '__main__':

    # Create a lineworld instance.
    env = LineWorld()
    nb_actions = env.nb_actions

    # Create a placeholder for inputs to the policy.
    state_vector = tf.placeholder('float32', [None, env.D])

    # Create the policy model.
    layers = []
    layers.append({'input_dim': env.D, 'units': 16})
    layers.append({'units': 1, 'activation': 'tanh'})
    policy = create_mlp(layers)
    pdf = DiagGaussian(nb_actions)

    # So we have our three necessary objects! Let's create a TRPO agent.
    agent = TRPOAgent(env, policy, pdf)
    paths = agent.learn()


