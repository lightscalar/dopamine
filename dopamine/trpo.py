import tensorflow as tf
import numpy as np
import pylab as plt
from dopamine.utils import *
from dopamine.values import *
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras import backend
import logging
from mathtools.utils import Vessel
logging.basicConfig(level=logging.INFO)
from ipdb import set_trace as debug


class TRPOAgent(object):
    '''Trust Region Policy Optimizer.'''

    def __init__(self, name, env, policy, pdf, cfg=None, load_model=False):
        '''Creates a TRPO instance.
        INPUTS
            name - str
                The name of the TRPO agent, which will be used when saving the
                weight data for policy and value function.
            env - object
                An environment simulation object.
            policy - object
                A neural network that takes in state space vectors and returns
                an action parameter vector.
            pdf - object
                A probability density/distribution; must provide a .sample
                method that takes in an action parameter vector.
        '''
        self.name = name.replace(' ', '_')

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
        cfg.setdefault('episodes_per_step', 100)
        cfg.setdefault('gamma', 0.995)
        cfg.setdefault('lambda', 0.96)
        cfg.setdefault('cg_damping', 0.1)
        cfg.setdefault('epsilon', 0.01)
        cfg.setdefault('model_file', 'weights/policy_{:s}.h5'.\
                format(self.name))
        cfg.setdefault('filter_file', 'weights/filters_{:s}.dat'.\
                format(self.name))
        cfg.setdefault('iterations_per_save', 1)
        cfg.setdefault('load_weights', False)
        cfg.setdefault('make_plots', False)
        self.cfg = cfg

        # And here is the policy that we're trying to optimize.
        if load_model:
            self.load_model()
        else:
            self.policy = policy

        # Define variables of interest.
        self.network_params = network_params = policy.trainable_weights

        # Action vector is the [mean, std] of the Gaussian action density.
        self.state_vectors = state_vectors = policy.input
        # raw_vectors = tf.clip_by_value(policy.output, -15, 15)
        # self.action_vectors = action_vectors = tf.nn.softmax(raw_vectors)
        self.action_vectors = action_vectors = policy.output

        self.action_vectors_old = action_vectors_old = \
                self.pdf.parameter_vector
        self.actions_taken = actions_taken = self.pdf.sample(action_vectors)
        self.advantages = advantages = tf.placeholder(dtype, [None],\
                name='advantages')

        # Compute the surrogate loss function.
        self.logp = logp = self.pdf.loglikelihood(actions_taken,\
                action_vectors)
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
        self.gvp = gvp = [tf.reduce_sum(g*t) for (g,t) in zip(grads, tangents)]

        # Take gradient of GVP and flatten the result to obtain the Fisher-
        # vector product.
        self.fvp = flat_gradient(gvp, network_params)

        # Create objects to convert flat to expanded parameters, & vice/versa.
        self.params_to_theta = ParamsToTheta(self.session, network_params)
        self.theta_to_params = ThetaToParams(self.session, network_params)

        # Estimate value function using another neural network.
        if load_model:
            self.vf = ValueFunction(self.name, self.session, load_model=True)
        else:
            input_dim = env.observation_space.shape[0]
            self.vf = ValueFunction(self.name, self.session, input_dim)

        # Initialize all of our variables.
        init = tf.global_variables_initializer()
        self.session.run(init)

        self.obsfilt = ZFilter(self.env.observation_space.shape, clip=10)
        self.rewfilt = ZFilter((), demean=False, clip=5)

    def load_model(self):
        '''Load saved weights from file.'''
        self.policy = load_model(self.cfg['model_file'])
        # theta = np.concatenate([np.reshape(x, np.prod(x.shape)) \
        #         for x in weights])
        # self.theta_to_params(theta)

    def save_model(self, best_model=True):
        '''Save weights for policy and value function.'''
        if best_model:
            print('Saving Best Model...')
            self.policy.save(self.cfg['model_file'])
            v = Vessel(self.cfg['filter_file'])
            v.filt = self.obsfilt
            v.save()
            self.vf.save_model()

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
        # self.env.reset_target()
        for itr in range(self.cfg['episodes_per_step']):
            if np.mod(itr, 1000) == 0:
                self.info('Episode {:d} of {:d}'.format(itr,\
                        self.cfg['episodes_per_step']))
            states, actions, action_vectors, rewards = [], [], [], []

            # Initialize the current state of the system.
            state = self.env.reset()
            done = False
            position = []

            nb_steps = 0
            while (not done): # keep simulating until episode terminates.

                # Estimate the action based on current policy!
                state = self.obsfilt(state)
                states.append(state)
                action_vector, action = self.act(state)
                action = action[0]
                # action_vector = action_vector.tolist()
                # action = action.tolist()

                if np.random.rand() > 0.9999:
                    print(action)
                    print(action_vector)

                # Now take a step!
                state, reward, done, info = self.env.step(action)

                # Apply filters to the states, rewards.
                reward = self.rewfilt(reward)
                rewards.append(reward)

                # Store these things for later.
                actions.append(action)
                action_vectors.append(action_vector)
                nb_steps += 1
                # if nb_steps>200: break

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
                feed_dict={self.state_vectors: np.atleast_2d(state)})

    def learn(self):
        '''Learn to control an agent in an environment.'''
        self.best_reward = -np.inf

        reward_trajectory = []
        for _itr in range(50000):

            # 1. Simulate paths using current policy --------------------------
            paths = self.simulate()
            # self.info(paths[0]['state_vectors'][0])

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
            theta_previous = 1*self.params_to_theta()

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

            if np.isnan(fisher_vector_product(-g)).any(): debug()

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
                self.theta_to_params(theta)
                return self.session.run(self.loss, feed_dict=feed)

            # Use a linesearch to take largest useful step.
            success, theta_new = linesearch(surrogate_loss, theta_previous,\
                    full_step, expected_improvement_rate)


            # Compute the new KL divergence.
            kl = self.session.run(self.kl_oldnew, feed_dict=feed)
            if np.isnan(kl):
                self.theta_to_params(theta_previous) # assigns old theta.
                self.info('> NaN encountered. Skipping updates.')
                debug()
                continue

            if kl > 1.5*self.cfg['epsilon']: # No big steps!
                self.theta_to_params(theta_previous) # assigns old theta.
                updated = False
            else:
                self.info ('> Updating theta.')
                updated = True
                self.theta_to_params(theta_new)

            # Calculate some metrics.
            mean_rewards = np.array(
                [path["rewards"].sum() for path in paths])
            reward_trajectory.append(mean_rewards.mean())

            # Update the value function.
            self.info('> Fitting the value function.')
            self.vf.fit(paths)

            self.info('> Iteration: {:d}'.format(_itr))
            self.info('> KL divergence: {:.3f}'.format(kl))
            if updated:
                self.info('> Theta updated')
            else:
                self.info('> Theta not updated')
            self.info('> Mean Reward: {:.4f}'.format(mean_rewards.mean()))
            self.info('> Surrogate Loss: {:.4}'.format(\
                    surrogate_loss(theta_new)))

            # Save best model to disk.
            if mean_rewards.mean() > self.best_reward:
                self.save_model(best_model=True)
                self.best_reward = mean_rewards.mean()

            # Save model parameters to disk.
            if np.mod(_itr, self.cfg['iterations_per_save']) == 0:
                self.info('> Saving policy and value function weights.')
                self.save_model()

            if self.cfg['make_plots']:
                # Plot rewards over time.
                plt.figure(100)
                plt.clf()
                plt.plot(reward_trajectory)
                plt.show()
                plt.pause(0.05)

        return advantages
