import tensorflow as tf
import numpy as np
from dopamine.agent import Agent
from dopamine.lineworld import *
from dopamine.net import SimpleNet
from dopamine.utils import *
from dopamine.values import *
from ipdb import set_trace as debug
import pylab as plt
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout


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
        self.pf = None

        # Set up configuration (if None is passed, fill empty dict).
        cfg = cfg if cfg else {}

        # Set defaults for TRPO optimizer.
        cfg.setdefault('episodes_per_step', 20)
        cfg.setdefault('gamma', 0.99)
        cfg.setdefault('lambda', 0.97)
        cfg.setdefault('cg_damping', 0.2)
        cfg.setdefault('epsilon', 0.01)
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
        self.loss = -tf.reduce_mean(tf.exp(logp)/tf.exp(logp_old) * advantages)
        self.policy_gradient = flat_gradient(self.loss, network_params)
        # surr = (-1.0 / N) * T.exp(logp_n - oldlogp_n).dot(adv_n)

        # The number of observations.
        N = state_vectors.shape[0]

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

        # Use another SimpleNet to model our value function.
        # self.states = tf.placeholder(dtype, [None, env.D])
        # layers = []
        # layers.append({'input_dim': self.env.D, 'units': 128})
        # layers.append({'units': 1, 'activation': 'linear'})
        # self.vf_model = create_mlp(layers, {'loss': 'mse'})
        self.vf = ValueFunction(env.D, self.session)

        # Initialize all of our variables.
        init = tf.global_variables_initializer()
        self.session.run(init)

        # Load previous training parameters.
        # self.policy.load_weights('excellent_run')
        # weights = self.policy.get_weights()
        # theta = np.concatenate([np.reshape(x, np.prod(x.shape)) for x in weights])
        # self.set_from_flat(theta)

    def compute_advantages(self, path):
        '''Computes advantages, give delta estimates.'''
        gamma = self.cfg['gamma']
        gl = (self.cfg['gamma'] * self.cfg['lambda'])
        path["returns"] = discount(path["rewards"], gamma)
        b = path['baseline'] # = vf
        b1 = np.append(b, 0)
        deltas = path["rewards"].flatten() + gamma*b1[1:] - b1[:-1]
        path['advantages'] = discount(deltas, gl)
        # path['advantages'] = path['returns'] # q-function
        return path

    def compute_values(self, path):
        '''Compute the values along a trajectory.'''
        gamma = self.cfg['gamma']
        path['values'] = discount(path['rewards'], gamma)
        return path

    def get_baseline(self, path):
        '''Return the predicted returns.'''
        return self.session.run(self.vf, feed_dict={self.states: path['state_vectors']})

    def get_advantages(self, path):
        path['baseline'] = self.get_baseline(path)
        path['values'] = discounted_sum(path['rewards'], self.cfg['gamma'])
        path['advantages'] = path['values'] - path['baseline']
        return path

    def simulate(self):
        '''Simulate the environment with respect to the given policy.'''

        # Initialize these things!
        paths = []

        print('> Simulating episodes.')
        for itr in range(self.cfg['episodes_per_step']):

            states, actions, action_vectors, rewards = [], [], [], []

            # Initialize the current state of the system.
            state = self.env.reset()
            done = False
            position = []

            state_filter = ZFilter(state.shape, demean=False, destd=False, clip=None)
            reward_filter = ZFilter((), demean=False, destd=False, clip=5)

            while (not done): # keep simulating until episode terminates.

                # Estimate the action based on current policy!
                # obs = np.hstack( (state, prev_state) )
                x = 1.0*state[0][0]
                position.append(x)

                # state = state_filter(state)
                # state = state_filter(state)
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

    def update_value_fn(self, path, returns, epochs=500):
        '''Train our neural network on current values.'''
        # returns = discount(path['rewards'], self.cfg['gamma'])
        # returns = self.estimate_value(
        states = path['state_vectors']
        self.vf_model.fit(states, returns, epochs=epochs, batch_size=10000)

    def estimate_value(self, paths):
        '''Time dependent estimate value.'''
        values = []
        for path in paths:
            values.append(discount(path['rewards'], 0.99).T)
        return np.vstack(values).mean(0)

    def learn(self):
        '''Learn to control an agent in an environment.'''

        for _itr in range(9000):

            # 1. Simulate paths using current policy --------------------------
            paths = self.simulate()
            print(paths[0]['state_vectors'][0])

            # 2. Generalized Advantage Estimation -----------------------------
            # vf = self.estimate_value(paths)
            self.vf.predict(paths)
            for path in paths:
                # path = self.get_advantages(path)
                path = self.compute_advantages(path)
                # self.update_value_fn(path, vf)

            # 2b. Assemble necessary data -------------------------------------
            state_vectors = np.concatenate([path['state_vectors'] for \
                    path in paths])
            advantages = np.concatenate([path['advantages'] for path in paths])
            actions_taken = np.concatenate([path['actions'] for path in paths])
            action_vectors = np.concatenate([path['action_vectors'] for \
                    path in paths])
            returns = np.concatenate([path['returns'] for path in paths])
            # returns -= returns.mean()
            # returns /= returns.std()

            # 3. TRPO update of policy ----------------------------------------
            theta_previous = 1*self.get_flat()

            # Normalize the advantages.
            # advantages -= advantages.mean()
            # advantages /= (advantages.std() + 1e-8)

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
                self.policy.set_weights(self.session.run(self.network_params))
                return self.session.run(self.loss, feed_dict=feed)

            # Use a linesearch to take largest useful step.
            success, theta_new = linesearch(surrogate_loss, theta_previous, full_step,\
                    expected_improvement_rate)

            print('Iteration: {:d}'.format(_itr))
            print('> Fitting the value function.')
            self.vf.fit(paths)

            # Compute the new KL divergence.
            kl = self.session.run(self.kl_oldnew, feed_dict=feed)
            print(np.linalg.norm(theta_new - theta_previous))

            if kl > 5*self.cfg['epsilon']:
                self.set_from_flat(theta_previous) # assigns old theta.
            else:
                print ('> Updating theta.')
                self.set_from_flat(theta_new)
                self.policy.set_weights(self.session.run(self.network_params))

            mean_rewards = np.array(
                [path["rewards"].mean() for path in paths])
            print('> KL divergence: {:.3f}'.format(kl))
            print('> Mean Reward: {:.4f}'.format(mean_rewards.mean()))
            print('> Surrogate Loss: {:.4}'.format(surrogate_loss(theta_new)))
            # return state_vectors, values
            if np.mod(_itr,1) == 0:
                plt.figure(100);
                plt.clf()
                plt.subplot(211)
                for path in paths:
                    plt.plot(path['position'], 'grey')
                    plt.ylim([-10,10])
                plt.plot(plt.xlim(), [5,5])
                plt.show()

                plt.subplot(212)
                for path in paths:
                    plt.plot(path['state_vectors'][:,2], 'grey')
                    plt.ylim([-1,1])
                plt.show()
                plt.pause(0.05)

        return advantages


if __name__ == '__main__':

    # Create a lineworld instance.
    env = LineWorld()
    nb_actions = env.nb_actions

    # Create a placeholder for inputs to the policy.
    state_vector = tf.placeholder('float32', [None, env.D])

    # Create the policy model.
    # layer_config = [(32, tf.nn.relu), (32, tf.nn.relu), (nb_actions, None)]
    # policy = SimpleNet(state_vector, layer_config)
    layers = []
    layers.append({'input_dim': env.D, 'units': 16})
    # layers.append({'units': 32})
    layers.append({'units': 1, 'activation': 'tanh'})
    policy = create_mlp(layers)

    pdf = DiagGaussian(nb_actions)

    # So we have our three necessary objects! Let's create a TRPO agent.
    agent = TRPOAgent(env, policy, pdf)
    paths = agent.learn()

    vf = ValueFunction(3, agent.session)
    vf.fit(paths)

