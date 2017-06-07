import numpy as np
import pylab as plt
from dopamine.stochastics import *
from ipdb import set_trace as debug


class DiscWorld(object):
    '''This provides a simulation environment for testing reinforcement
       learning algorithms. It provides a one-dimensional space with a one-
       dimensional action space. Action is a Gaussian random variable
       corresponding to the magnitude and direction of thrust.
    '''

    def __init__(self):
        self.inpt = 0
        self.x = 10 * (0.5 - np.random.rand())
        self.y = 10 * (0.5 - np.random.rand())
        self.vx = 0
        self.vy = 0
        self.ax = 0.0
        self.ay = 0.0
        self.t = 0
        self.dt = 0.1
        self.max_time = 30
        self.target_x = 5
        self.target_y = 5
        self.total_steps = 0

    def step(self, action):

        # We have a two-dimensional continuous action space now.
        self.ax = action[0][0]
        self.ay = action[0][1]

        # Update position.
        self.t += self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.vx += self.ax * self.dt
        self.vy += self.ay * self.dt
        self.reward = -self._distance_to_target
        self.total_steps += 1

        return self.state, self.reward, self.done

    @property
    def _distance_to_target(self):
        x, y = self.x, self.y
        tx, ty = self.target_x, self.target_y
        return np.sqrt((x - tx)**2 + (y - ty)**2)

    @property
    def nb_actions(self):
        '''Continuous actions corresponding to horizontal/vertical thrust.'''
        return 2

    @property
    def done(self):
        '''Are we done with the current episode?'''
        return (self.t>self.max_time)

    @property
    def state(self):
        '''The currently observed state of the system.'''
        return np.atleast_2d([self.x, self.y, self.vx, self.vy, self.ax, \
                self.ay])

    @property
    def D(self):
        '''Returns dimension of the state space.'''
        return self.state.shape[1]

    def render(self, paths, fignum=100):
        '''Illustrate trajectories.'''
        plt.figure(fignum);
        plt.clf()
        for path in paths:
            # X coordinate.
            plt.subplot(211)
            plt.plot(path['state_vectors'][:,0], 'red')
            plt.ylim([-20, 20])
            plt.xlim([0, 300])
            plt.plot(plt.xlim(), [5,5])
            plt.grid(True)
            plt.ylabel('x position')

            # Y coordinate.
            plt.subplot(212)
            plt.plot(path['state_vectors'][:,0], 'red')
            plt.xlim([0, 300])
            plt.ylim([-20, 20])
            plt.plot(plt.xlim(), [5,5])
            plt.grid(True)
            plt.ylabel('y position')
        plt.show()
        plt.pause(0.05)

    @property
    def steps_per_episode(self):
        return int(self.max_time/self.dt)

    def reset(self):
        '''Reset state vector, time, velocity, etc.'''
        self.inpt = 0
        self.x, self.vx, self.t = 10*np.random.rand(), 2*np.random.randn(), 0
        self.y, self.vy, self.t = 10*np.random.rand(), 2*np.random.randn(), 0
        return self.state

    def simulate(self, agent):
        '''Simulate the environment using the provided agent to control all the
        things. The agent must have a take_action method that spits out a valid
        action for the environment.'''
        x = self.reset()
        done = False
        xs = []
        rs = []
        while not done:
            a = agent.take_action(x)
            x, reward, done = self.step(a)
            xs.append(x)
            rs.append(reward)
        return np.vstack(xs), np.vstack(rs)


if __name__ == '__main__':

    # Make an environment.
    env = LineWorld()
    pdf = DiagGaussian(2)

    x = []
    rewards = []
    done = False
    param = pdf.parameter_vector
    sample = pdf.sample(param)
    with tf.Session() as sess:
        while not done:

            # Simple random gaussian policy.
            action = sess.run(sample, feed_dict={param: [[0,0]]})
            print(action)

            # Step the simulation.
            state, reward, done = env.step(action)
            x.append([state[0][0], state[0][1]])
            reward.append(reward)

    # Plot this guy.
    x = np.vstack(x)
    from mathtools.vanity import *
    setup_plotting()
    import seaborn
    plot(x[:,0], x[:,1])
