import numpy as np
from dopamine.stochastics import *
from ipdb import set_trace as debug


class LineWorld(object):
    '''This provides a simulation environment for testing reinforcement
       learning algorithms. It provides a one-dimensional space with a one-
       dimensional action space. Action is a Gaussian random variable
       corresponding to the magnitude and direction of thrust.
    '''

    def __init__(self):
        self.x = 10 * np.random.rand()
        self.vx = 0
        self.ax_mag = 0.5
        self.ax = 0.5
        self.t = 0
        self.dt = 0.1
        self.max_time = 30
        self.target_x = 5
        self.total_steps = 0

    def step(self, action):

        # We have a one-dimensional continuous action space now.
        # The action here is simply the acceleration at this instant.
        self.ax = action[0][0]
        if np.abs(self.ax)>10:
            # We limit the maximum acceleration that can be requested.
            self.ax = np.sign(self.ax) * 10

        # Update position.
        self.t += self.dt
        self.x += self.vx * self.dt
        self.vx += self.ax * self.dt
        self.reward = -self._distance_to_target
        self.total_steps += 1

        return self.state, self.reward, self.done

    @property
    def _distance_to_target(self):
        return np.sqrt((self.x - self.target_x)**2)

    @property
    def nb_actions(self):
        '''Only a single scalar-valued action available here.'''
        return 1

    @property
    def done(self):
        '''Are we done with the current episode?'''
        return (self.t>self.max_time)

    @property
    def state(self):
        '''The currently observed state of the system.'''
        return np.atleast_2d([self.x, self.vx, self.ax, self.target_x])

    @property
    def D(self):
        '''Returns dimension of the state space.'''
        return self.state.shape[1]

    @property
    def steps_per_episode(self):
        return int(self.max_time/self.dt)

    def reset(self):
        '''Reset state vector, time, velocity, etc.'''
        self.x, self.vx, self.t = 10*np.random.rand(), 2*np.random.randn(), 0
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

    x = []
    done = False
    while not done:

        # Simple random gaussian policy.
        action = [0, np.log(1.25)]

        # Step the simulation.
        state, reward, done = env.step(action)
        x.append(state[0][0])

    # Plot this guy.
    from mathtools.vanity import *
    setup_plotting()
    import seaborn
    plot(x)



