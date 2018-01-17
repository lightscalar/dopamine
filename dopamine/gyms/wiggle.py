import numpy as np


class Wiggle(object):

    def __init__(self):
        # Initialize the system.
        self.reset()

    def reset(self):
        # Reset the simulation.
        # Spring/piston configuration.
        self.k = 1 # N/meter
        self.l0 = 1
        self.p = 0 # piston position

        # Mass parameters.
        self.mass = 10 * 1e-3 # ten grams (0.01 kg)
        self.gamma = 9e-1 # air resistance
        self.x = 1.00 * self.l0 # mass position
        self.v = 0
        self.a = 0

        # Simulation parameters.
        self.dt = 1e-2
        self.t = 0
        self.X = []
        self.V = []
        self.A = []
        self.P = []
        self.T = []

        # Return system state on a reset.
        return self.state

    @property
    def action_space(self):
        # 1-dimensional action space.
        return np.zeros(1)

    @property
    def observation_space(self):
        # 3-dimensional observation space.
        return np.zeros(4)

    def step(self, action):
        # Action corresponds to position of piston, p:
        self.p = action[0]
        self.P.append(self.p)

        # Move the simulation forward one discrete timestep.
        dt = self.dt
        self.T.append(self.t)
        self.t += dt

        # Compute current acceleration.
        self.A.append(self.a)
        self.a = -(self.x - self.p - self.l0)/self.mass - self.gamma * self.v

        # Compute current velocity.
        self.V.append(self.v)
        self.v += self.a * dt

        # Compute the current position.
        self.X.append(self.x)
        self.x += self.v * dt

        return self.state, self.reward, self.done, self.info

    @property
    def info(self):
        # Return relevant environment information.
        return {}

    @property
    def state(self):
        # Current state vector of the system
        return [self.x, self. v, self.a, self.t]

    @property
    def done(self):
        # We will run the simulation for 10 seconds.
        return self.t > 10

    @property
    def reward(self):
        # Reward specifiction.
        bonus = 0
        if self.t < 5:
            if np.abs(self.x - self.l0)>1e-2:
                bonus = -100
            return -(self.x - self.l0)**2 + bonus
        else:
            if np.abs(self.x - 1.5 * self.l0)>1e-2:
                bonus = -100
            return -(self.x - 1.5 * self.l0)**2 + bonus
        # if self.t < 5:
        #     if np.abs(self.x - self.l0) > 1e-2:
        #         return -100 - np.abs(self.x - self.l0)
        #     else:
        #         return -np.abs(self.x - self.l0)
        # else:
        #     if np.abs(self.x - 1.5 * self.l0) > 1e-2:
        #         return -100 - np.abs(self.x - 1.5 * self.l0)
        #     else:
        #         return -np.abs(self.x - 1.5 * self.l0)


if __name__ == '__main__':
    import pylab as plt
    plt.ion()
    plt.close('all')

    sim = Wiggle()
    dt = sim.dt
    T = 10 # seconds
    p = 0

    for itr in range(int(T/sim.dt)):
        t = itr * dt
        if (t>5):
            while p<=0.5:
                p += 0.01/0.5 * dt
        sim.step([p])

    plt.plot(sim.T, sim.X)
    plt.ylim([0.5, 2.5])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mass Position (meters)')
    plt.savefig('ringing.png')
