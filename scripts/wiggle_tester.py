from dopamine.gym import make_env
from keras.models import load_model
import numpy as np
from mathtools.utils import Vessel
from ipdb import set_trace as debug
from gym import wrappers
import pylab as plt


if __name__ == '__main__':

    # Create an environment.
    env = make_env('wiggle')

    # Create a policy model.
    name = 'wiggles'
    location = '../weights/policy_{:s}.h5'.format(name)
    policy = load_model(location)
    obsfilt = Vessel('../weights/filters_{:s}.dat'.format(name))

    done = False
    obs = env.reset()
    while not done:
        obs = obsfilt.filt(obs)
        action = policy.predict(np.atleast_2d(obs))
        obs, reward, done, info = env.step(action[0])

        if done:
            print('REWARD: {:2f}'.format(reward))
            break

    plt.ion()
    plt.close('all')
    plt.figure(100)
    plt.plot(env.T, env.X)
    plt.ylim([0.5, 2.5])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mass Position (meters)')
    plt.savefig('rl-snap.png')

    plt.figure(200)
    plt.plot(env.T, env.A)
    # plt.ylim([0.5, 2.5])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mass Position (meters)')
    plt.savefig('control-signal.png')
