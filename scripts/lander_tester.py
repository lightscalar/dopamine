import gym
from keras.models import load_model
import numpy as np
from mathtools.utils import Vessel
from ipdb import set_trace as debug
from gym import wrappers


if __name__ == '__main__':

    # Create an environment.
    env = gym.make('LunarLanderContinuous-v2')
    # env = wrappers.Monitor(env, 'tmp/lunar-continuous-v1', force=True)

    # Create a policy model.
    name = 'continuous_v2'
    location = 'weights/policy_{:s}.h5'.format(name)
    policy = load_model(location)
    obsfilt = Vessel('weights/filters_{:s}.dat'.format(name))

    nb_episodes = 10
    for i_episode in range(nb_episodes):
        print('> Episode {:d} of {:d}'.format(i_episode, nb_episodes))
        obs = env.reset()
        done = False
        while not done:
            env.render()
            obs = obsfilt.filt(obs)
            action = np.clip(policy.predict(np.atleast_2d(obs)),-1,1)
            # action = env.action_space.sample()
            # probs = np.exp(policy.predict(np.atleast_2d(obs)))[0]
            # action = np.random.choice(len(probs), 1, p=probs/sum(probs))
            # obs, reward, done, info = env.step(action)
            obs, reward, done, info = env.step(action[0])
            if done:
                print('REWARD: {:2f}'.format(reward))
                break

    env.close()
