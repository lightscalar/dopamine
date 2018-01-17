from dopamine.trpo import *
from dopamine.stochastics import *
from keras.models import model_from_json
import gym

# Create an instance of the lunar lander environment.
env = gym.make('LunarLanderContinuous-v2')
D = env.observation_space.shape[0]
A = env.action_space.shape[0]

layers = []
layers.append({'input_dim': D, 'units': 128, 'activation': 'relu'})
layers.append({'units': 128, 'activation': 'relu'})
layers.append({'units': A, 'activation': 'tanh'})
policy = create_mlp(layers)

# Create a categorical random variable with appropriate number of classes.
pdf = DiagGaussian(A)

# Create our TRPO agent.
agent = TRPOAgent('continuous_v2', env, policy, pdf, load_model=False)
agent.learn()


# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(reward)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
#     print('Finished with this episode!')

env.close()
