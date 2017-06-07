from dopamine.trpo import *
from dopamine.lineworld import *

# Create a lineworld instance.
env = LineWorld()

# Create the policy model.
layers = []
layers.append({'input_dim': env.D, 'units': 16})
layers.append({'units': env.nb_actions, 'activation': 'tanh'})
policy = create_mlp(layers)
pdf = DiagGaussian(env.nb_actions)

# So we have our three necessary objects! Let's create a TRPO agent.
agent = TRPOAgent(env, policy, pdf)
paths = agent.learn()


