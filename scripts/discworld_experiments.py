from dopamine.trpo import *
from dopamine.discworld import *

# Create a lineworld instance.
env = DiscWorld()

# Create the policy model.
layers = []
layers.append({'input_dim': env.D, 'units': 100})
# layers.append({'units': 100, 'activation': 'relu'})
# layers.append({'units': 100, 'activation': 'relu'})
layers.append({'units': env.nb_actions, 'activation': 'tanh'})
policy = create_mlp(layers)

# Use Gaussian continuous action vectors.
pdf = DiagGaussian(env.nb_actions, stddev=1.0)

# So we have our three necessary objects! Let's create a TRPO agent.
agent = TRPOAgent(env, policy, pdf)
agent.learn()


