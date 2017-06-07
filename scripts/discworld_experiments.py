from dopamine.trpo import *
from dopamine.discworld import *

# Create a lineworld instance.
env = DiscWorld()

# Create the policy model.
layers = []
layers.append({'input_dim': env.D, 'units': 32})
layers.append({'units': env.nb_actions, 'activation': 'tanh'})
policy = create_mlp(layers)

# Use Gaussian continuous action vectors.
pdf = DiagGaussian(env.nb_actions)

# So we have our three necessary objects! Let's create a TRPO agent.
agent = TRPOAgent(env, policy, pdf)
agent.learn()


