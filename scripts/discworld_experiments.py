from dopamine.trpo import *
from dopamine.discworld import *

# Create a lineworld instance.
env = LineWorld()
nb_actions = env.nb_actions

# Create a placeholder for inputs to the policy.
state_vector = tf.placeholder('float32', [None, env.D])

# Create the policy model.
layers = []
layers.append({'input_dim': env.D, 'units': 16})
layers.append({'units': 1, 'activation': 'tanh'})
policy = create_mlp(layers)
pdf = DiagGaussian(nb_actions)

# So we have our three necessary objects! Let's create a TRPO agent.
agent = TRPOAgent(env, policy, pdf)
paths = agent.learn()


