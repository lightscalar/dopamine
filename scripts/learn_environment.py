'''Start learning the specified environment.'''
import argparse
from dopamine.gym import make_env
from dopamine.stochastics import *
from dopamine.trpo import *
import gym
import yaml
from ipdb import set_trace as debug

# Handle command line arguments.
parser = argparse.ArgumentParser()

# Define the command line arguments that we'll accept.
parser.add_argument(
    "-v",
    "--verbosity",
    type=int,
    choices=[0, 1, 2],
    default=1,
    help="increase output verbosity"
    )
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="specify the configuration file to use"
    )

# Grab all the arguments that were passed in to the file.
args = parser.parse_args()

try:
    # Load the specified configuration file.
    f = open(args.config, 'r')
    config = yaml.load(f)
except:
    # Ach. Something went sideways.
    print('Could not read the config file.')
    raise

# Load the environment.
if config['open_ai_env']:
    env = gym.make(config['environment'])
else:
    env = make_env(config['environment'])


# State/action space dimensions...
D = env.observation_space.shape[0]
A = env.action_space.shape[0]

# Are we loading models?
load_models = config['load_models']

# Create the neural network policy.
layers = []
nb_layers = len(config['policy'])
for layer_nb, policy_layer in enumerate(config['policy']):
    layer = policy_layer['layer']
    if layer_nb == 0: # first layer
        layer['input_dim'] = D
    elif layer_nb == nb_layers - 1: # last layer
        layer['units'] = A
    layers.append(layer)

policy = create_mlp(layers)

# Create the probability density function.
if config['pdf'] == 'DiagGaussian':
    pdf = DiagGaussian(A)

# Create an instance of the TRPO agent.
agent = TRPOAgent(config['experiment_tag'], env, policy, pdf, load_models)

# Start learning!
agent.learn()
