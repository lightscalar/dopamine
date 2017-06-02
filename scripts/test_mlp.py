from dopamine.utils import *

layers = []
layers.append({'input_dim': 4, 'units': 64})
layers.append({'units': 1, 'activation': 'softmax'})
model = create_mlp(layers)

