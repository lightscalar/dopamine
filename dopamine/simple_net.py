import numpy as np
from ipdb import set_trace as debug


'''
------------------------------------------------------------------------------
DEFINE ACTIVATION FUNCTIONS
------------------------------------------------------------------------------
'''
def relu(x):
    # Rectified Linear Unit (RELU) activation; only positive values survive.
    x[x<0] = 0
    return x


def tanh(x):
    # Tanh activation function.
    return np.tanh(x)


def linear(x):
    # Linear activation function; simply return inputs untouched.
    return x


'''
------------------------------------------------------------------------------
DEFINE NEURAL NETWORK CLASS
------------------------------------------------------------------------------
'''
class Network(object):

    def __init__(self, input_dimension, layers):
        '''Initialize the weights and layer of the network.
        INPUTS
            input_dimension - int
                The input dimension of the data to the network.
            layers - array_like
                An array of tuples:
                    [(nb_units_1, act_fun_1), (nb_units_2, act_fun_2), ... ]
                specifying the number of units (equivalently, the output
                dimension of the layer), as well as the type of activation
                function to be used for that for network layers.
        '''
        self.d_in = input_dimension
        self.layers = {}
        d_in = input_dimension
        self.W = {}
        self.b = {}
        self.activation = {}

        # Initialize the weight matrices and biases of the network...
        for layer_num, layer in enumerate(layers):
            d_out, activation = layer
            self.W[layer_num] = np.sqrt(d_in) * np.random.randn(d_in, d_out)
            self.b[layer_num] = np.sqrt(d_in) * np.random.randn(d_out)
            self.activation[layer_num] = activation
            d_in = d_out # next layer must accept n_out inputs because math

    def forward(self, inputs):
        '''Forward propagate the inputs through the neural network.
        INPUTS
            inputs - array_like
                An input to the neural network. Must match the specfied input
                dimension or errors will be thrown.
        OUTPUTS
            output - array_like
                The output of the neural network. Its dimension is determined
                by the number of units in the final specified layer.
        '''
        out = np.array(inputs)
        # Loop through each layer; multiply weights W and add bias b; apply the
        # activation function.
        for key in self.W.keys():
            W = self.W[key]
            b = self.b[key]
            act = self.activation[key]
            z = np.dot(out, W) + b
            out = act(z)
        return out


if __name__ == '__main__':

    '''EXAMPLES OF FORWARD PROPAGATION...'''
    # Define layers of neural network with two hidden layers and tanh
    # activation functions, and scalar output.
    input_dimension = 1
    layers = [(128, tanh), (128, tanh), (1, linear)]

    # Create a network.
    net = Network(input_dimension, layers)

    # Put in some input, generate some output
    output = net.forward(0.3)
    print('Output of first network is {:.3f}'.format(output[0][0]))

    # Create a new network with RELU activations and five-dimensional output.
    input_dimension = 2
    layers = [(256, relu), (256, relu), (5, linear)]
    net = Network(input_dimension, layers)
    output = net.forward([0.3, 0.7])
    print('Output of second network is {:s}'.format(str(output)))
