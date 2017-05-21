import numpy as np
import tensorflow as tf
from dopamine.utils import *
from ipdb import set_trace as debug


class SimpleNet(object):
    '''Simple multilayer perceptron.'''

    def __init__(self, x, layers_config):
        '''Create a multilayer perceptron.
        INPUTS
            layer_cfg - array-like
                An array containing a collection of tuples,
                    (n_out, _activation_fn)
                which specify layer output dimension and activation function.
        '''

        # Initialize a tensorflow session.
        self.sess = tf.Session()

        # Initialize the weights and biases of the network.
        self.weights = {}
        self.biases = {}
        self.layers = {}
        self.x = x

        # Input data dictates the size of the input layer.
        n_in = int(x.shape[1])
        _input = x

        for itr, layer in enumerate(layers_config):

            # We're iterating over all layers in the network.
            n_out, activation = layer
            weight_name = 'w{:d}'.format(itr)
            bias_name = 'b{:d}'.format(itr)

            # Randomly initialize the layer weights.
            self.weights[weight_name] = w = \
                    tf.Variable(tf.random_normal((n_in, n_out), \
                    stddev=np.sqrt(n_in)), name=weight_name)

            # Randomly initialize the layer biases.
            self.biases[bias_name] = b = \
                    tf.Variable(tf.random_normal([n_out],\
                    stddev=np.sqrt(n_in)), name=bias_name)

            # Create layers via matrix multiplication!
            self.layers['layer_{:d}'.format(itr)] = layer = \
                    tf.add(tf.matmul(_input, w), b)

            # If activation function exists, use it; otherwise it's linear.
            if activation:
                layer = activation(layer)
            else:
                # the activation is, by default, linear.
                pass

            # Output of this layer becomes the input of the next.
            _input = layer
            n_in = int(n_out)

        # Final layer output is our network.
        self.output = layer

        # Keep track of the trainable parameters of this network.
        self.vars = []
        for _, matrix in self.weights.items():
            self.vars.append(matrix)
        for _, vector in self.biases.items():
            self.vars.append(vector)


    def predict(self, x_, sess):
        '''Forward propagate the specified state, x_, through the network.'''
        output = sess.run(self.output, feed_dict={self.x: x_})
        return output


    @property
    def theta(self):
        '''Return a flattened version of all network parameters.'''
        theta = np.array([])




if __name__ == '__main__':

    # Start over again!
    tf.reset_default_graph()

    # Define our network.
    x = tf.placeholder('float32', [None, 5])
    output_dim = 2
    layers = [(64, tf.nn.relu), (64, tf.nn.relu), (output_dim, tf.nn.softmax)]
    net = SimpleNet(x, layers)
    layers = [(32, tf.nn.relu), (32, tf.nn.relu), (output_dim, tf.nn.softmax)]
    net_2 = SimpleNet(x, layers)

    # Create some sample data.
    nb_samples = 126
    x_ = np.random.randn(126, 5)

    # Forward propagate this data through the network.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        pred = net.output
        sess.run(init)
        out = sess.run(pred, feed_dict={x: x_})
        out_2 = net.predict(x_, sess)
        all_vars = sess.run(net.vars)

    assert( out.shape == (nb_samples, output_dim) )

    # If we're talking about softmax output.
    assert (np.sum(out, 1).mean() == 1)


