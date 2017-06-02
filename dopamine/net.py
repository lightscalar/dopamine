import numpy as np
import tensorflow as tf
from dopamine.utils import *
from ipdb import set_trace as debug


class SimpleNet(object):
    '''Simple multilayer perceptron.'''

    def __init__(self, inputs, layers_config):
        '''Create a multilayer perceptron.
        INPUTS
            layer_cfg - array-like
                An array containing a collection of tuples,
                    (n_out, activation_fn)
                which specify layer output dimension and activation function.
        '''

        # Initialize the weights and biases of the network.
        self.params_dict = {}
        self.layers = {}
        self.inputs = inputs
        _inputs = inputs

        # Input data dictates the size of the input layer.
        n_in = int(inputs.shape[1])

        for itr, layer in enumerate(layers_config):

            # We're iterating over all layers in the network.
            n_out, activation = layer
            weight_name = 'w{:d}'.format(itr)
            bias_name = 'b{:d}'.format(itr)

            # Randomly initialize the layer weights.
            self.params_dict[weight_name] = w = \
                    tf.Variable(tf.random_normal((n_in, n_out), \
                    stddev=np.sqrt(n_in)), name=weight_name)

            # Randomly initialize the layer biases.
            self.params_dict[bias_name] = b = \
                    tf.Variable(tf.random_normal([n_out],\
                    stddev=np.sqrt(n_in)), name=bias_name)

            # Create layers via matrix multiplication!
            self.layers['layer_{:d}'.format(itr)] = layer = \
                    tf.add(tf.matmul(_inputs, w), b)

            # If activation function exists, use it; otherwise it's linear.
            if activation:
                layer = activation(layer)
            else:
                # the activation is, by default, linear.
                pass

            # Output of this layer becomes the input of the next.
            _inputs = layer
            n_in = int(n_out)

        # Final layer output is our network.
        self.outputs = layer

        # Define our target data placeholder.
        self.targets = tf.placeholder(dtype, [None, layer.shape[1]])

        # Keep track of the trainable parameters of this network.
        self.params = []
        for k, param in self.params_dict.items():
            self.params.append(param)

        # Compute the square loss.
        self.square_loss =  tf.reduce_mean(tf.squared_difference(self.outputs,\
                self.targets))

        # Create the optimizer (for use in value function fitting, etc.)
        # self.optimizer = tf.train.AdamOptimizer().\
        #         minimize(self.square_loss)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).\
        #         minimize(self.square_loss)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1).\
                minimize(self.square_loss)


    def fit(self, session, x, y, nb_itr=500, batch_size=1000, tol=1e-2):
        '''Fit network to batch data (x,y) using stochastic gradient descent.'''
        N = x.shape[0]
        if not batch_size:
            batch_size = N
        else:
            batch_size = np.min([batch_size, N])

        # for itr in range(nb_itr):
        cost = 1e4
        itr = 0
        while cost > 1.0:
            samples = np.random.permutation(N)[:batch_size]
            x_ = x[samples,:]
            y_ = y[samples,:]
            fd = {self.inputs: x_, self.targets: y_}
            _, cost = session.run([self.optimizer, self.square_loss],\
                    feed_dict=fd)
            if np.mod(itr, 50) == 0:
                print('> Current loss: {:0.3f}'.format(cost))
            itr += 1
            if itr > nb_itr: break
            # if cost < 1:
            #     break

    def predict(self, sess, x_):
        '''Forward propagate the specified state, x_, through the network.'''
        return sess.run(self.outputs, feed_dict={self.inputs: x_})

    def __call__(self, sess, x_):
        '''Alias for the predict method.'''
        return self.predict(sess, x_)

    @property
    def theta(self):
        '''Return a flattened version of all network parameters.'''
        theta = np.array([])

    @property
    def param_dim(self):
        '''Return the total dimension of the parameter space.'''
        dim = 0
        for p in self.params:
            dim += numel(p)
        return dim



if __name__ == '__main__':

    # Start over again!
    tf.reset_default_graph()

    # Define our network.
    x = tf.placeholder('float32', [None, 5])
    output_dim = 2
    layers = [(64, tf.nn.relu), (64, tf.nn.relu), (output_dim, None)]
    net = SimpleNet(x, layers)
    layers = [(32, tf.nn.relu), (32, tf.nn.relu), (output_dim, None)]
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
    # assert (np.sum(out, 1).mean() == 1)


