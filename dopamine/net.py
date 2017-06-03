import numpy as np
import tensorflow as tf
import pylab as plt
from tensorflow.contrib.opt import ScipyOptimizerInterface
dtype = 'float32'


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
        self.targets = tf.placeholder(dtype, [None, layer.shape[1]], name='targets')

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
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).\
                minimize(self.square_loss)
        # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1).\
        #         minimize(self.square_loss)


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
        while cost > tol:
            samples = np.random.permutation(N)[:batch_size]
            x_ = x[samples,:]
            y_ = y[samples,:]
            fd = {self.inputs: x_, self.targets: y_}
            _, cost = session.run([self.optimizer, self.square_loss],\
                    feed_dict=fd)
            if np.mod(itr, 50) == 0:
                print('> Loss at iteration {:d}: {:0.3f}'.format(itr, cost))
            itr += 1
            if itr > nb_itr: break

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

    # Create a function to learn.
    def func(x):
        return np.atleast_2d(3 * x[:,0]**1 + 2 * x[:,1]**2).T
        return np.atleast_2d(np.sin(x[:,0])**2 + np.cos(2*np.pi*x[:,1]/0.6)).T

    # Sample data.
    M = 5000
    X_train = np.random.uniform(2*np.pi, size=(M,2))
    y_train = func(X_train)

    # Create our neural network.
    x = tf.placeholder(dtype, [None, 2], name='neuralinput')
    layers = [(64, tf.nn.relu), (1, None)] # 64 hidden units; RELU activation
    # layers = [(32, tf.nn.relu), (1, None)] # 32 hidden units; RELU activation
    net = SimpleNet(x, layers)
    sess = tf.Session()

    ins = net.inputs
    tgt = net.targets

    feed = {ins: X_train, tgt: y_train}

    # Initialize.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Create the thing.
    loss = net.square_loss
    optimizer = ScipyOptimizerInterface(loss, options={'maxiter':100})
    optimizer.minimize(sess, feed_dict=feed)
    cost = sess.run(net.square_loss, feed_dict={net.inputs: X_train, net.targets: y_train})
    print(cost)

    # Train the network.
    # net.fit(sess, X_train, y_train, nb_itr=20000, batch_size=1000)


    plt.figure(100)
    plt.clf()
    plt.plot(y_train, net.predict(sess, X_train), 'o')
    plt.show()
    plt.title('Should be approximately a straight line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.axes().set_aspect('equal')
    plt.grid(True)


