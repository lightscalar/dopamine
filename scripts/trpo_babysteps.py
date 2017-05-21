'''Try to train a simple neural network using Trust Region/Line Search.'''
import numpy as np
from keras.utils.np_utils import to_categorical
import pylab as plt
import seaborn as sns
from seaborn import xkcd_rgb as xkcd
import tensorflow as tf
from dopamine.basics import *


def sample_data(points_per_class, dimension=2, number_classes=3):
    '''Generate sample data points on a spiral.'''
    N = points_per_class # number of points per class
    D = dimension # dimensionality
    K = number_classes # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in np.arange(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y


if __name__ =='__main__':

    # Create some testing data.
    X,y = sample_data(100)
    y_ = to_categorical(y)


    # Plot points to demonstrate the nonlinear decision boundaries.
    colors = np.argmax(y_,1)
    plt.ion()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

    # Create a neural network to train.
    _x = tf.placeholder('float32', [None, 2])
    _y = tf.placeholder('float32', [None, 3])
    output_dim = 3
    layers = [(32, tf.nn.relu), (32, tf.nn.relu), (output_dim, None)]
    net = SimpleNet(_x, layers)
    squared_deltas = tf.reduce_sum(tf.square( tf.subtract(net.output, _y)),1)
    loss = tf.reduce_mean(squared_deltas,0)
    grads = tf.gradients(loss, net.vars)
    theta = flatten_collection(net.vars)
    flat_grad = flat_gradient(loss, net.vars)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        fd = {_x:X, _y:y_}
        print(sess.run(loss, feed_dict=fd))
        out = sess.run(squared_deltas, feed_dict=fd)
        g_out = sess.run(grads, feed_dict=fd)
        theta = sess.run(theta)
        fgrad = sess.run(flat_grad, feed_dict=fd)
