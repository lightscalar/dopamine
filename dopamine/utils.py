import tensorflow as tf
import numpy as np
from ipdb import set_trace as debug
from keras.models import Sequential
from keras.layers import Dense
from scipy.signal import lfilter
from dopamine.running_stats import *


dtype='float32'


class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


def filter_states(state, states):
    '''Filter the states using a running mean/std.'''
    if len(states)>0:
        running_mean = np.mean(states)
        running_std = np.std(states)
        return (state - running_mean)/running_std
    else:
        return

def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.
    inputs
    ------
    x: ndarray
    gamma: float
    outputs
    -------
    y: ndarray with same shape as x, satisfying
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    assert x.ndim >= 1
    return lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


def create_mlp(layers, cfg=None):
    '''Creates a Keras-based MLP.'''
    cfg = cfg if cfg else {}
    cfg.setdefault('optimizer', 'rmsprop')
    cfg.setdefault('loss', 'mae')

    model = Sequential()
    for layer in layers:
        layer.setdefault('units', 64)
        layer.setdefault('activation', 'relu')
        layer.setdefault('activation', 'relu')
        layer.setdefault('kernel_initializer', 'glorot_normal')
        model.add(Dense(**layer))

    model.compile(**cfg)
    return model


def discounted_sum(r, discount_factor):
    '''Computed discounted reward sum.'''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)): # start at the end, work backwards.
        running_add = running_add * discount_factor + r[t]
        discounted_r[t] = running_add
    return discounted_r


def flatten(tensor):
    '''Flattens a tensor into a one dimensional object.'''
    shape = tensor.shape
    return tf.reshape(tensor, [int(np.prod(shape))])


def numel(tensor):
    '''Return the total number of elements in a tensor.'''
    return np.prod(var_shape(tensor))


def var_shape(tensor):
    '''Returns the shape of the tensor. Throws error if unknown dimension is
       present. So don't use this with placeholders; only Variables.
    '''
    # Grab shape from tensor.
    shape = [k.value for k in tensor.get_shape()]

    # No unknown dimensions allowed!
    assert all(isinstance(a, int) for a in shape)

    # And we're done.
    return shape


def flatten_collection(var_list):
    '''Returns flattened list of concatenated parameters.'''
    return tf.concat([tf.reshape(v, [numel(v)]) for v in var_list],0)


def flat_gradient(loss, var_list):
    '''Returns flattened version of gradient of loss w.r.t. specified
       parameters.
    '''
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v,grad) in \
            zip(var_list, grads)],0)


def slice_2d(x, indx0, indx1):
    '''Takes a two dimensional slice through a tensor.'''
    indx0 = tf.cast(indx0, tf.int32)
    indx1 = tf.cast(indx1, tf.int32)
    shape = tf.cast(tf.shape(x), tf.int32)
    ncols = shape[1]
    x_flat = tf.reshape(x,[-1])
    return tf.gather(x_flat, indx0 * ncols + indx1)


class SetFromFlat(object):
    '''Reconstitute the network weights and biases from a flat theta vector.'''

    def __init__(self, session, params):

        # Assign local tensorflow session.
        self.session = session

        # Collect shapes of weights and biases in the neural network.
        shapes = [*map(var_shape, params)]

        # Compute total length of the theta vector.
        total_size = np.sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])

        # Loop through all variables; reshape portions of theta back into the
        # W matrices and b vectors, etc.
        assigns = []
        start = 0
        for (shape, param) in zip(shapes, params):
            size = np.prod(shape)
            assigns.append(tf.assign(param, \
                    tf.reshape(theta[start:start+size], shape)))
            start += size

        # Create a tensorflow operation that executes these assignments.
        self.todo = tf.group(*assigns)

    def __call__(self, theta):
        '''Given a flat theta vector, put these values back into our neural
           network weights/biases.'''
        self.session.run(self.todo, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, params):
        '''Flatten trainable parameters into a long vector.'''
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in params], axis=0)

    def __call__(self):
        '''Evaluate the tensorflow concatentation operation.'''
        return self.op.eval(session=self.session)


def make_tangents(tangent, params):
    '''Build list of variables that map network parameters to a flat tangent.'''
    shapes = [*map(var_shape, params)]
    start = 0
    tangents = []
    for shape in shapes:
        size = np.prod(shape)
        param = tf.reshape(tangent[start:(start+size)], shape)
        tangents.append(param)
        start += size
    return tangents


def conjugate_gradient(f_Ax, b, cg_iters=10, tol=1e-10):
    '''Performs conjugate gradient descent.
    ARGS
        f_Ax - function
            Function returning the matrix/vector product of interest -- the
            Ax in the equation Ax=b that we're trying to solve.
        b - array_like
            The right hand side of the Ax=b equation.
        cg_iters - int [default: 10]
            The number of iterations that we allow.
        tol - float [default: 1e-10]
            The residual tolerance. If we drop below this, stop iterating.
    OUT
        x - array_like
            The approximate solution to the problem Ax = b, as determined by
            the conjugate gradient method.
    '''
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    minval = np.inf
    bestx = x
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < minval:
            bestx = x
        print('RDOT: {:.4f}'.format(rdotr))
        if rdotr < tol:
            break
    return x


def linesearch(f, x, fullstep, expected_improve_rate):
    '''Performs a linesearch in direction of fullstep, starting at x, in
    order to minimize f.
    ARGS
        f - function
            Function evaluating the cost at point x.
        x - array_like
            The starting point of the line search.
        full_step - array_like
            The direction in which we should be searching.
        expected_improve_rate - float
            Based on local slope, how much improvement do we expect?
    OUT
        x - array_like
            The best solution found along the specified line.
    '''
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print('A/E/R: {:.3f}/{:.3f}/{:.3f}'.\
                format(actual_improve, expected_improve, ratio))
        if (ratio > accept_ratio) and (actual_improve > 0):
            print('Line search success.')
            return True, xnew
    print('Line search fail.')
    return False, x


