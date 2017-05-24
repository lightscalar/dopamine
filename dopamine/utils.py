import tensorflow as tf
import numpy as np

dtype='float32'


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
        shapes = map(var_shape, params)

        # Compute total length of the theta vector.
        total_size = np.sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])

        # Loop through all variables; reshape portions of theta back into the
        # W matrices and b vectors, etc.
        assigns = []
        for (shape, v) in zip(shapes, params):
            size = np.prod(shape)
            assigns.append(v, tf.reshape(theta[start:start+size],shape))
            start += size

        # Create a tensorflow operation that makes the above assignment.
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        '''Given a flat theta vector, put these values back into our neural
           network.
        '''
        self.op.eval(session=self.session, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, params):
        '''Flatten trainable parameters into a long vector.'''
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in params], axis=0)

    def __call__(self):
        '''Evalutate the tensorflow concatentation operation.'''
        return self.op.eval(session=self.session)


def make_tangents(tangent, params):
    '''Build list of variables that map network parameters to a flat tangent.'''
    shapes = map(var_shape, params)
    start = 0
    tangents = []
    for shape in shapes:
        size = np.prod(shape)
        param = tf.reshape(tangent[start:(start+size)], shape)
        tangents.append(param)
        start += size
    return tangents
