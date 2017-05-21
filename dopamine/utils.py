import tensorflow as tf
import numpy as np


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
    '''Returns flattened version of gradient w.r.t. specified parameters.'''
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v,grad) in \
            zip(var_list, grads)],0)
