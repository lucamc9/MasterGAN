import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                          decay=self.momentum,
                          updates_collections=None,
                          epsilon=self.epsilon,
                          scale=True,
                          is_training=train,
                          scope=self.name)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

class CapsConv(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_units: integer, the length of the output vector of a capsule.
        with_routing: boolean, this capsule is routing with the
        lower-level layer capsule.
        num_outputs: the number of capsule in this layer.
    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_units, with_routing=True):
        self.num_units = num_units
        self.with_routing = with_routing
    def __call__(self, sess, input, num_outputs, batch_size, kernel_size=None, stride=None):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size = batch_size
        self.sess = sess
        if not self.with_routing:
            #assert input.get_shape() == [None, 28,28,256]

            capsules = tf.contrib.layers.conv2d(input, self.num_outputs,
                            self.kernel_size, self.stride, padding="VALID",
                            activation_fn=tf.nn.relu)
            capsules = tf.reshape(capsules, (self.batch_size, -1, self.num_units, 1))

            # [batch_size, 1152, 8, 1]
            capsules = squash(capsules)
            print(capsules.get_shape())
            assert capsules.get_shape() == [self.batch_size, 1152, 8, 1]
            return(capsules)
        else:
            # the DigitCap layer
            # reshape the input into shape [128,1152,8,1]
            input = tf.reshape(input, shape=(self.batch_size, 1152, 8,1))

            #b_IJ : [1, num_caps_1, num_caps_1_plus_1, 1]
            b_IJ = tf.zeros(shape=[1,1152,10,1], dtype=np.float32)
            capsules = []
            for j in range(self.num_outputs):
                with tf.variable_scope('caps_' + str(j)):
                    caps_j, b_IJ = capsule(self.sess, input, b_IJ, j)
                    capsules.append(caps_j)

            #return a tensor with shape [batch_size,10,16,1]
            capsules = tf.concat(capsules, axis=1)
            assert capsules.get_shape() == [self.batch_size,10,16,1]

        return(capsules)

class CapsConv2(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_units: integer, the length of the output vector of a capsule.
        with_routing: boolean, this capsule is routing with the
        lower-level layer capsule.
        num_outputs: the number of capsule in this layer.
    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_units, with_routing=True):
        self.num_units = num_units
        self.with_routing = with_routing
    def __call__(self, sess, input, num_outputs, batch_size, kernel_size=None, stride=None):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size = batch_size
        self.sess = sess
        if not self.with_routing:
            #assert input.get_shape() == [None, 28,28,256]

            capsules = tf.contrib.layers.conv2d(input, self.num_outputs,
                        self.kernel_size, self.stride, padding="VALID",
                        activation_fn=tf.nn.relu)
            capsules = tf.reshape(capsules, (self.batch_size, -1, self.num_units, 1))

            # [batch_size, 1152, 8, 1]
            capsules = squash(capsules)
            print(capsules.get_shape())
            assert capsules.get_shape() == [self.batch_size, 1152, 8, 1]
            return(capsules)

        else:
            # the DigitCap layer
            # reshape the input into shape [128,1152,8,1]
            input = tf.reshape(input, shape=(self.batch_size, 1152, 8,1))

            #b_IJ : [1, num_caps_1, num_caps_1_plus_1, 1]
            b_IJ = tf.zeros(shape=[1,1152,10,1], dtype=np.float32)
            capsules = []
            for j in range(self.num_outputs):
                with tf.variable_scope('caps_' + str(j)):
                    caps_j, b_IJ = capsule(self.sess, input, b_IJ, j)
                    capsules.append(caps_j)

            #return a tensor with shape [batch_size,10,16,1]
            capsules = tf.concat(capsules, axis=1)
            assert capsules.get_shape() == [self.batch_size,10,16,1]

        return(capsules)

def capsule(sess, input, b_IJ, idx_j):
    ''' The routing algorithm for one capsule in the layer l+1.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, length(u_i)=8, 1]
        shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, 1, length(v_j)=16, 1] representing the
        vector output `v_j` of capsule j in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
    '''
    with tf.variable_scope('routing'):

        w_initializer = np.random.normal(size=[1, 1152, 8, 16], scale=0.01)

        W_Ij = tf.Variable(w_initializer, dtype=tf.float32)
        sess.run(tf.global_variables_initializer())
        # repeat W_Ij with batch_size times to shape [batch_size, 1152, 8, 16]
        W_Ij = tf.tile(W_Ij, [64, 1, 1, 1])

        # calc u_hat
        # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 16, 1]
        u_hat = tf.matmul(W_Ij, input, transpose_a=True)
        assert u_hat.get_shape() == [64, 1152, 16, 1]

        shape = b_IJ.get_shape().as_list()
        size_splits = [idx_j, 1, shape[2] - idx_j - 1]
        for r_iter in range(3):
            # line 4:
            # [1, 1152, 10, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            assert c_IJ.get_shape() == [1, 1152, 10, 1]

            # line 5:
            # weighting u_hat with c_I in the third dim,
            # then sum in the second dim, resulting in [batch_size, 1, 16, 1]
            b_Il, b_Ij, b_Ir = tf.split(b_IJ, size_splits, axis=2)
            c_Il, c_Ij, b_Ir = tf.split(c_IJ, size_splits, axis=2)
            assert c_Ij.get_shape() == [1, 1152, 1, 1]

            s_j = tf.multiply(c_Ij, u_hat)
            s_j = tf.reduce_sum(tf.multiply(c_Ij, u_hat),
                                axis=1, keep_dims=True)
            assert s_j.get_shape() == [64, 1, 16, 1]

            # line 6:
            # squash using Eq.1, resulting in [batch_size, 1, 16, 1]
            v_j = squash(s_j)
            assert s_j.get_shape() == [64, 1, 16, 1]

            # line 7:
            # tile v_j from [batch_size ,1, 16, 1] to [batch_size, 1152, 16, 1]
            # [16, 1].T x [16, 1] => [1, 1], then reduce mean in the
            # batch_size dim, resulting in [1, 1152, 1, 1]
            v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)
            assert u_produce_v.get_shape() == [64, 1152, 1, 1]
            b_Ij += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
            b_IJ = tf.concat([b_Il, b_Ij, b_Ir], axis=2)

        return(v_j, b_IJ)

def squash(vector):
    '''Squashing function.
    Args:
        vector: A 4-D tensor with shape [batch_size, num_caps, vec_len, 1],
    Returns:
        A 4-D tensor with the same shape as vector but
        squashed in 3rd and 4th dimensions.
    '''
    vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector)))  # a scalar
    scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
    vec_squashed = scalar_factor * tf.divide(vector, vec_abs)  # element-wise
    return(vec_squashed)
