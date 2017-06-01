
import numpy as np
import tensorflow as tf

#
# -----------------------------------------------------------------------
#
# Helper functions
#

#
# define a layer consisting of a convolution plus bias, followed by a relu and batch norm
#

def conv( x, filter_size=8, stride=2, num_filters=64, is_output=False, name="conv" ):

    filter_height, filter_width = filter_size, filter_size
    in_channels = x.get_shape().as_list()[-1]
    out_channels = num_filters

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", shape=[filter_height, filter_width, in_channels, out_channels],
                             initializer = tf.contrib.layers.variance_scaling_initializer() )
        b = tf.get_variable( "b", shape=[out_channels],
                             initializer = tf.contrib.layers.variance_scaling_initializer() )

        conv = tf.nn.conv2d( x, W, [1, stride, stride, 1], padding="SAME" )
        out = tf.nn.bias_add(conv, b)
        if not is_output:
#            out = tf.contrib.layers.batch_norm( tf.nn.relu(out) )
            out = tf.nn.relu(out)

    return out

#
# a transpose convolution, plus bias, followed by relu
#

def convt( x, out_shape, filter_size=8, stride=2, is_output=False, name="convt" ):

    filter_height, filter_width = filter_size, filter_size
    in_channels = x.get_shape().as_list()[-1]

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", shape=[filter_height, filter_width, out_shape[-1], in_channels],
                             initializer = tf.contrib.layers.variance_scaling_initializer() )
        b = tf.get_variable( "b", shape=[out_shape[-1]],
                             initializer = tf.contrib.layers.variance_scaling_initializer() )

        conv = tf.nn.conv2d_transpose( x, W, out_shape, [1, stride, stride, 1], padding="SAME" )
        out = tf.nn.bias_add( conv, b )
        if not is_output:
#            out = tf.contrib.layers.batch_norm( tf.nn.relu(out) )
            out = tf.nn.relu(out)

    return out

#
# a simple fully connected layer, plus bias, followed by a relu
#

def fc( x, out_size=50, is_output=False, name="fc" ):

    in_size = x.get_shape().as_list()[-1]

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", shape=[in_size, out_size],
                             initializer = tf.contrib.layers.variance_scaling_initializer() )
        b = tf.get_variable( "b", shape=[out_size],
                             initializer = tf.contrib.layers.variance_scaling_initializer() )
        out = tf.matmul(x, W) + b

        if not is_output:
#            out = tf.contrib.layers.batch_norm( tf.nn.relu( out ) )
            out = tf.nn.relu( out )

    return out

#
# -----------------------------------------------------------------------
#

def model( input_data ):

    BATCHSIZE = int( input_data.get_shape()[0] )

    # ----------------------------------------------
    # conv1a = conv( input_data, filter_size=3, stride=2, num_filters=64, name="conv1a" )
    # conv1b = conv( conv1a, filter_size=3, stride=2, num_filters=64, name="conv1b" )
    # flat_layer = tf.reshape( conv1b, [ BATCHSIZE, -1 ] )  # [ 16, 436800 ]

    # ----------------------------------------------
    conv1a = conv( input_data, filter_size=15, stride=2, num_filters=64, name="conv1a" )
    conv1b = conv( conv1a, filter_size=15, stride=2, num_filters=64, name="conv1b" )
    pool12 = tf.nn.max_pool( conv1b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool12" )

    conv2a = conv( pool12, filter_size=7, stride=2, num_filters=128, name="conv2a" )
    conv2b = conv( conv2a, filter_size=7, stride=2, num_filters=128, name="conv2b" )
    pool23 = tf.nn.max_pool( conv2b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool23" )

    flat_layer = tf.reshape( pool23, [ BATCHSIZE, -1 ] )  # [ 16, 436800 ]
    fc1 = fc( flat_layer, out_size=512, name="fc1" )
    fc2 = fc( fc1, out_size=128, name="fc2" )
    output = 0.3*fc( fc2, out_size=1, is_output=True, name="fc3" )

    # ----------------------------------------------
    # conv1a = conv( input_data, filter_size=3, stride=1, num_filters=64, name="conv1a" )
    # conv1b = conv( conv1a, filter_size=3, stride=1, num_filters=64, name="conv1b" )
    # pool12 = tf.nn.max_pool( conv1b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool12" )

    # conv2a = conv( pool12, filter_size=3, stride=1, num_filters=128, name="conv2a" )
    # conv2b = conv( conv2a, filter_size=3, stride=1, num_filters=128, name="conv2b" )
    # pool23 = tf.nn.max_pool( conv2b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool23" )

    # conv3a = conv( pool23, filter_size=3, stride=1, num_filters=256, name="conv3a" )
    # conv3b = conv( conv3a, filter_size=3, stride=1, num_filters=256, name="conv3b" )
    # pool34 = tf.nn.max_pool( conv3b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool34" )

    # conv4a = conv( pool34, filter_size=3, stride=1, num_filters=512, name="conv4a" )
    # conv4b = conv( conv4a, filter_size=3, stride=1, num_filters=256, name="conv4b" )
    # pool45 = tf.nn.max_pool( conv4b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool45" )

    # conv5a = conv( pool45, filter_size=3, stride=1, num_filters=128, name="conv5a" )
    # conv5b = conv( conv5a, filter_size=3, stride=1, num_filters=128, name="conv5b" ) # [16,273,25,1024]

    # flat_layer = tf.reshape( conv5b, [ BATCHSIZE, -1 ] )  # [ 16, 7168 ]

    # print "flat_layer size: ", flat_layer.get_shape()

    # fc1 = fc( flat_layer, out_size=512, name="fc1" )
    # fc2 = fc( fc1, out_size=128, name="fc2" )
    # output = 0.3*fc( fc2, out_size=1, is_output=True, name="fc3" )

    return output
