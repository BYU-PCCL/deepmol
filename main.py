
# Results:
# BS 16 - SS 0.01 - converged at MSE 1.01


import numpy as np
import tensorflow as tf
from model import model

BATCHSIZE = 16
LEARNING_RATE = 0.001

def whiten(x):
    x = x - np.mean( x )
    x = x / np.sqrt( np.var( x ) )
    return x

if not 'inputs' in vars():
    print "Loading data..."
    inputs_v = np.load( '/opt/wingated/mouli/inputs_valence.npy' )
    inputs_v = inputs_v.transpose( [2,0,1] )  # put batch dimension first
    inputs_v = whiten( inputs_v )

    #inputs_c = np.load( '/opt/wingated/mouli/inputs_core.npy' )
    #inputs_c = inputs_c.transpose( [2,0,1] )  # put batch dimension first
    #inputs_c = whiten( inputs_c )

    inputs = np.reshape( inputs_v, [ inputs_v.shape[0], inputs_v.shape[1], inputs_v.shape[2], 1 ] ) # add "channel" dimension
    #inputs = np.zeros(( 4357, 398, 512, 2 ))
    #inputs[:,:,:,0] = inputs_v
    #inputs[:,:,:,1] = inputs_c

    inputs = inputs.astype('float32')

    outputs = np.load( '/opt/wingated/mouli/outputs.npy' )
    outputs = outputs.transpose()
    outputs = whiten( outputs )

    # outputs: mean=-1470, var=33196.891


NUM_DATA = inputs.shape[0]

input_shape = [ BATCHSIZE, inputs.shape[1], inputs.shape[2], inputs.shape[3] ]
output_shape = [ BATCHSIZE, 1 ]

# -----------------------------------------------------

def np_loss( X ):
    return np.log( np.sqrt( np.mean( ( 182.0 * X )**2.0 ) ) ) / 0.6931471 # log(2.0)

def np_loss_now( X ):
    return np.log( np.sqrt( np.mean( X**2.0 ) ) ) / 0.6931471 # log(2.0)

# -----------------------------------------------------

tf.reset_default_graph()
sess = tf.Session()

input_data = tf.placeholder( tf.float32, input_shape )
output_data = tf.placeholder( tf.float32, output_shape )

with tf.name_scope( "model" ):
    output_hat = model( input_data )

with tf.name_scope( "cost_function" ):

    print "output_hat shape: ", output_hat.get_shape()
    print "output_data shape: ", output_data.get_shape()

    loss = tf.reduce_sum( tf.nn.l2_loss( output_hat - output_data, name="loss" ) )
    # log_2 RMSE - our target is to get this below 2.5
    l2rmse = tf.log( tf.sqrt( tf.reduce_mean( ( 182.*output_hat - 182.*output_data)**2.0 ) ) ) / 0.6931471 # log(2.0)

optim = tf.train.AdamOptimizer( LEARNING_RATE ).minimize( loss )

saver = tf.train.Saver()
# WARM START
#saver.restore( sess, "./model.ckpt" )
    
sess.run( tf.initialize_all_variables() )

# ================================================================

def np_total_loss():
    global inputs, input_data, outputs, output_data, BATCHSIZE

    DS = inputs.shape[0]
    DS = int(int(DS/BATCHSIZE)*BATCHSIZE)

    ohs = np.zeros((DS,1))

    for ind in range( 0,DS,BATCHSIZE ):
        oh = sess.run( output_hat, feed_dict={input_data:inputs[ind:ind+BATCHSIZE,:,:,:], output_data:outputs[ind:ind+BATCHSIZE,0:1]} )
        ohs[ind:ind+BATCHSIZE,0:1] = oh

    l2rmse_loss = np_loss( ohs - outputs[0:DS,0:1] )
    l1_loss = np.mean( np.abs( ohs - outputs[0:DS,0:1] ) )

#    oh = sess.run( output_hat, feed_dict={input_data:inputs[inds,:,:,:], output_data:outputs[inds]} )
#    total_loss = np_loss( oh - outputs[inds,0:1] )

    print "L1: %4f\tLOSS: %.4f" % ( l1_loss, l2rmse_loss )

    return l1_loss, l2rmse_loss, ohs


# ================================================================

vals = []
best = 100000000

for iter in range( 1000000 ):

    inds = np.random.choice( NUM_DATA, size=[BATCHSIZE] )

    _, opt_val, loss_val = sess.run( [optim,loss,l2rmse], feed_dict={input_data:inputs[inds,:,:,:], output_data:outputs[inds,0:1]} )

    print("  %d\toptobj=%.4f\t[l2rmse=%.4f]" % ( iter, opt_val, loss_val ))

    if iter % 100==0:
        np_total_loss()
#        vals.append( total_loss )
#        np.save( 'vals.npy', vals )
#      if loss_val < best:
#          saver.save( sess, './model.ckpt' )
#          best = loss_val
