
# Results:
# BS 16 - SS 0.01 - converged at MSE 1.01


import numpy as np
import tensorflow as tf
from model import model

BATCHSIZE = 16
LEARNING_RATE = 0.00001

def whiten(x):
    x = x - np.mean( x )
    x = x / np.sqrt( np.var( x ) )
    return x

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

NUM_DATA = inputs.shape[0]

input_shape = [ BATCHSIZE, inputs.shape[1], inputs.shape[2], inputs.shape[3] ]
output_shape = [ BATCHSIZE, 1 ]

# -----------------------------------------------------

def np_loss( X ):
    return np.log( np.sqrt( np.mean( ( 182.0 * X )**2.0 ) ) ) / 0.6931471 # log(2.0)

# -----------------------------------------------------

tf.reset_default_graph()
sess = tf.Session()

input_data = tf.placeholder( tf.float32, input_shape )
output_data = tf.placeholder( tf.float32, output_shape )

with tf.name_scope( "model" ):
    output_hat = model( input_data )

    print "output_hat shape: ", output_hat.get_shape()

with tf.name_scope( "cost_function" ):
    loss = tf.reduce_sum( tf.nn.l2_loss( output_hat - output_data, name="loss" ) )
    # log_2 RMSE - our target is to get this below 2.5
    l2rmse = tf.log( tf.sqrt( tf.reduce_mean( ( 182.*output_hat - 182.*output_data)**2.0 ) ) ) / 0.6931471 # log(2.0)

optim = tf.train.AdamOptimizer( LEARNING_RATE ).minimize( loss )

saver = tf.train.Saver()
# WARM START
#saver.restore( sess, "./model.ckpt" )
    
sess.run( tf.initialize_all_variables() )

# ================================================================

vals = []
best = 100000000

DS = inputs.shape[0]
DS = int(int(DS/BATCHSIZE)*BATCHSIZE)

ohs = np.zeros((1,DS))

for iter in range( 1000000 ):

    inds = np.random.choice( NUM_DATA, size=[BATCHSIZE] )
        
    _, loss_val = sess.run( [optim,l2rmse], feed_dict={input_data:inputs[inds,:,:,:], output_data:outputs[inds]} )

#    print("  %d %.4f" % ( iter, loss_val ))

    if iter%100==0:

        for ind in range( 0,DS,BATCHSIZE ):
            oh = sess.run( output_hat, feed_dict={input_data:inputs[ind:ind+BATCHSIZE,:,:,:], output_data:outputs[ind:ind+BATCHSIZE]} )
            ohs[0,ind:ind+BATCHSIZE] = oh.T
        total_loss = np_loss( ohs.T - outputs[0:DS] )
        print "%d LOSS: %.4f" % ( iter, total_loss )

        vals.append( total_loss )
        np.save( 'vals.npy', vals )

#      if loss_val < best:
#          saver.save( sess, './model.ckpt' )
#          best = loss_val


lvs = []
nplvs = []
for ind in range( 0,DS,BATCHSIZE ):
    oh,lv = sess.run( [output_hat,l2rmse], feed_dict={input_data:inputs[ind:ind+BATCHSIZE,:,:,:], output_data:outputs[ind:ind+BATCHSIZE]} )
    ohs[0,ind:ind+BATCHSIZE] = oh.T
    lvs.append( lv )
    nplvs.append( np_loss( oh - outputs[ind:ind+BATCHSIZE] ) )
    if ind%100==0:
        print ind
