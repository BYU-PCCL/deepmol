
import numpy as np
import tensorflow as tf
from model import model

from augment import jitter_image
from data import *
from losses import *

# -----------------------------------------------------

inputs, outputs = load_data()
test_inds, train_inds = load_test_train_inds()

test_inputs = inputs[ test_inds, :, :, : ]
test_outputs = outputs[ test_inds, : ]

train_inputs = inputs[ train_inds, :, :, : ]
train_outputs = outputs[ train_inds, : ]

# -----------------------------------------------------

BATCHSIZE = 16
LEARNING_RATE = 0.001

input_shape = [ BATCHSIZE, inputs.shape[1], inputs.shape[2], inputs.shape[3] ]
output_shape = [ BATCHSIZE, 1 ]

# -----------------------------------------------------

tf.reset_default_graph()
sess = tf.Session()

input_ph = tf.placeholder( tf.float32, input_shape )
output_ph = tf.placeholder( tf.float32, output_shape )

with tf.name_scope( "model" ):
    output_hat = model( input_ph )

with tf.name_scope( "cost_function" ):
    loss = tf.reduce_sum( tf.nn.l2_loss( output_hat - output_ph, name="loss" ) )
    # log_2 RMSE - our target is to get this below 2.5
    l2rmse = tf.log( tf.sqrt( tf.reduce_mean( ( 182.*output_hat - 182.*output_ph)**2.0 ) ) ) / 0.6931471 # log(2.0)

optim = tf.train.AdamOptimizer( LEARNING_RATE ).minimize( loss )

saver = tf.train.Saver()
# WARM START
#saver.restore( sess, "./model.ckpt" )
    
sess.run( tf.initialize_all_variables() )

# ================================================================

results = []
best = 100000000

for iter in range( 1000000 ):

    inds = np.random.choice( train_inputs.shape[0], size=[BATCHSIZE] )
    train_images = np.squeeze(train_inputs[inds,:,:,:])
    train_images = np.expand_dims([jitter_image(image) for image in train_images], 3)

    _, opt_val, loss_val = sess.run( [optim,loss,l2rmse], feed_dict={input_ph:train_images, output_ph:train_outputs[inds,0:1]} )

    if iter % 30==0:

        tst_l1, tst_l2rmse, tst_ohs = np_total_loss( sess, output_hat, test_inputs, input_ph, test_outputs, output_ph, BATCHSIZE )
        tr_l1, tr_l2rmse, tr_ohs = np_total_loss( sess, output_hat, train_inputs, input_ph, train_outputs, output_ph, BATCHSIZE )

        print("  %d\toptobj=%.4f\t[l2rmse=%.4f]\tTest/train loss: %.4f %.4f" % ( iter, opt_val, loss_val, tst_l2rmse, tr_l2rmse ))

        results.append( [ tst_l1, tst_l2rmse, tr_l1, tr_l2rmse ] )
        np.save( 'results.npy', results )

    if loss_val < best:
        saver.save( sess, './model.ckpt' )
        best = loss_val
