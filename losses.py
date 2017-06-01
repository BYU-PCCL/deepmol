import numpy as np
import tensorflow as tf

# -----------------------------------------------------

# calculate loss, assuming data is whitened
def np_loss( X ):
    return np.log( np.sqrt( np.mean( ( 182.0 * X )**2.0 ) ) ) / 0.6931471 # log(2.0)

# calculate loss, assuming data is not whitened
def np_loss_now( X ):
    return np.log( np.sqrt( np.mean( X**2.0 ) ) ) / 0.6931471 # log(2.0)

# ================================================================

def np_total_loss( sess, output_hat, inputs, input_ph, outputs, output_ph, BATCHSIZE ):

    DS = inputs.shape[0]
    DS = int(int(DS/BATCHSIZE)*BATCHSIZE)

    ohs = np.zeros((DS,1))

    # iterate over all data in BATCHSIZE -sized batches, and accumulate output answers
    for ind in range( 0,DS,BATCHSIZE ):
        oh = sess.run( output_hat, feed_dict={input_ph:inputs[ind:ind+BATCHSIZE,:,:,:], output_ph:outputs[ind:ind+BATCHSIZE,0:1]} )
        ohs[ind:ind+BATCHSIZE,0:1] = oh

    l2rmse_loss = np_loss( ohs - outputs[0:DS,0:1] )
    l1_loss = np.mean( np.abs( ohs - outputs[0:DS,0:1] ) )

    print "L1: %4f\tLOSS: %.4f" % ( l1_loss, l2rmse_loss )

    return l1_loss, l2rmse_loss, ohs

