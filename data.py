import numpy as np

def whiten(x):
    x = x - np.mean( x )
    x = x / np.sqrt( np.var( x ) )
    return x

def load_data():
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

    return inputs, outputs

def load_test_train_inds():
    test_inds = np.load( '/opt/wingated/mouli/test_inds.npy' )
    train_inds = np.load( '/opt/wingated/mouli/train_inds.npy' )
    return test_inds, train_inds

def make_test_train_inds( cnt, pct=0.20 ):

    inds = np.random.permutation( range(cnt) )

    end_ind = int(cnt*pct)

    test_inds = inds[0:end_ind]
    train_inds = inds[end_ind:]

    np.save( '/opt/wingated/mouli/test_inds.npy', test_inds )
    np.save( '/opt/wingated/mouli/train_inds.npy', train_inds )
