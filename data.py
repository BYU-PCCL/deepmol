import numpy as np

def whiten(x):
    x = x - np.mean( x )
    x = x / np.sqrt( np.var( x ) )
    return x

def load_data():
    print "Loading data..."
    inputs_v = np.load( 'data/inputs_valence.npy' )[90:-90, 120:-120, :]
    inputs_v = inputs_v.transpose( [2,0,1] )  # put batch dimension first
    inputs_v = whiten(inputs_v)

    #inputs_c = np.load( 'data/inputs_core.npy' )
    #inputs_c = inputs_c.transpose( [2,0,1] )  # put batch dimension first
    #inputs_c = whiten( inputs_c )

    inputs = np.expand_dims( inputs_v, 3 ) # add "channel" dimension
    #inputs = np.zeros(( 4357, 398, 512, 2 ))
    #inputs[:,:,:,0] = inputs_v
    #inputs[:,:,:,1] = inputs_c

    inputs = inputs.astype('float32')

    outputs = np.load( 'data/outputs.npy' )
    outputs = outputs.transpose()
    outputs = whiten( outputs )

    # outputs: mean=-1470, var=33196.891

    return inputs, outputs

def load_test_train_inds():
    test_inds = np.load( 'data/test_inds.npy' )
    train_inds = np.load( 'data/train_inds.npy' )
    return test_inds, train_inds

def make_test_train_inds( cnt, pct=0.20 ):

    inds = np.random.permutation( range(cnt) )

    end_ind = int(cnt*pct)

    test_inds = inds[0:end_ind]
    train_inds = inds[end_ind:]

    np.save( 'data/test_inds.npy', test_inds )
    np.save( 'data/train_inds.npy', train_inds )
