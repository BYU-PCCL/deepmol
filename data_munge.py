
import numpy as np
import h5py
import matplotlib.cm as cm

A = h5py.File('./qm7_2d_images_v73.mat')

valence_results = []
atomic_results = []
dirac_results = []
core_results = []

for i in range( len(A['rhoD']['rho']) ):
    o = A['rhoD']['rho'][i][0]
#    valence_results.append( A[o]['valence'] )
#    atomic_results.append( A[o]['atomic'] )
#    dirac_results.append( A[o]['dirac'] )
    core_results.append( A[o]['core'] )

# valence_rmat = np.stack( valence_results, axis=2 )
# np.save( './inputs_valence.npy', valence_rmat )

# atomic_rmat = np.stack( atomic_results, axis=2 )
# np.save( './inputs_atomic.npy', atomic_rmat )

# dirac_rmat = np.stack( dirac_results, axis=2 )
# np.save( './inputs_dirac.npy', dirac_rmat )

core_rmat = np.stack( core_results, axis=2 )
np.save( './inputs_core.npy', core_rmat )

#foo = scipy.io.loadmat('./energies.mat')
#np.save( './outputs.npy', foo['T'] )
