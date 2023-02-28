import pickle
import numpy as np
import NoiseSOM as ns
import h5py
    
path_cats = '/global/cscratch1/sd/acampos/SOM/cats_wide'
# This is just an example of wide field data file you can use (warning, you will have to do some selection on it)
#catname = '/project/projectdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5'
catname = '/global/cscratch1/sd/acampos/sompz_data/Y3_mastercat_03_31_20.h5'

with h5py.File(catname,'r') as f:
    
    ind_mcal  = f['index']['select']
    total_length = len(ind_mcal)

    bands = ['r', 'i', 'z']
    fluxes_d = np.zeros((total_length, len(bands)))
    fluxerrs_d = np.zeros((total_length, len(bands)))

    for i, band in enumerate(bands):
        print(i, band)
        fluxes_d[:, i] = f['/catalog/metacal/unsheared/flux_%s' % band][...][ind_mcal]
        fluxerrs_d[:, i] = f['/catalog/metacal/unsheared/flux_err_%s' % band][...][ind_mcal]        


# Train the SOM with this set (takes a few hours on laptop!)
nTrain = 10000000

# Scramble the order of the catalog for purposes of training
indices = np.random.choice(fluxes_d.shape[0], size=nTrain, replace=False)

# Some specifics of the SOM training
hh = ns.hFunc(nTrain, sigma=(30, 1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

# Now training the SOM (currently using a 48x48 SOM, but you can change it)
som = ns.NoiseSOM(metric, fluxes_d[indices, :], fluxerrs_d[indices, :],
                  learning=hh,
                  shape=(32, 32),
                  wrap=False, logF=True,
                  initialize='sample',
                  minError=0.02)

# And save the resultant weight matrix
np.save("%s/som_wide_32_32_1e7.npy" % path_cats, som.weights)
