import pickle
import numpy as np
from sompz import NoiseSOM as ns

output_path = '/global/cscratch1/sd/acampos/SOM/cats_sompz'
deep_file = '/global/cscratch1/sd/acampos/sompz_data/v0.50/deep_balrog.pkl'

deep_data = pickle.load(open(deep_file, 'rb'), encoding='latin1')
len_deep = len(deep_data['BDF_FLUX_DERED_CALIB_U'])

bands = ['U', 'G', 'R', 'I', 'Z', 'J', 'H', 'K']
fluxes_d = np.zeros((len_deep, len(bands)))
fluxerrs_d = np.zeros((len_deep, len(bands)))

for i, band in enumerate(bands):
    print(i, band)
    fluxes_d[:, i] = deep_data['BDF_FLUX_DERED_CALIB_%s' % band]
    fluxerrs_d[:, i] = deep_data['BDF_FLUX_ERR_DERED_CALIB_%s' % band]

# Train the SOM with this set (takes a few hours on laptop!)
nTrain = fluxes_d.shape[0]

# Scramble the order of the catalog for purposes of training
indices = np.random.choice(fluxes_d.shape[0], size=nTrain, replace=False)

# Some specifics of the SOM training
hh = ns.hFunc(nTrain, sigma=(30, 1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

# Now training the SOM 
som = ns.NoiseSOM(metric, fluxes_d[indices, :], fluxerrs_d[indices, :],
                  learning=hh,
                  shape=(64, 64),
                  wrap=False, logF=True,
                  initialize='sample',
                  minError=0.02)

# And save the resultant weight matrix
np.save(f'{output_path}/som_deep_64_64.npy', som.weights)
