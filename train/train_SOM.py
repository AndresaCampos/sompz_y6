import pickle
import numpy as np
import NoiseSOM as ns

path_cats = 'your/favorite/path/here/'
# This is just an example of deep field data file you can use (warning, you will have to do some selection on it)
deep_file = '/global/cscratch1/sd/aamon/deep_ugriz.mof02_sn.jhk.ff04_c.jhk.ff02_052020_realerrors_May20calib.pkl'
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

# Now training the SOM (currently using a 48x48 SOM, but you can change it)
som = ns.NoiseSOM(metric, fluxes_d[indices, :], fluxerrs_d[indices, :],
                  learning=hh,
                  shape=(48, 48),
                  wrap=False, logF=True,
                  initialize='sample',
                  minError=0.02)

# And save the resultant weight matrix
np.save("%s/som_deep_48_48.npy" % path_cats, som.weights)
