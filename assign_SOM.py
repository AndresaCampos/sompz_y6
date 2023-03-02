import pickle
import numpy as np
from sompz import NoiseSOM as ns

# We load the same objects we used for training and we will find the SOM cell for each one, using 
# the SOM we already trained. You could use different objects too, this is just for illustration. 

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

# Now, instead of training the SOM, we load the SOM we trained:
som_weights = np.load("%s/som_deep_48_48.npy" % path_cats, allow_pickle=True)
hh = ns.hFunc(nTrain, sigma=(30, 1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
som = ns.NoiseSOM(metric, None, None,
                  learning=hh,
                  shape=(48, 48),
                  wrap=False, logF=True,
                  initialize=som_weights,
                  minError=0.02)
subsamp = 1

# Now we classify the objects into cells and save these cells
cells_test, dist_test = som.classify(fluxes_d[::subsamp, :], fluxerrs_d[::subsamp, :])
np.savez("%s/som_deep_48x48_assign.npz" % path_cats, cells=cells_test, dist=dist_test)
