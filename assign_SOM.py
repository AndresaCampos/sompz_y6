import pickle
import numpy as np
from sompz import NoiseSOM as ns

# We load the same objects we used for training and we will find the SOM cell for each one, using 
# the SOM we already trained. You could use different objects too, this is just for illustration. 

som_type = 'deep'
data_type = 'balrog'
output_path = f'/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3/{som_type}_{data_type}'
# next, the path where you saved the output from train phase
deep_som_path = '/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3'
deep_som = 'som_deep_64_64.npy'
som_dim = 64

deep_file = '/global/cscratch1/sd/acampos/sompz_data/v0.50_andresa/deep_balrog.pkl'
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
som_weights = np.load(f'{deep_som_path}/{deep_som}', allow_pickle=True)
hh = ns.hFunc(nTrain, sigma=(30, 1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
som = ns.NoiseSOM(metric, None, None,
                  learning=hh,
                  shape=(som_dim, som_dim),
                  wrap=False, logF=True,
                  initialize=som_weights,
                  minError=0.02)
subsamp = 1

# Now we classify the objects into cells and save these cells
cells_test, dist_test = som.classify(fluxes_d[::subsamp, :], fluxerrs_d[::subsamp, :])
np.savez("%s/som_deep_64x64_assign.npz" % output_path, cells=cells_test, dist=dist_test)
