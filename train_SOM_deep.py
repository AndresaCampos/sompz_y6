import pickle
import sys
import numpy as np
import yaml
from sompz import NoiseSOM as ns

if len(sys.argv) == 1:
    cfgfile = 'y3_sompz.cfg'
else:
    cfgfile = sys.argv[1]

with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)

# Read variables from config file
output_path = cfg['out_dir']
som_dim = cfg['deep_som_dim']
deep_file = cfg['deep_balrog_file']
bands = cfg['deep_bands']
bands_label = cfg['deep_bands_label']
bands_err_label = cfg['deep_bands_err_label']

# Load data
deep_data = pickle.load(open(deep_file, 'rb'), encoding='latin1')

# Create flux and flux_err vectors
len_deep = len(deep_data[bands_label+bands[0]])
fluxes_d = np.zeros((len_deep, len(bands)))
fluxerrs_d = np.zeros((len_deep, len(bands)))

for i, band in enumerate(bands):
    print(i, band)
    fluxes_d[:, i] = deep_data[bands_label+band]
    fluxerrs_d[:, i] = deep_data[bands_err_label+band]

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
                  shape=(som_dim, som_dim),
                  wrap=False, logF=True,
                  initialize='sample',
                  minError=0.02)

# And save the resultant weight matrix
np.save(f'{output_path}/som_deep_{som_dim}_{som_dim}.npy', som.weights)
