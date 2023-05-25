import sys
import numpy as np
import yaml
from sompz import NoiseSOM as ns
import h5py

if len(sys.argv) == 1:
    cfgfile = 'y3_sompz.cfg'
else:
    cfgfile = sys.argv[1]

with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)

# Read variables from config file
output_path = cfg['out_dir']
bands = cfg['wide_bands']
bands_label = cfg['wide_bands_label']
bands_err_label = cfg['wide_bands_err_label']
som_dim = cfg['wide_som_dim']
wide_file = cfg['wide_file']
wide_h5_path = cfg['wide_h5_path']
no_shear = cfg['shear_types'][0]

# Load data
with h5py.File(wide_file, 'r') as f:
    if cfgfile == 'y3_sompz.cfg':
        selection = f['index']['select']
        total_length = len(selection)

        fluxes_d = np.zeros((total_length, len(bands)))
        fluxerrs_d = np.zeros((total_length, len(bands)))

        for i, band in enumerate(bands):
            print(i, band)
            fluxes_d[:, i] = f[wide_h5_path + no_shear + bands_label + band][...][selection]
            fluxerrs_d[:, i] = f[wide_h5_path + no_shear + bands_err_label + band][...][selection]
    else:
        selection = f[wide_h5_path + no_shear + bands_label + 'i']
        total_length = len(selection)

        fluxes_d = np.zeros((total_length, len(bands)))
        fluxerrs_d = np.zeros((total_length, len(bands)))

        for i, band in enumerate(bands):
            print(i, band)
            fluxes_d[:, i] = f[wide_h5_path + no_shear + bands_label + band][...]
            fluxerrs_d[:, i] = f[wide_h5_path + no_shear + bands_err_label + band][...]


# Train the SOM with this set (takes a few hours on laptop!)
nTrain = cfg['nwide_train']

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
np.save(f'{output_path}/som_wide_{som_dim}_{som_dim}_1e7.npy', som.weights)
