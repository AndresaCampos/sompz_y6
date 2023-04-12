import numpy as np
from sompz import NoiseSOM as ns
import h5py

output_path = '/global/cscratch1/sd/acampos/sompz_y6/output_y6_data_preliminary'
catname = '/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5a/metadetect_desdmv5a_cutsv5.h5'

with h5py.File(catname, 'r') as f:
    ind_mcal = f['mdet']['noshear']['pgauss_band_flux_i']
    total_length = len(ind_mcal)

    bands = ['g','r', 'i', 'z']
    fluxes_d = np.zeros((total_length, len(bands)))
    fluxerrs_d = np.zeros((total_length, len(bands)))

    for i, band in enumerate(bands):
        print(i, band)
        fluxes_d[:, i] = f['/mdet/noshear/pgauss_band_flux_%s' % band][...]
        fluxerrs_d[:, i] = f['/mdet/noshear/pgauss_band_flux_err_%s' % band][...]

    # Train the SOM with this set (takes a few hours on laptop!)
nTrain = 10000000

# Scramble the order of the catalog for purposes of training
indices = np.random.choice(fluxes_d.shape[0], size=nTrain, replace=False)

# Some specifics of the SOM training
hh = ns.hFunc(nTrain, sigma=(30, 1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

# Now training the SOM
som = ns.NoiseSOM(metric, fluxes_d[indices, :], fluxerrs_d[indices, :],
                  learning=hh,
                  shape=(32, 32),
                  wrap=False, logF=True,
                  initialize='sample',
                  minError=0.02)

# And save the resultant weight matrix
np.save(f'{output_path}/som_wide_32_32_1e7_y6preliminary.npy', som.weights)
