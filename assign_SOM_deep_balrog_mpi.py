import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from sompz import NoiseSOM as ns

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

som_type = 'deep'
data_type = 'balrog'
shear = 'unsheared'
output_path = f'/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3/{som_type}_{data_type}'
deep_som_path = '/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3'
deep_som = 'som_deep_64_64.npy'
som_dim = 64

# This is just an example of wide field data file you can use
catname = '/global/cscratch1/sd/acampos/sompz_data/v0.50_andresa/deep_balrog.pkl'

bands = ['U', 'G', 'R', 'I', 'Z', 'J', 'H', 'K']

if rank == 0:
    df = pd.read_pickle(catname)

    fluxes = {}
    flux_errors = {}

    for i, band in enumerate(bands):
        print(i, band)
        fluxes[band] = np.array_split(
            df['BDF_FLUX_DERED_CALIB_%s' % band].values,
            nprocs
        )

        flux_errors[band] = np.array_split(
            df['BDF_FLUX_ERR_DERED_CALIB_%s' % band].values,
            nprocs
        )

else:
    # data = None
    fluxes = {b: None for b in bands}
    flux_errors = {b: None for b in bands}

# scatter data
for i, band in enumerate(bands):
    fluxes[band] = comm.scatter(fluxes[band], root=0)
    flux_errors[band] = comm.scatter(flux_errors[band], root=0)

# prepare big data matrix
fluxes_d = np.zeros((fluxes[bands[0]].size, len(bands)))
fluxerrs_d = np.zeros((flux_errors[bands[0]].size, len(bands)))

for i, band in enumerate(bands):
    fluxes_d[:, i] = fluxes[band]
    fluxerrs_d[:, i] = flux_errors[band]

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

nsubsets = 10

inds = np.array_split(np.arange(len(fluxes_d)), nsubsets)


# This function checks whether you have already run that subset, and if not it runs the SOM classifier
def assign_som(ind):
    print(f'Running rank {rank}, index {ind}')
    filename = f'{output_path}/{som_type}_{data_type}/som_{som_type}_assign_{data_type}_{rank}_subsample_{ind}.npz'
    if not os.path.exists(filename):
        print('Running')
        cells_test, _ = som.classify(fluxes_d[inds[ind]], fluxerrs_d[inds[ind]])
        np.savez(filename, cells=cells_test)
    else:
        print('File already exists')


for index in range(nsubsets):
    assign_som(index)
