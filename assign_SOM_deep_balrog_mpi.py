import os
import sys
import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI
from sompz import NoiseSOM as ns

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if len(sys.argv) == 1:
    cfgfile = 'y3_sompz.cfg'
else:
    cfgfile = sys.argv[1]

with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)

som_type = 'deep'
data_type = 'deep_balrog'

# Read variables from config file
output_path = cfg['out_dir']
som_deep = cfg['som_deep']
som_dim = cfg['deep_som_dim']
deep_balrog_file = cfg['deep_balrog_file']
bands = cfg['deep_bands']
bands_label = cfg['deep_bands_label']
bands_err_label = cfg['deep_bands_err_label']

# Load data
if rank == 0:
    df = pd.read_pickle(deep_balrog_file)

    fluxes = {}
    flux_errors = {}
    for i, band in enumerate(bands):
        print(i, band)
        fluxes[band] = np.array_split(
            df[bands_label + band].values,
            nprocs
        )
        flux_errors[band] = np.array_split(
            df[bands_err_label + band].values,
            nprocs
        )
    os.system(f'mkdir -p {output_path}/{som_type}_{data_type}')
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
som_weights = np.load(f'{output_path}/{som_deep}', allow_pickle=True)
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
