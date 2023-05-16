import os
import sys
import numpy as np
import h5py
import yaml
from sompz import NoiseSOM as ns
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if len(sys.argv) == 1:
    cfgfile = 'y3_sompz.cfg'
else:
    cfgfile = sys.argv[1]

with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)

som_type = 'wide'
shear = 'unsheared'

som_wide = cfg['som_wide']
som_dim = cfg['wide_som_dim']
output_path = cfg['out_dir']
wide_h5_path = cfg['wide_h5_path']
bands_label = cfg['wide_bands_label']
bands_err_label = cfg['wide_bands_err_label']

# This is just an example of wide field data file you can use
wide_file = cfg['wide_file']

bands = cfg['wide_bands']

if rank == 0:
    with h5py.File(wide_file, 'r') as f:
        ind_mcal = f['index']['select']
        total_length = len(ind_mcal)

        fluxes = {}
        flux_errors = {}
        for i, band in enumerate(bands):
            print(i, band)
            fluxes[band] = np.array_split(
                f[wide_h5_path + shear + bands_label + band][...][ind_mcal],
                nprocs
            )
            flux_errors[band] = np.array_split(
                f[wide_h5_path + shear + bands_err_label + band][...][ind_mcal],
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

nTrain = fluxes_d.shape[0]

# Now, instead of training the SOM, we load the SOM we trained:
som_weights = np.load(f'{output_path}/{som_wide}', allow_pickle=True)
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
    filename = f'{output_path}/{som_type}_{shear}/som_wide_32_32_1e7_assign_{som_type}_{shear}_{rank}_subsample_{ind}.npz'
    if not os.path.exists(filename):
        print('Running')
        cells_test, _ = som.classify(fluxes_d[inds[ind]], fluxerrs_d[inds[ind]])
        np.savez(filename, cells=cells_test)
    else:
        print('File already exists')


for index in range(nsubsets):
    assign_som(index)
