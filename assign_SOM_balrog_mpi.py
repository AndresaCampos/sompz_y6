import multiprocessing as mp
import os
import pickle
import numpy as np
import h5py
import NoiseSOM as ns
from mpi4py import MPI
import pandas as pd


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

path_out = '/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3/wide_balrog_230116'
path_wide = '/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3'
som_type = 'wide'
data_type = 'balrog'
shear = 'unsheared'

bands = ['r', 'i', 'z']

if rank == 0:
    # catname = '/global/cscratch1/sd/aamon/balrog/y3_balrog2_v1.2_merged_select2_bstarcut_matchflag1.5asec_snr_SR_corrected_uppersizecuts.h5'
    # df = pd.read_hdf(catname, mode='r')
    catname = '/global/cscratch1/sd/acampos/sompz_data/v0.50_andresa/deep_balrog.pkl'
    df = pd.read_pickle(catname)

    fluxes = {}
    flux_errors = {}

    # fluxes_d = np.zeros((len_wide, len(bands)))
    # fluxerrs_d = np.zeros((len_wide, len(bands))) 

    for i, band in enumerate(bands):
        print(i, band)
        fluxes[band] = np.array_split(
            df['unsheared/flux_%s' % band].values,
            nprocs
        )

        flux_errors[band] = np.array_split(
            df['unsheared/flux_err_%s' % band].values,
            nprocs
        )

        # fluxes_d[:, i] = f['/catalog/metacal/unsheared/flux_%s' % band][...][ind_mcal]
        # fluxerrs_d[:, i] = f['/catalog/metacal/unsheared/flux_err_%s' % band][...][ind_mcal]
            
    
    # fluxes = np.array_split(fluxes_d, nprocs)
    # flux_errors = np.array_split(fluxerrs_d, nprocs)
    
    
#     # determine the size of each sub-task
#     ave, res = divmod(len_wide, nprocs)
#     counts = [ave + 1 if p < res else ave for p in range(nprocs)]

#     # determine the starting and ending indices of each sub-task
#     starts = np.array([sum(counts[:p]) for p in range(nprocs)])
#     ends = np.array([sum(counts[:p+1]) for p in range(nprocs)])
#     slice_sizes = ends - starts

#     # converts data into a list of arrays
#     # data = []
#     fluxes = []
#     flux_errors = []
#     for p in range(nprocs):
#         # slices = [fluxes_d[starts[p]:ends[p]],fluxerrs_d[starts[p]:ends[p]]]
#         fluxes.append(fluxes_d[starts[p]:ends[p]])
#         flux_errors.append(fluxerrs_d[starts[p]:ends[p]])
    
else:
    # data = None
    fluxes = {b: None for b in bands}
    flux_errors = {b: None for b in bands}

# data = comm.scatter(data, root=0)

for i, band in enumerate(bands):
    fluxes[band] = comm.scatter(fluxes[band], root=0)
    flux_errors[band] = comm.scatter(flux_errors[band], root=0)
    
# prepare big data matrix
fluxes_d = np.zeros((fluxes[bands[0]].size, len(bands)))
fluxerrs_d = np.zeros((flux_errors[bands[0]].size, len(bands)))

for i, band in enumerate(bands):
    fluxes_d[:, i] = fluxes[band]
    fluxerrs_d[:, i] = flux_errors[band]


#print(rank, len(data),len(data[0]), len(data[0][0]))

# fluxes_d = data[0].copy()
# fluxerrs_d = data[1].copy()

#print(rank, len(fluxes_d[0]),len(fluxerrs_d[0]))

# Train the SOM with this set (takes a few hours on laptop!)
nTrain = fluxes_d.shape[0]

# Now, instead of training the SOM, we load the SOM we trained:
som_weights = np.load("%s/som_wide_32_32_1e7.npy" % path_wide, allow_pickle=True)
hh = ns.hFunc(nTrain, sigma=(30, 1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
som = ns.NoiseSOM(metric, None, None,
                  learning=hh,
                  shape=(32, 32),
                  wrap=False, logF=True,
                  initialize=som_weights,
                  minError=0.02)

nsubsets = 10

inds = np.array_split(np.arange(len(fluxes_d)), nsubsets)    

# This function checks whether you have already run that subset, and if not it runs the SOM classifier
def assign_som(index):
    print(f'Running rank {rank}, index {index}')
    filename = f'{path_out}/som_{som_type}_32x32_1e7_assign_{data_type}_{shear}_{rank}_subsample_{index}.npz'
    if os.path.exists(filename) == False:
        print('Running')
        cells_test, _ = som.classify(fluxes_d[inds[index]], fluxerrs_d[inds[index]])
        np.savez(filename, cells=cells_test)
    else:
        print('File already exists')

for index in range(nsubsets):
    assign_som(index)