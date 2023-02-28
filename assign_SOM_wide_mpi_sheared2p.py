import multiprocessing as mp
import os
import pickle
import numpy as np
import h5py
import NoiseSOM as ns
from mpi4py import MPI
from pickle import load


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

path_cats = '/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3'
som_type = 'wide'
shear = 'sheared_2p'
# This is just an example of wide field data file you can use (warning, you will have to do some selection on it)
#catname = '/project/projectdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5'
catname = '/global/cscratch1/sd/acampos/sompz_data/Y3_mastercat_03_31_20.h5'

bands = ['r', 'i', 'z']

if rank == 0:
    with h5py.File(catname,'r') as f:

        ind_mcal  = f['index']['select']
        total_length = len(ind_mcal)

        bands = ['r', 'i', 'z']
        fluxes = {}
        flux_errors = {}

        # fluxes_d = np.zeros((len_wide, len(bands)))
        # fluxerrs_d = np.zeros((len_wide, len(bands))) 

        for i, band in enumerate(bands):
            print(i, band)
            fluxes[band] = np.array_split(
                f[f'/catalog/metacal/{shear}/flux_{band}'][...][ind_mcal],
                nprocs
            )

            flux_errors[band] = np.array_split(
                f[f'/catalog/metacal/{shear}/flux_err_{band}'][...][ind_mcal],
                nprocs
            )
        
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
som_weights = np.load("%s/som_wide_32_32_1e7.npy" % path_cats, allow_pickle=True)
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
    filename = f'{path_cats}/{som_type}_{shear}/som_wide_32x32_1e7_assign_{som_type}_{shear}_{rank}_subsample_{index}.npz'
    if os.path.exists(filename) == False:
        print('Running')
        cells_test, _ = som.classify(fluxes_d[inds[index]], fluxerrs_d[inds[index]])
        np.savez(filename, cells=cells_test)
    else:
        print('File already exists')

for index in range(nsubsets):
    assign_som(index)