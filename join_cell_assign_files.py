import h5py
import numpy as np


# def join_cell_assign_files(path_cats, som_type, shear):
#     cells = []
#     for item in range(256):
#         for subsample in range(10):
#             cells_subsample = np.load(f'{path_cats}/{som_type}_{shear}/som_wide_32x32_1e7_assign_{som_type}_{shear}_{item}_subsample_{subsample}.npz')['cells']
#             cells.append(cells_subsample)

#     cells = np.concatenate(cells)
#     np.savetxt(f'cells_{som_type}_{shear}_newSOM_y3.txt', cells)
#     print(f'done with {shear}')

def join_cell_assign_files(path_cats, som_type, shear):
    cells = []
    for item in range(256):
        for subsample in range(10):
            cells_subsample = np.load(
                f'{path_cats}/{som_type}_{shear}/som_wide_32x32_1e7_assign_{som_type}_{shear}_{item}_subsample_{subsample}.npz')[
                'cells']
            cells.append(cells_subsample)

    cells = np.concatenate(cells)
    with h5py.File('sompz.hdf5', 'w', track_order=True) as f:
        f.create_dataset(f'catalog/sompz/{shear}/cell_wide', data=cells)
    print(f'done with {shear}')


def join_cell_assign_files_unsheared(path_cats, som_type):
    cells = []
    for item in range(128):
        for subsample in range(10):
            cells_subsample = \
            np.load(f'{path_cats}/{som_type}_unsheared/som_wide_32x32_1e7_assign_{item}_subsample_{subsample}.npz')[
                'cells']
            cells.append(cells_subsample)

    cells = np.concatenate(cells)
    with h5py.File('sompz.hdf5', 'w', track_order=True) as f:
        f.create_dataset(f'catalog/sompz/unsheared/cell_wide', data=cells)
    print(f'done with unsheared')


som_type = 'wide'
path_cats = '/global/cscratch1/sd/acampos/sompz/test/full_run_on_data/SOM/cats_Y3'

join_cell_assign_files(path_cats, som_type, 'sheared_1m')
join_cell_assign_files(path_cats, som_type, 'sheared_1p')
join_cell_assign_files(path_cats, som_type, 'sheared_2m')
join_cell_assign_files(path_cats, som_type, 'sheared_2p')
join_cell_assign_files_unsheared(path_cats, som_type)
