import h5py
import pickle
import numpy as np


def join_cell_assign_files(path_cats, som_type, shear, run_name):
    cells = []
    for item in range(256):
        for subsample in range(10):
            if som_type == 'wide':
                cells_subsample = np.load(
                    f'{path_cats}/{som_type}_{shear}/som_wide_32_32_1e7_assign_{shear}_{item}_subsample_{subsample}.npz')[
                    'cells']
            else:
                cells_subsample = np.load(
                    f'{path_cats}/{som_type}_{shear}/som_deep_assign_{shear}_{item}_subsample_{subsample}.npz')[
                    'cells']
            cells.append(cells_subsample)
    cells = np.concatenate(cells)
    
    with open(f"path_cats/cells_{som_type}_{shear}_{run_name}.pkl", "wb") as output_file:
        pickle.dump(cells, output_file)
    
    # with h5py.File('sompz.hdf5', 'w', track_order=True) as f:
    #     f.create_dataset(f'catalog/sompz/{shear}/cell_wide', data=cells)
    # print(f'done with {shear}')


path_cats = '/global/cfs/projectdirs/des/acampos/sompz_output/y6_data_preliminary'
run_name = 'y6preliminary'

# join_cell_assign_files(path_cats, som_type, 'sheared_1m')
# join_cell_assign_files(path_cats, som_type, 'sheared_1p')
# join_cell_assign_files(path_cats, som_type, 'sheared_2m')
# join_cell_assign_files(path_cats, som_type, 'sheared_2p')

join_cell_assign_files(path_cats, 'deep', 'deep_balrog',run_name)
join_cell_assign_files(path_cats, 'wide', 'deep_balrog',run_name)
