import h5py
import numpy as np
import pandas as pd
import pickle
from astropy.coordinates import SkyCoord
from astropy import units as u


def build_balrog_df(balrog_file,
                    deep_cells_assignment_balrog_file,
                    wide_cells_assignment_balrog_file):
    with open(balrog_file, 'rb') as f:
        balrog_data = pickle.load(f, encoding='latin1')

    print("Length of balrog_data: " + str(len(balrog_data)))

    cells_deep_balrog = np.loadtxt(deep_cells_assignment_balrog_file, dtype=np.int32)
    cells_wide_balrog = np.loadtxt(wide_cells_assignment_balrog_file, dtype=np.int32)

    balrog_data['cell_deep_unsheared'] = cells_deep_balrog
    balrog_data['cell_wide_unsheared'] = cells_wide_balrog

    return balrog_data


def build_spec_df(cosmos_file, deep_data):
    cosmos_z = pd.read_hdf(cosmos_file)
    cosmos = deep_data[deep_data['FIELD'] == 'COSMOS'].copy()

    c = SkyCoord(ra=cosmos_z['ALPHA_J2000'].values * u.degree, dec=cosmos_z['DELTA_J2000'].values * u.degree)
    catalog = SkyCoord(ra=cosmos['RA'].values * u.degree, dec=cosmos['DEC'].values * u.degree)

    matchlim = 0.75
    idx, d2d, d3d = catalog.match_to_catalog_sky(c)
    is_match = d2d < matchlim * u.arcsec

    cosmos['Z'] = -1
    cosmos.loc[is_match, 'Z'] = cosmos_z.iloc[idx[is_match], cosmos_z.columns.get_loc('PHOTOZ')].values

    cosmos['Z2'] = -1
    cosmos.loc[is_match, 'Z2'] = cosmos_z.iloc[idx[is_match], cosmos_z.columns.get_loc('ZP_2')].values

    cosmos['2peak'] = -1
    cosmos.loc[is_match, '2peak'] = cosmos_z.iloc[idx[is_match], cosmos_z.columns.get_loc('SECOND_PEAK')].values

    zpdfcols = ["Z{:.2f}".format(s).replace(".", "_") for s in np.arange(0, 6.01, 0.01)]
    zpdfcols_indices = [cosmos_z.columns.get_loc(_) for _ in zpdfcols]
    cosmos[zpdfcols] = pd.DataFrame(-1 * np.ones((len(cosmos), len(zpdfcols))), columns=zpdfcols, index=cosmos.index)
    cosmos.loc[is_match, zpdfcols] = cosmos_z.iloc[idx[is_match], zpdfcols_indices].values

    cosmos.loc[is_match, 'LAIGLE_ID'] = cosmos_z.iloc[idx[is_match], cosmos_z.columns.get_loc('ID')].values
    ids, counts = np.unique(cosmos.loc[is_match, 'LAIGLE_ID'], return_counts=True)

    print('n duplicated Laigle', len(counts[counts > 1]))

    print("all cosmos deep: ", len(cosmos['BDF_MAG_DERED_CALIB_R']))
    print("matched cosmos deep: ", len(cosmos['BDF_MAG_DERED_CALIB_R'].loc[is_match]))
    print("unmatched cosmos deep: ", len(cosmos['BDF_MAG_DERED_CALIB_R'][cosmos['Z'] == -1]))

    cosmos = cosmos[cosmos['Z'] != -1].copy()

    return cosmos


def bin_assignment_spec(spec_data, deep_som_size, wide_som_size, inj_counts, inj_ids, bin_edges):
    # assign gals in redshift sample to bins
    spec_data['tomo_bin'] = pd.cut(spec_data['Z'], bin_edges, labels=[0, 1, 2, 3])

    ncells_with_spec_data = len(np.unique(spec_data['cell_wide_unsheared'].values))
    cell_bin_assignment = np.ones(wide_som_size, dtype=int) * -1
    cells_with_spec_data = np.unique(spec_data['cell_wide_unsheared'].values)

    groupby_obj_value_counts = spec_data.groupby('cell_wide_unsheared')['tomo_bin'].value_counts()

    for c in cells_with_spec_data:
        bin_assignment = groupby_obj_value_counts.loc[c].index[0]
        cell_bin_assignment[c] = bin_assignment

    # reformat bins into dict
    tomo_bins_wide = {}
    for i in range(4):
        tomo_bins_wide[i] = np.where(cell_bin_assignment == i)[0]

    return tomo_bins_wide


def build_wide_df(wide_field_file,
                  wide_cells_unsheared_file,
                  wide_cells_sheared1m_file,
                  wide_cells_sheared1p_file,
                  wide_cells_sheared2m_file,
                  wide_cells_sheared2p_file):
    cells_wide = np.loadtxt(wide_cells_unsheared_file, dtype=np.int32)
    cells_sheared_1m = np.loadtxt(wide_cells_sheared1m_file, dtype=np.int32)
    cells_sheared_1p = np.loadtxt(wide_cells_sheared1p_file, dtype=np.int32)
    cells_sheared_2m = np.loadtxt(wide_cells_sheared2m_file, dtype=np.int32)
    cells_sheared_2p = np.loadtxt(wide_cells_sheared2p_file, dtype=np.int32)

    df_dict_cell = {'cell_wide_unsheared': np.array(cells_wide),
                    'cell_wide_sheared_1m': np.array(cells_sheared_1m),
                    'cell_wide_sheared_1p': np.array(cells_sheared_1p),
                    'cell_wide_sheared_2m': np.array(cells_sheared_2m),
                    'cell_wide_sheared_2p': np.array(cells_sheared_2p)}
    df_dict = {}
    df_dict.update(df_dict_cell)
    wide_data_pass = pd.DataFrame(df_dict)

    with h5py.File(wide_field_file, 'r') as f:  # this is the master catalog

        # Wide Data
        select_metacal = f['index']['select']

        wide_data_pass['coadd_object_id'] = np.array(f['catalog/metacal/unsheared/coadd_object_id'][:])[select_metacal]
        # wide_data_pass['unsheared/flux_i'] = np.array(f['catalog/metacal/unsheared/flux_i'][:])[select_metacal]
        # wide_data_pass['unsheared/flux_r'] = np.array(f['catalog/metacal/unsheared/flux_r'][:])[select_metacal]
        # wide_data_pass['unsheared/flux_z'] = np.array(f['catalog/metacal/unsheared/flux_z'][:])[select_metacal]
        wide_data_pass['unsheared/T'] = np.array(f['catalog/metacal/unsheared/T'][:])[select_metacal]
        wide_data_pass['unsheared/snr'] = np.array(f['catalog/metacal/unsheared/snr'][:])[select_metacal]

        # ADD WEIGHTS
        wide_overlap_weight = np.ones(len(select_metacal))
        wide_overlap_weight *= np.array(f['catalog/metacal/unsheared/R11'][:])[select_metacal] + \
                               np.array(f['catalog/metacal/unsheared/R22'][:])[select_metacal]
        wide_overlap_weight *= np.array(f['catalog/metacal/unsheared/weight'][:])[select_metacal]

        wide_data_pass['overlap_weight'] = wide_overlap_weight

    print("done")

    return wide_data_pass
