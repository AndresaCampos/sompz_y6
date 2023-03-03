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


