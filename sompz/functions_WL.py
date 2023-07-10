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

    try:
        balrog_data['cell_deep'] = pd.read_pickle(deep_cells_assignment_balrog_file)
        balrog_data['cell_wide_unsheared'] = pd.read_pickle(wide_cells_assignment_balrog_file)
    except:
        balrog_data['cell_deep'] = pd.read_csv(deep_cells_assignment_balrog_file, header=None, dtype=np.int32)
        balrog_data['cell_wide_unsheared'] = pd.read_csv(wide_cells_assignment_balrog_file, header=None, dtype=np.int32)

    return balrog_data

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
    cosmos = pd.concat(
        [cosmos, pd.DataFrame(-1 * np.ones((len(cosmos), len(zpdfcols))), columns=zpdfcols, index=cosmos.index)],
        axis=1)
    cosmos.loc[is_match, zpdfcols] = cosmos_z.iloc[idx[is_match], zpdfcols_indices].values

    cosmos.loc[is_match, 'LAIGLE_ID'] = cosmos_z.iloc[idx[is_match], cosmos_z.columns.get_loc('ID')].values
    ids, counts = np.unique(cosmos.loc[is_match, 'LAIGLE_ID'], return_counts=True)

    print('n duplicated Laigle', len(counts[counts > 1]))

    print("all cosmos deep: ", len(cosmos['BDF_MAG_DERED_CALIB_R']))
    print("matched cosmos deep: ", len(cosmos['BDF_MAG_DERED_CALIB_R'].loc[is_match]))
    print("unmatched cosmos deep: ", len(cosmos['BDF_MAG_DERED_CALIB_R'][cosmos['Z'] == -1]))

    cosmos = cosmos[cosmos['Z'] != -1].copy()

    return cosmos


def build_wide_df(wide_field_file, wide_data_assignment_df):
    with h5py.File(wide_field_file, 'r') as f:  # this is the master catalog

        # Wide Data
        try:
            wide_data_assignment_df['coadd_object_id'] = np.array(f['/mdet/noshear/uid'][:])  # [select_metacal]
            print("read coadd_object_id done")

            wide_data_assignment_df['unsheared/flux_g'] = np.array(
                f['/mdet/noshear/pgauss_band_flux_g'][:])  # [select_metacal]
            wide_data_assignment_df['unsheared/flux_i'] = np.array(
                f['/mdet/noshear/pgauss_band_flux_i'][:])  # [select_metacal]
            wide_data_assignment_df['unsheared/flux_r'] = np.array(
                f['/mdet/noshear/pgauss_band_flux_r'][:])  # [select_metacal]
            wide_data_assignment_df['unsheared/flux_z'] = np.array(
                f['/mdet/noshear/pgauss_band_flux_z'][:])  # [select_metacal]
            print("read unsheared/fluxes done")

            wide_data_assignment_df["unsheared/snr"] = np.array(f['/mdet/noshear/gauss_s2n'][:])
            print("read unsheared/snr done")
            wide_data_assignment_df["unsheared/size_ratio"] = np.array(f['/mdet/noshear/gauss_T_ratio'][:])
            print("read unsheared/T done")

            gauss_g_cov_1_1 = np.array(f['/mdet/noshear/gauss_g_cov_1_1'][:])
            gauss_g_cov_2_2 = np.array(f['/mdet/noshear/gauss_g_cov_2_2'][:])
            weights = 1 / (0.17 * 2 + 0.5 * (gauss_g_cov_1_1 + gauss_g_cov_2_2))

            wide_data_assignment_df['unsheared/weight'] = weights
            print("read unsheared/weight done")


        except:
            select_metacal = f['index']['select']
            print("read select metacal done")

            wide_data_assignment_df['coadd_object_id'] = np.array(f['catalog/metacal/unsheared/coadd_object_id'][:])[
                select_metacal]
            print("read coadd_object_id done")

            # wide_data_assignment_df['unsheared/flux_i']=np.array(f['catalog/metacal/unsheared/flux_i'][:])[select_metacal]
            # wide_data_assignment_df['unsheared/flux_r']=np.array(f['catalog/metacal/unsheared/flux_r'][:])[select_metacal]
            # wide_data_assignment_df['unsheared/flux_z']=np.array(f['catalog/metacal/unsheared/flux_z'][:])[select_metacal]

            wide_data_assignment_df['unsheared/T'] = np.array(f['catalog/metacal/unsheared/T'][:])[select_metacal]
            print("read unsheared/T done")

            wide_data_assignment_df['unsheared/snr'] = np.array(f['catalog/metacal/unsheared/snr'][:])[select_metacal]
            print("read unsheared/snr done")

            #         wide_data_assignment_df['unsheared/R11'] = np.array(f['catalog/metacal/unsheared/R11'][:])[select_metacal]
            #         print("read unsheared/R11 done")

            #         wide_data_assignment_df['unsheared/R22'] = np.array(f['catalog/metacal/unsheared/R22'][:])[select_metacal]
            #         print("read unsheared/R22 done")

            wide_data_assignment_df['unsheared/weight'] = np.array(f['catalog/metacal/unsheared/weight'][:])[
                select_metacal]
            print("read unsheared/weight done")

    return wide_data_assignment_df


def bin_assignment_spec(spec_data, deep_som_size, wide_som_size, bin_edges):
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


# def calculate_wide_overlap_weight(unsheared_R11, unsheared_R22, unsheared_weight):
#     wide_overlap_weight = np.ones(len(unsheared_weight))
#     wide_overlap_weight *= np.array(unsheared_R11) + np.array(unsheared_R22)
#     wide_overlap_weight *= np.array(unsheared_weight)

#     return wide_overlap_weight


def tomo_bins_wide_2d(tomo_bins_wide_dict):
    tomo_bins_wide = tomo_bins_wide_dict.copy()
    for k in tomo_bins_wide:
        if tomo_bins_wide[k].ndim == 1:
            tomo_bins_wide[k] = np.column_stack((tomo_bins_wide[k], np.ones(len(tomo_bins_wide[k]))))
        renorm = 1. / np.average(tomo_bins_wide[k][:, 1])
        tomo_bins_wide[k][:, 1] *= renorm  # renormalize so the mean weight is 1; important for bin conditioning
    return tomo_bins_wide
