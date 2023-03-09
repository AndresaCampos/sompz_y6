import numpy as np
import fitsio
import twopoint
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.signal import savgol_filter


def flux2mag(flux, zero_pt=30):
    """Converts fluxes to Magnitudes"""
    return zero_pt - 2.5 * np.log10(flux)


def smooth_response_weight(snr, size_ratio, file):
    snmin = 10
    snmax = 300
    sizemin = 0.5
    sizemax = 5
    steps = 20
    r = np.genfromtxt(file)

    def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps
        x = np.maximum(x, xmin)
        x = np.minimum(x, xmax)
        y = np.maximum(y, ymin)
        y = np.minimum(y, ymax)
        logstepx = np.log10(xmax / xmin) / xsteps
        logstepy = np.log10(ymax / ymin) / ysteps
        indexx = (np.log10(x / xmin) / logstepx).astype(int)
        indexy = (np.log10(y / ymin) / logstepy).astype(int)
        indexx = np.minimum(indexx, xsteps - 1)
        indexy = np.minimum(indexy, ysteps - 1)
        return indexx, indexy

    def apply_loggrid(x, y, grid, xmin=snmin, xmax=snmax, xsteps=steps, ymin=sizemin, ymax=sizemax, ysteps=steps):
        indexx, indexy = assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps)
        res = np.zeros(len(x))
        res = grid[indexx, indexy]
        return res

    smoothresponse = apply_loggrid(snr, size_ratio, r)
    return smoothresponse


def calculate_weights(smooth_response_file, snr, size_ratio, injection_counts, unsheared_weight, data_len):
    smooth_response = smooth_response_weight(snr, size_ratio, smooth_response_file)
    w = np.ones(data_len)
    w *= smooth_response / 2
    w /= injection_counts
    w *= unsheared_weight
    return w


def calculate_pcchat(deep_som_size, wide_som_size, cell_deep_assign, cell_wide_assign, overlap_weight):
    pcchat_num = np.zeros((deep_som_size, wide_som_size))
    np.add.at(pcchat_num,
              (cell_deep_assign, cell_wide_assign),
              overlap_weight)

    pcchat_denom = pcchat_num.sum(axis=0)
    pcchat = pcchat_num / pcchat_denom[None]

    # any nonfinite in pcchat are to be treated as 0 probabilty
    pcchat = np.where(np.isfinite(pcchat), pcchat, 0)

    return pcchat


def get_cell_weights(data, overlap_weighted, key):
    """Given data, get cell weights and indices

    Parameters
    ----------
    data :  Dataframe we extract parameters from
    overlap_weighted : If True, use mean overlap weights of cells.
    key :   Which key we are grabbing

    Returns
    -------
    cells :         The names of the cells
    cell_weights :  The fractions of the cells
    """
    if overlap_weighted:
        cws = data.groupby(key)['overlap_weight'].sum()
    else:
        cws = data.groupby(key).size()

    cells = cws.index.values.astype(int)
    cws = cws / cws.sum()

    cell_weights = cws.values
    return cells, cell_weights


def get_cell_weights_wide(data, overlap_weighted_pchat, cell_key='cell_wide', force_assignment=False, **kwargs):
    """Given data, get cell weights p(chat) and indices from wide SOM

    Parameters
    ----------
    data             : Dataframe we extract parameters from
    overlap_weighted_pchat : If True, use mean overlap weights of wide cells in p(chat)
    cell_key         : Which key we are grabbing. Default: cell_wide
    force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True

    Returns
    -------
    cells        :  The names of the cells
    cell_weights :  The fractions of the cells
    """
    # if force_assignment:
    #     data[cell_key] = self.assign_wide(data, **kwargs)
    return get_cell_weights(data, overlap_weighted_pchat, cell_key)


def histogram_from_fullpz(df, key, overlap_weighted, bin_edges, full_pz_end=6.00, full_pz_npts=601):
    """Preserve bins from Laigle"""
    dz_laigle = full_pz_end / (full_pz_npts - 1)
    condition = np.sum(~np.equal(bin_edges, np.arange(0 - dz_laigle / 2.,
                                                      full_pz_end + dz_laigle,
                                                      dz_laigle)))
    assert condition == 0

    single_cell_hists = np.zeros((len(df), len(key)))

    overlap_weights = np.ones(len(df))
    if overlap_weighted:
        overlap_weights = df['overlap_weight'].values

    single_cell_hists[:, :] = df[key].values

    # normalize sompz p(z) to have area 1
    dz = 0.01
    area = np.sum(single_cell_hists, axis=1) * dz
    area[area == 0] = 1  # some galaxies have pz with only one non-zero point. set these galaxies' histograms to have
    # area 1
    area = area.reshape(area.shape[0], 1)
    single_cell_hists = single_cell_hists / area

    # response weight normalized p(z)
    single_cell_hists = np.multiply(overlap_weights, single_cell_hists.transpose()).transpose()

    # sum individual galaxy p(z) to single cell p(z)
    hist = np.sum(single_cell_hists, axis=0)

    # renormalize p(z|c)
    area = np.sum(hist) * dz
    hist = hist / area

    return hist


def get_deep_histograms(data, deep_data, key, cells, overlap_weighted_pzc, bins, overlap_key='overlap_weight',
                        deep_som_size=64 * 64, deep_map_shape=(64 * 64,), interpolate_kwargs={}):
    """Return individual deep histograms for each cell. Can interpolate for empty cells.

    Parameters
    ----------
    deep_data             : cosmos data used here for Y3
    key                   : Parameter to extract from dataframe
    cells                 : A list of deep cells to return sample from, or a single int.
    overlap_weighted_pzc  : Use overlap_weights in p(z|c) histogram if True. Also required if you want to bin conditionalize
    overlap_key           : column name for the overlap weights in the dataframe, default to 'overlap_weight'
    bins                  : Bins we histogram the values into
    interpolate_kwargs    : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away.
    The two kwargs are    : 'scale_length' and 'max_length'
    Returns
    -------
    hists : a histogram of the values from self.data[key] for each deep cell
    """

    if len(interpolate_kwargs) > 0:
        cells_keep = cells
        cells = np.arange(deep_som_size)
    else:
        cells_keep = cells

    hists = []
    missing_cells = []
    populated_cells = []
    for ci, c in enumerate(cells):
        try:
            df = deep_data.groupby('cell_deep').get_group(c)
            if type(key) is str:
                z = df[key].values
                if overlap_weighted_pzc:
                    # print("WARNING: You are using a deprecated point estimate Z. No overlap weighting enabled.
                    # You're on your own now.")#suppress
                    weights = df[overlap_key].values
                else:
                    weights = np.ones(len(z))
                hist = np.histogram(z, bins, weights=weights, density=True)[
                    0]  # make weighted histogram by overlap weights
                populated_cells.append([ci, c])
            elif type(key) is list:
                # use full p(z)
                assert (bins is not None)
                hist = histogram_from_fullpz(df, key, overlap_weighted=overlap_weighted_pzc, bin_edges=bins)
            hists.append(hist)
        except KeyError as e:
            missing_cells.append([ci, c])
            hists.append(np.zeros(len(bins) - 1))
    hists = np.array(hists)

    if len(interpolate_kwargs) > 0:
        # print('Interpolating {0} missing histograms'.format(len(missing_cells)))
        missing_cells = np.array(missing_cells)
        populated_cells = np.array(populated_cells)
        hist_conds = np.isin(cells, populated_cells[:, 1]) & np.all(np.isfinite(hists), axis=1)
        for ci, c in missing_cells:
            if c not in cells_keep:
                # don't worry about interpolating cells we won't use anyways
                continue

            central_index = np.zeros(len(deep_map_shape), dtype=int)
            # unravel_index(c, deep_map_shape, central_index)  # fills central_index
            cND = np.zeros(len(deep_map_shape), dtype=int)
            weight_map = np.zeros(deep_som_size)
            # gaussian_rbf(weight_map, central_index, cND, deep_map_shape, **interpolate_kwargs)  # fills weight_map
            hists[ci] = np.sum(hists[hist_conds] * (weight_map[hist_conds] / weight_map[hist_conds].sum())[:, None],
                               axis=0)

        # purge hists back to the ones we care about
        hists = hists[cells_keep]

    return hists


def histogram(data, deep_data, key, cells, cell_weights, pcchat, overlap_weighted_pzc, deep_som_size=64 * 64, bins=None,
              individual_chat=False, interpolate_kwargs={}):
    """Return histogram from values that live in specified wide cells by querying deep cells that contribute

    Parameters
    ----------
    key                  : Parameter(s) to extract from dataframe
    cells                : A list of wide cells to return sample from, or a single int.
    cell_weights         : How much we weight each wide cell. This is the array p(chat | sample)
    overlap_weighted_pzc : Weight contribution of galaxies within c by overlap_weight, if True. Weighting for p(c|chat) is done using stored transfer matrix.
    bins                 : Bins we histogram the values into
    individual_chat      : If True, compute p(z|chat) for each individual cell in cells. If False, compute a single p(z|{chat}) for all cells.
    interpolate_kwargs   : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away. The two kwargs are: 'scale_length' and 'max_length'

    Returns
    -------
    hist : a histogram of the values from self.data[key]

    Notes
    -----
    This method tries to marginalize wide assignments into what deep assignments it has

    """
    # get sample, p(z|c)
    all_cells = np.arange(deep_som_size)
    hists_deep = get_deep_histograms(data, deep_data, key=key, cells=all_cells, overlap_weighted_pzc=overlap_weighted_pzc,
                                     bins=bins, interpolate_kwargs=interpolate_kwargs)
    if individual_chat:  # then compute p(z|chat) for each individual cell in cells and return histograms
        hists = []
        for i, (cell, cell_weight) in enumerate(zip(cells, cell_weights)):
            # p(c|chat,s)p(chat|s) = p(c,chat|s)
            possible_weights = pcchat[:, [cell]] * np.array([cell_weight])[None]  # (n_deep_cells, 1)
            # sum_chat p(c,chat|s) = p(c|s)
            weights = np.sum(possible_weights, axis=-1)
            conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
            # sum_c p(z|c) p(c|s) = p(z|s)
            hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

            dx = np.diff(bins)
            normalization = np.sum(dx * hist)
            if normalization != 0:
                hist = hist / normalization
            hists.append(hist)
        return hists
    else:  # compute p(z|{chat}) and return histogram
        # p(c|chat,s)p(chat|s) = p(c,chat|s)
        possible_weights = pcchat[:, cells] * cell_weights[None]  # (n_deep_cells, n_cells)
        # sum_chat p(c,chat|s) = p(c|s)
        weights = np.sum(possible_weights, axis=-1)
        conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
        # sum_c p(z|c) p(c|s) = p(z|s)
        hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

        dx = np.diff(bins)
        normalization = np.sum(dx * hist)
        if normalization != 0:
            hist = hist / normalization
        return hist


def redshift_distributions_wide(data,
                                deep_data,
                                overlap_weighted_pchat,
                                overlap_weighted_pzc,
                                bins,
                                pcchat,
                                tomo_bins={},
                                key='Z',
                                force_assignment=True,
                                interpolate_kwargs={}, **kwargs):
    """Returns redshift distribution for sample

    Parameters
    ----------
    data :  Data sample of interest with wide data
    deep_data: cosmos data
    overlap_weighted_pchat  : If True, use overlap weights for p(chat)
    overlap_weighted_pzc : If True, use overlap weights for p(z|c)
                Note that whether p(c|chat) is overlap weighted depends on how you built pcchat earlier.
    bins :      bin edges for redshift distributions data[key]
    tomo_bins : Which cells belong to which tomographic bins. First column is
                cell id, second column is an additional reweighting of galaxies in cell.
                If nothing is passed in, then we by default just use all cells
    key :       redshift key
    force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
    interpolate_kwargs : arguments to pass in for performing interpolation
    between cells for redshift hists using a 2d gaussian of sigma
    scale_length out to max_length cells away. The two kwargs are:
    'scale_length' and 'max_length'

    Returns
    -------
    hists : Either a single array (if no tomo_bins) or multiple arrays

    """
    if len(tomo_bins) == 0:
        cells, cell_weights = get_cell_weights_wide(data, overlap_weighted_pchat=overlap_weighted_pchat,
                                                    force_assignment=force_assignment, **kwargs)
        if cells.size == 0:
            hist = np.zeros(len(bins) - 1)
        else:
            hist = histogram(data, deep_data, key=key, cells=cells, cell_weights=cell_weights,
                             overlap_weighted_pzc=overlap_weighted_pzc, bins=bins,
                             interpolate_kwargs=interpolate_kwargs)
        return hist
    else:
        cells, cell_weights = get_cell_weights_wide(data, overlap_weighted_pchat,
                                                    force_assignment=force_assignment, **kwargs)
        cellsort = np.argsort(cells)
        cells = cells[cellsort]
        cell_weights = cell_weights[cellsort]

        # break up hists into the different bins
        hists = []
        for tomo_key in tomo_bins:
            cells_use = tomo_bins[tomo_key][:, 0]
            cells_binweights = tomo_bins[tomo_key][:, 1]
            cells_conds = np.searchsorted(cells, cells_use, side='left')
            if len(cells_conds) == 0:
                hist = np.zeros(len(bins) - 1)
            else:
                hist = histogram(data, deep_data, key=key, cells=cells[cells_conds],
                                 cell_weights=cell_weights[cells_conds] * cells_binweights, pcchat=pcchat,
                                 overlap_weighted_pzc=overlap_weighted_pzc, bins=bins,
                                 interpolate_kwargs=interpolate_kwargs)
            hists.append(hist)
        hists = np.array(hists)
        return hists


def plot_nz(hists, zbins, outfile, xlimits=(0, 2), ylimits=(0, 3.25)):
    plt.figure(figsize=(16., 9.))
    for i in range(len(hists)):
        plt.plot((zbins[1:] + zbins[:-1]) / 2., hists[i], label='bin ' + str(i))
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$p(z)$')
    plt.legend()
    plt.title('n(z)')
    plt.savefig(outfile)
    plt.close()


def nz_bin_conditioned(wfdata, cosmos, overlap_weighted_pchat, overlap_weighted_pzc, tomo_cells, zbins, pcchat,
                       cell_wide_key='cell_wide', zkey='Z'):
    """ Function to obtain p(z|bin,s): the redshift distribution of a tomographic bin
    including the tomographic selection effect in p(z|chat).

    Implementation note:
    This is going to sneak the bin conditionalization into the overlap weights, and then divide them back out.
    This is a simple way of achieving to not completely lose cells c that contribute to p(c|chat) but don't have a z in b.
    Not the cleanest code written by a human.

        Parameters
        ----------
        wfdata : Wide field data
        overlap_weighted_pchat : If True, weight chat by the sum of overlap weights, not number of galaxies, in wide field data.
        tomo_cells : Which cells belong to this tomographic bin. First column is
                     cell id, second column is an additional reweighting of galaxies in that cell.
        zbins : redshift bin edges.
        cell_wide_key : key for wide SOM cell id information in spec_data.
        cell_deep_key : key for wide SOM cell id information in spec_data.
        #cells : A list of deep cells to return sample from, or a single int.
        #cell_weights : How much we weight each wide cell. This is the array p(c | sample)
    """
    print('full redshift sample:', len(cosmos))
    # print('cell_wide_key: ', cell_wide_key)
    # print('self.data[cell_wide_key].shape', self.data[cell_wide_key].shape)
    # print('tomo_cells', tomo_cells)
    bl = len(cosmos[cosmos['cell_wide_unsheared'].isin(tomo_cells[:, 0])])

    print('subset of redshift sample in bin:', bl)

    f = 1.e9  # how much more we weight the redshift of a galaxy that's in the right bin

    stored_overlap_weight = cosmos['overlap_weight'].copy()  # save for later

    if not overlap_weighted_pzc:  # we need to use it, but you don't want to
        cosmos['overlap_weight'] = np.ones(len(cosmos))

    cosmos.loc[cosmos['cell_wide_unsheared'].isin(tomo_cells[:, 0]), 'overlap_weight'] *= f

    nz = redshift_distributions_wide(data=wfdata, deep_data=cosmos, overlap_weighted_pchat=overlap_weighted_pchat,
                                     overlap_weighted_pzc=True,
                                     bins=zbins, pcchat=pcchat, tomo_bins={"mybin": tomo_cells}, key=zkey,
                                     force_assignment=False, cell_key=cell_wide_key)

    cosmos['overlap_weight'] = stored_overlap_weight.copy()  # open jar

    return nz[0]


def plot_nz_overlap(list_of_nz, list_of_labels, outdir):
    colors = 'rgbcmyk'
    plt.figure(figsize=(16, 9))

    ### nz_all is an array: (n_tomobins, n_zbins)
    for j, (nz_all, label) in enumerate(zip(list_of_nz, list_of_labels)):
        nz_all_overlap_somM = np.einsum('mz,nz->mn', nz_all, nz_all)
        nz_all_overlap_var_somM = np.einsum('m,n->mn', np.sqrt(np.diag(nz_all_overlap_somM)),
                                            np.sqrt(np.diag(nz_all_overlap_somM)))
        nz_all_overlap_somM /= nz_all_overlap_var_somM

        ### Plot the overlap matrix as a series of 4 lines showing the columns of the matrix.
        [plt.plot(range(1 + i, 5), x[i:], 's--', color=colors[j]) for i, x in enumerate(nz_all_overlap_somM)]
        [plt.plot(range(1 + i, 5), x[i:], 's--', color=colors[j]) for i, x in enumerate(nz_all_overlap_somM)]
        plt.ylabel('N(z) Overlap')
        plt.legend(loc=0, fontsize=15)
        plt.xticks(np.arange(1, 5, 1))
    plt.savefig(outdir + 'Y3_nz_overlap_plot.png');
    plt.close()
    return


def pileup(hists, zs, zmeans, z_pileup, dz, weight, nbins):
    """Cuts off z, zmean and Nz at a pileup-z, and stacks tail on pileup-z and renormalises"""
    ## Pile up very high z in last bin
    import copy
    # print(hists)
    hists_piled = copy.copy(hists)
    zbegin = int(z_pileup / dz)
    print("Dz, new-end-z,weight: ", dz, z_pileup, weight)
    for b in range(nbins):
        s = np.sum(hists[b, zbegin:])
        hists_piled[b, zbegin - 1] += s * weight
        hists_piled[b, zbegin:] = 0.

    # print(hists_piled)
    zs = zs[:zbegin + 1]
    zmeans_piled = zmeans[:zbegin]
    hists_piled = hists_piled[:, :zbegin]
    # print(hists_piled)

    for b in range(nbins):
        hists_piled[b, :] = hists_piled[b, :] / np.sum(hists_piled[b, :] * dz)

    # print(hists_piled)
    return zs, zmeans_piled, hists_piled


def redshift_histograms_stats(hists_true, hists_estimated, bins, legend_estimated):
    """Compute some statistics for the set of (true, estimated) histograms in each tomographic bin
    Parameters
    ----------
    hists_true :       An array of normalized histogramms for the different tomographic bins (the truth)
    hists_estimated :  An array of normalized histogramms for the different tomographic bins (the estimation)
    bins :             The bins corresponding to hists_true and hists_estimated
    legend_estimated:  The legend of the estimated histogram, must be 'deep' or 'wide'.
    Returns
    -------
    results :          Normalisation, mean, sigma for each tomographic bin for truth and estimated
    deltas :           A DataFrame containing for each bin the difference between the truth and estimation in mean z and sigma z
    """
    import pandas as pd

    if hists_true.ndim == 2:
        if hists_true.shape[0] != hists_estimated.shape[0]:
            raise ValueError('hists_true must contain the same number of histograms as hists_estimated')
        if hists_true.shape[1] != hists_estimated.shape[1]:
            raise ValueError('hists_true must have the same number of bins as hists_estimated')
        if hists_true.shape[1] != (bins.shape[0] - 1):
            raise ValueError('the number of bins must correspond to the length of the histograms')
        # Ignore the empty histogramms
        non_zero_true = (hists_true.sum(axis=1) != 0)
        non_zero_estimated = (hists_estimated.sum(axis=1) != 0)
        hists_true = hists_true[non_zero_true * non_zero_estimated]
        hists_estimated = hists_estimated[non_zero_true * non_zero_estimated]
    elif hists_true.ndim == 1:
        if hists_true.shape[0] != hists_estimated.shape[0]:
            raise ValueError('hists_true must have the same number of bins as hists_estimated')
        if hists_true.shape[0] != (bins.shape[0] - 1):
            raise ValueError('the number of bins must correspond to the length of the histograms')
        hists_true = [hists_true]
        hists_estimated = [hists_estimated]
    else:
        raise ValueError('hists_true has not the correct dimension')

    results = {'norm': [], 'mean': [], 'sigma': [], 'label': [], 'tomo': []}

    for tomo, (hist_true_deep, hist_deep) in enumerate(zip(hists_true, hists_estimated)):
        # color = 'C{0}'.format(tomo)
        for hist, label in zip([hist_true_deep, hist_deep], ['true', legend_estimated]):
            norm, mean, sigma = one_point_statistics(hist, bins)
            results['norm'].append(norm)
            results['mean'].append(mean)
            results['sigma'].append(sigma)
            results['label'].append(label)
            results['tomo'].append(tomo)
    results = pd.DataFrame(results)

    delta_mean_z = results[results['label'] == 'true']['mean'].values - results[results['label'] == legend_estimated][
        'mean'].values
    delta_sigma_z = results[results['label'] == 'true']['sigma'].values - results[results['label'] == legend_estimated][
        'sigma'].values
    deltas = pd.concat((pd.Series(delta_mean_z, name='delta <z>'), pd.Series(delta_sigma_z, name='delta sigma(z)')),
                       axis=1)

    return results, deltas


def one_point_statistics(y, bins):
    """Given a histogram and its bins return summary statistics

    Parameters
    ----------
    y :     A histogram of values
    bins :  The bins of the histogram

    Returns
    -------
    normalization, mean, sigma
    """
    dx = np.diff(bins)
    x = 0.5 * (bins[1:] + bins[:-1])
    normalization = np.trapz(y, x=x, dx=dx)
    mean = np.trapz(x * y, x=x, dx=dx) / normalization
    var = np.trapz((x - mean) ** 2 * y, x=x, dx=dx) / normalization
    sigma = np.sqrt(var)
    return normalization, mean, sigma


def plot_redshift_histograms(hists_true, hists_estimated, bins, title=None, legend_estimated='estimated',
                             legend_true='true', max_pz=3.5, max_z=2.0):
    """Plot the set of (true, estimated) histograms in each tomographic bin

    Parameters
    ----------
    hists_true :       An array of normalized histograms for the different tomographic bins (the truth)
    hists_estimated :  An array of normalized histograms for the different tomographic bins (the estimation)
    bins :             The bins corresponding to hists_true and hists_estimated
    title :            The title of the figure
    legend_estimated:  The legend of the estimated histogram

    Returns
    -------
    fig :       A matplotlib figure of this plot
    """
    from matplotlib.pyplot import subplots
    if hists_true.ndim == 2:
        if hists_true.shape[0] != hists_estimated.shape[0]:
            raise ValueError('hists_true must contain the same number of histograms as hists_estimated')
        if hists_true.shape[1] != hists_estimated.shape[1]:
            raise ValueError('hists_true must have the same number of bins as hists_estimated')
        if hists_true.shape[1] != (bins.shape[0] - 1):
            raise ValueError('the number of bins must correspond to the length of the histograms')
        # Ignore the empty histograms
        non_zero_true = (hists_true.sum(axis=1) != 0)
        non_zero_estimated = (hists_estimated.sum(axis=1) != 0)
        hists_true = hists_true[non_zero_true * non_zero_estimated]
        hists_estimated = hists_estimated[non_zero_true * non_zero_estimated]
    elif hists_true.ndim == 1:
        if hists_true.shape[0] != hists_estimated.shape[0]:
            raise ValueError('hists_true must have the same number of bins as hists_estimated')
        if hists_true.shape[0] != (bins.shape[0] - 1):
            raise ValueError('the number of bins must correspond to the length of the histograms')
        hists_true = [hists_true]
        hists_estimated = [hists_estimated]
    else:
        raise ValueError('hists_true does not have the correct dimensions')

    # if legend_estimated not in ['wide', 'deep']:
    #     raise ValueError('The legend_estimated must be either wide or deep.')

    fig, ax = subplots(figsize=(12, 8))

    for tomo, (hist_true, hist_estimated) in enumerate(zip(hists_true, hists_estimated)):
        color = 'C{0}'.format(tomo)
        for hist, label, linestyle in zip([hist_true, hist_estimated], [legend_true, legend_estimated], ['-', '--']):
            xtd, ytd = histogramize(bins, hist)
            if tomo == 0:
                plotlabel = label
            else:
                plotlabel = None
            ax.plot(xtd, ytd, color=color, linestyle=linestyle, linewidth=2, label=plotlabel)
            if label == 'true':
                ax.fill_between(xtd, 0, ytd, color=color, alpha=0.5, label=None)
    ax.set_xlabel('$z$', fontsize=20)
    ax.set_ylabel('$P(z)$', fontsize=20)
    ax.set_xlim(0, max_z)
    ax.set_ylim(0, max_pz)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=14)
    if title is not None:
        ax.set_title(title, fontsize=20)
    fig.tight_layout()

    return fig


def get_mean_sigma(zmeans, hists):
    """Returns means and sigmas for each tomo bin
    """
    means = np.zeros(4)
    sigmas = np.zeros(4)

    for i in range(4):
        means[i] = np.sum(hists[i] * zmeans) / np.sum(hists[i])
        sigmas[i] = np.sqrt(np.sum(hists[i] * (zmeans - means[i]) ** 2) / np.sum(hists[i]))

    return means, sigmas


def histogramize(bins, y):
    """Return set of points that make it look like a histogram.
    bins is assumed to be one longer than y. You then just "plot" these!"""
    xhist = []
    yhist = []
    for i in range(len(y)):
        xhist.append(bins[i])
        xhist.append(bins[i + 1])
        yhist.append(y[i])
        yhist.append(y[i])
    xhist = np.array(xhist)
    yhist = np.array(yhist)
    return xhist, yhist


def save_des_nz(hists, zbins, n_bins, outdir, run_name, suffix):
    ### output n(z) to fits  in y1 format ###

    import astropy.io.fits as fits
    import os

    bin_spacing = (zbins[1] - zbins[0]) / 2.
    z_low = zbins[:-1]
    z_mid = z_low + bin_spacing
    z_upper = zbins[1:]

    (bin_1, bin_2, bin_3, bin_4) = hists[:n_bins]

    col1 = fits.Column(name='Z_LOW', format='D', array=z_low)
    col2 = fits.Column(name='Z_MID', format='D', array=z_mid)
    col3 = fits.Column(name='Z_HIGH', format='D', array=z_upper)
    col4 = fits.Column(name='BIN1', format='D', array=bin_1)
    col5 = fits.Column(name='BIN2', format='D', array=bin_2)
    col6 = fits.Column(name='BIN3', format='D', array=bin_3)
    col7 = fits.Column(name='BIN4', format='D', array=bin_4)

    hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7])

    print('mkdir -p ' + outdir)
    os.system('mkdir -p ' + outdir)
    os.system('chmod -R a+rx ' + outdir)

    nz_out = outdir + 'Y3_y3_redshift_distributions_{}_{}.fits'.format(run_name, suffix)
    print('write ' + nz_out)
    hdu.writeto(nz_out, overwrite=True)
    os.system('chmod a+r ' + nz_out)


def to2point(outfile, lastnz, templatef, runname, label, data_dir):
    # open-up the saved final fits
    nz = fitsio.read(lastnz)

    # open the template
    oldnz = twopoint.TwoPointFile.from_fits(templatef)

    # puts the nzs into 2pt file
    bins = ['BIN1', 'BIN2', 'BIN3', 'BIN4']
    for i, bin in enumerate(bins):
        # print(oldnz.kernels[0].nzs[i])
        oldnz.kernels[0].zlow = nz['Z_LOW']
        oldnz.kernels[0].z = nz['Z_MID']
        oldnz.kernels[0].zhigh = nz['Z_HIGH']
        oldnz.kernels[0].nzs[i] = nz[bin]
        # print(oldnz.kernels[0].nzs[i])
    oldnz.to_fits(outfile, clobber=True, overwrite=True)
    


def smooth(outfilesmooth, twoptfile, nzsmoothfile, runname, label, data_dir, oldnz):

    # Troxel's smoothing adapted
    nosmooth = twopoint.TwoPointFile.from_fits(twoptfile)
    z = nosmooth.kernels[0].z
    for i in range(4):
        b = savgol_filter(nosmooth.kernels[0].nzs[i], 25, 2)
        f = interp.interp1d(nosmooth.kernels[0].z, b, bounds_error=False, fill_value=0.)
        nosmooth.kernels[0].nzs[i] = f(z)
    nosmooth.to_fits(outfilesmooth, clobber=True, overwrite=True)
    np.savetxt(nzsmoothfile, np.vstack((nosmooth.kernels[0].zlow, nosmooth.kernels[0].nzs[0],
                                        nosmooth.kernels[0].nzs[1], nosmooth.kernels[0].nzs[2],
                                        nosmooth.kernels[0].nzs[3])).T)

    oldnz = twopoint.TwoPointFile.from_fits(twoptfile)
    means_smooth, sigmas_smooth = get_mean_sigma(nosmooth.kernels[0].z, nosmooth.kernels[0].nzs)
    means_bc_piled, sigmas_bc_piled = get_mean_sigma(oldnz.kernels[0].z, oldnz.kernels[0].nzs)

    plt.figure(figsize=(12., 8.))
    colors = ['blue', 'orange', 'green', 'red']
    for i in range(4):
        plt.fill_between(oldnz.kernels[0].z, oldnz.kernels[0].nzs[i], color=colors[i], alpha=0.3)  # ,label="fiducial")
        plt.axvline(means_smooth[i], linestyle='-.', color=colors[i], label=str(i) + ' %.3f' % (means_smooth[i]))
        plt.plot(nosmooth.kernels[0].z, nosmooth.kernels[0].nzs[i], color=colors[i])  # ,label="smooth")
        plt.axvline(means_bc_piled[i], linestyle='-', color=colors[i],
                    label=str(i) + ' smooth: %.3f' % (means_bc_piled[i]))
    plt.xlabel(r'$z$', fontsize=16)
    plt.ylabel(r'$p(z)$', fontsize=16)
    plt.xlim(0, 2)
    plt.ylim(-0.5, 4)
    plt.legend(loc='upper right', fontsize=16)
    plt.title('Wide n(z) for Y3 SOM', fontsize=16)
    plt.savefig(data_dir + 'Y3_smooth_wide_nz_faint.png')
