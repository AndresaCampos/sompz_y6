import numpy as np
import pandas
from matplotlib import pyplot as pl


# import cmasher as cmr

class NoiseSOM:
    """Class to build a SOM that deals with noisy data."""

    def __init__(self, 
                 metric, 
                 data, 
                 errors, 
                 learning,
                 shape=(32, 32),
                 minError=0.01,
                 wrap=False,
                 logF=True,
                 initialize='uniform',
                 gridOverDimensions=None):
        """
        metric: a class to define the distance metric and shifting rules.
        data, errors: arrays of shape (M,N) giving the values and errors, respectively,
                      for each of the N features of M training points.  If
                      data=None, and an array is passed as initialize argument,
                      then this will be adopted as the weights with no training.
        learning: a function giving strength of cell shifts as function of training
                 iteration number and the distance (in SOM space) between a cell
                 and the BMU.  Takes arguments ( xy, shape, wrap, index of bmu, iteration)
                 and returns an array of given shape holding values for each cell.
        shape: shape of the SOM cell array (any number of dimensions allowed)
        minError: floor placed on observational error on each feature (or fractional
                 error if logF=True).
        wrap:  does the SOM have periodic boundary conditions?
        logF:  should features be treated logarithmically for initial setup?
        initialize:  how to set up initial values of weights, choices:
            'uniform': randomly spaced uniformly within (log) bounds of data
            'sample':  start from a sample of the data
            <array>:   use specified array of dimens shape+(N,)
        gridOverDimensions: if you give a tuple of length equal to the number of
                 dimensions of the SOM, then the weights will be initialized to be
                 in a (log) grid over these dimensions of feature space
                 (and retain other initialization method in other dimensions).
        """

        # save parameters
        self.metric = metric
        self.shape = np.array(shape)
        self.wrap = wrap
        self.logF = logF
        self.minError = minError
        if data is None:
            # No data given for training.  We'd better have an array given to initialize.
            if isinstance(initialize, np.ndarray):
                # Last dimension of input weights is feature length
                self.N = initialize.shape[-1]
                # Initialize with given array
                if initialize.shape[:-1] == tuple(self.shape):
                    # Copy and flatten the weight array
                    self.weights = np.array(initialize).reshape(-1, self.N)
                elif len(initialize.shape) == 2 and np.prod(self.shape) == initialize.shape[0]:
                    # Array is already flattened, just copy it
                    self.weights = np.array(initialize)
                else:
                    raise ValueError('Wrong shape for initialize ndarray', initialize.shape)
                if self.logF and np.min(self.weights.flatten()) <= 0:
                    # Cannot deal with negative weights in a log SOM:
                    raise ValueError('Non-positive feature in initialization of ' +
                                     'log-domain NoiseSOM')
                return  # No training needed
            else:
                # Failure if there is no initial weight vector given.
                raise ValueError('Neither training data nor weight matrix given for NoiseSOM')

        # Train the SOM using data

        self.N = data.shape[1]
        nTrain = data.shape[0]

        if self.logF:
            ee = np.maximum(errors, np.abs(data) * self.minError)
        else:
            ee = np.maximum(errors, self.minError)

        # Create SOM and initial weight vectors
        # First get weight bounds
        if self.logF:
            # Spread uniformly on log space, clipping low features at +1-sigma
            minF = np.log(np.min(np.maximum(data, errors), axis=0))
            maxF = np.log(np.max(data, axis=0))
        else:
            # Spread linearly to max/in
            minF = np.min(data, axis=0)
            maxF = np.max(data, axis=0)

        nCells = np.product(shape)
        if initialize == 'uniform':
            # Populate weights with random numbers
            self.weights = np.random.rand(nCells, self.N)
            self.weights = minF + (maxF - minF) * self.weights
            if self.logF:
                # Put weights back into linear form
                self.weights = np.exp(self.weights)
        elif initialize == 'sample':
            # Populate weights with a random sample from the data
            indices = np.random.choice(data.shape[0], size=self.shape,
                                       replace=False).flatten()
            if logF:
                # Allow no negatives - lower bound at 1-sigma
                self.weights = np.maximum(data[indices, :], errors[indices, :])
            else:
                self.weights = data[indices, :]
        elif isinstance(initialize, np.ndarray):
            # Initialize with given array
            if initialize.shape == tuple(self.shape) + (self.N,):
                # Copy and flatten the weight array
                self.weights = np.array(initialize).reshape(-1, self.N)
            elif len(initialize.shape) == 2 and np.prod(self.shape) == initialize.shape[0]:
                # Array is already flattened, just copy it
                self.weights = np.array(initialize)
            else:
                raise ValueError('Wrong shape for initialize ndarray', initialize.shape)
            if self.logF and np.min(self.weights.flatten()) <= 0:
                raise ValueError('Non-positive feature in initialization of ' +
                                 'log-domain NoiseSOM')
        else:
            raise ValueError('Invalid initialize: ' + str(initialize))

        if gridOverDimensions is not None:
            # Place the initial weights in a grid over some dimensions
            if len(gridOverDimensions) != len(self.shape) or np.min(gridOverDimensions) < 0 or np.max(
                    gridOverDimensions) >= len(self.shape):
                print("NoiseSOM requested grid over", gridOverDimensions,
                      "does not match dimensions of SOM", self.shape)
                raise TypeError
            indices = np.unravel_index(np.arange(self.weights.shape[0]),
                                       [self.shape[i] for i in gridOverDimensions])
            for i, j in enumerate(gridOverDimensions):
                self.weights[:, j] = minF[j] + \
                                     (0.5 + indices[i]) * ((maxF[j] - minF[j]) / self.shape[j])
                if self.logF:
                    self.weights[:, j] = np.exp(self.weights[:, j])

        # Create array holding x1,x2,... coordinates of
        # each cell in SOM space
        xxx = np.meshgrid(*[np.arange(i) for i in self.shape])
        xy = np.vstack([i.flatten() for i in xxx[::-1]]).transpose()
        del xxx

        # Shuffle data for training
        order = np.arange(nTrain)
        np.random.shuffle(order)

        # Training loop
        minLearn = 0.001  # Don't update cells whose learning function is below this
        for i in range(nTrain):
            if i % 10000 == 0:
                print('Training', i)
            # Calculate p , get BMU
            dd = data[order[i]]
            err = ee[order[i]]
            bmu = self.getBMU(dd, err)

            # Get the learning function values
            fLearn = learning(xy, shape=self.shape, wrap=self.wrap, bmu=bmu, iteration=i)

            # At this point mask to only cells that will learn something
            use = fLearn >= minLearn

            # Ask the metric to update the cell features 
            ww = self.weights[use, :]
            self.metric.update(ww, fLearn[use], dd, err)
            self.weights[use, :] = ww

        return

    def chisq(self, data, errors):
        """
        Return (flattened) array of -2 ln(probabilities) for each cell,
        i.e. distance-squared.
        """

        return self.metric(self.weights, data, errors)

    def getBMU(self, data, errors):
        """
        Assign a feature vector to a cell with maximum probability.
        Returns flattened index of BMU.
        """
        return np.argmin(self.chisq(data, errors))

    def classify(self, data, errors):
        """
        Return a vector of BMU's for each row of data.
        Also returns a vector of the distance^2 to each BMU.
        """
        # Break the inputs into chunks for speed
        blocksize = 10
        nPts = data.shape[0]
        bmu = np.zeros(nPts, dtype=int)
        dsq = np.zeros(nPts, dtype=float)
        for first in range(0, nPts, blocksize):
            if first % 10 == 0:
                print("classifying", first)
            last = min(first + blocksize, nPts)
            d = self.metric(self.weights, data[first:last], errors[first:last])
            bb = np.argmin(d, axis=0)
            bmu[first:last] = bb
            dsq[first:last] = d[bb, np.arange(d.shape[1])]
        return bmu, dsq

    def fuzzyProb(self, fluxes, invVars,
                  scale=None, sPenalty=None,
                  maxScale=False):
        """
        Calculate the relative probability of obtaining the `fluxes` given the
        SOM cell fluxes, assuming Gaussian errors on each feature with
        `invvar` as inverse variance of each feature.

        Marginalize over the `scale` array of scaling factors of nodal fluxes,
        with `sPenalty` giving the -2 ln(p(s)) each scale value.

        `fluxes` and `invVars` are both (nTargets,nFeatures) arrays.

        if `scale` or `sPenalty` are `None`, then they are taken from the
        SOM metric.

        If `maxScale` is True, then the probability of the best-fit scale factor
        is used (including its `sPenalty`), rather than marginalizing.

        Returns an (nCells,nTargets) array giving relative p(flux | cell) under assumption
        of diagonal Gaussian errors on the fluxes.  Note these do *not* have
        common normalization across targets, only across cells for fixed target.
        """
        chunk = 256
        probs = np.zeros((self.weights.shape[0], fluxes.shape[0]), dtype=float)
        if scale is None:
            scale = self.metric.s
        if sPenalty is None:
            sPenalty = self.metric.sPenalty
        for first in range(0, fluxes.shape[0], chunk):
            last = min(fluxes.shape[0], first + chunk)
            ss = slice(first, last)
            if first % 1024 == 0:
                print('Doing', first)
            # Contract sums over feature dimensions
            snn = np.einsum('ik,ik,jk->ij', self.weights, self.weights, invVars[ss], optimize=True)
            snt = np.einsum('ik,jk,jk->ij', self.weights, fluxes[ss], invVars[ss], optimize=True)
            stt = np.einsum('jk,jk,jk->j', fluxes[ss], fluxes[ss], invVars[ss], optimize=True)
            # Construct chisq as function of s
            chisq = stt[np.newaxis, :, np.newaxis] - (2 * scale) * snt[:, :, np.newaxis] \
                    + snn[:, :, np.newaxis] * (scale * scale) + sPenalty
            # To avoid underflows, take out min chisq for each target
            chisq0 = np.min(np.min(chisq, axis=2), axis=0)
            chisq -= chisq0[np.newaxis, :, np.newaxis]
            if maxScale:
                # Take best scale factor 
                probs[:, ss] = np.exp(-0.5 * np.min(chisq, axis=2))
            else:
                # Sum probabilities over s
                probs[:, ss] = np.sum(np.exp(-0.5 * chisq), axis=2)
            del chisq
        return probs


class hFunc:
    """
    An implementation of a SOM learning function as given by Speagle
    """

    def __init__(self, nTrain, a=(0.5, 0.1), sigma=(10., 1.)):
        self.nTrain = float(nTrain)
        self.a = a
        self.sigma = sigma
        return

    def __call__(self, xy, shape, wrap, bmu, iteration):
        """
        Return array of learning weights (0<=wt<=1) for each cell
        """
        f = iteration / self.nTrain
        aFactor = 1. / ((1. - f) / self.a[0] + f / self.a[1])
        invS = ((1. - f) / self.sigma[0] + f / self.sigma[1]) ** 2
        dxy = xy - np.unravel_index(bmu, shape)
        if wrap:
            dxy = np.remainder(dxy + shape // 2, shape) - shape // 2
        return aFactor * np.exp(-0.5 * np.sum(dxy * dxy, axis=1) * invS)


'''
 Define a metric interface as having two calls
 `Metric(cell_features, target_features, target_errors)`  which returns an 
    nCells x nTargets matrix giving in element (i,j) the 
    distance^2 from cell at i to target j  
    (a "cell" does not have an error associated with it).
 
 `Metric.update(cells, fractions, features, errors)` updates the nCells x nFeatures `cells` array
     to move `fractions` of the way to the `features`, where `fractions` has shape (nCells,) of
     values between 0 and 1.  The new nodal features must remain positive.
'''


class AsinhMetric:
    """
    Class meeting the metric interface which does a good job
    of being linear at low S/N, log at high S/N for data"""

    def __init__(self, lnScaleSigma=0.4, lnScaleStep=0.02, maxSigma=3.):
        """Create a distance metric between scale-smeared cells and some features
        lnScaleSigma: sigma of a Gaussian in ln(s), where s is an overall scale factor
            applied to the feature vector of a SOM cell, that will be marginalized
            over.  Enter <=0 to just use unit scale factor.
        maxSigma:  largest number of sigma to extend scale factor.
        lnScaleStep: step size in ln(scale) used when integrating over scale
        """
        if lnScaleSigma > 0.:
            # Create an array of scale factors and distance-sq to the
            # center of the fuzzy template
            nS = int(np.ceil(maxSigma * lnScaleSigma / lnScaleStep))
            lnS = np.linspace(-maxSigma * lnScaleSigma, 3. * lnScaleSigma, nS)
            self.s = np.exp(lnS)
            self.sPenalty = (lnS / lnScaleSigma) ** 2
        else:
            self.s = np.ones(1, dtype=float)
            self.sPenalty = self.s * 0.
        return

    def __call__(self, cells, features, errors):
        if len(cells.shape) != 2:
            raise ValueError('Metric cells is wrong dimension')
        if features.shape != errors.shape:
            raise ValueError('Metric features and errors do not match')
        if cells.shape[-1] != features.shape[-1]:
            raise ValueError('Metric cells and features have mismatched no. of features')
        if len(features.shape) == 1:
            vf = (features / errors).reshape(1, features.shape[0])
            ee = errors.reshape(vf.shape)
        elif len(features.shape) == 2:
            vf = features / errors
            ee = errors
            # vf is S/N of the features, with shape (nTargets,nFeatures)
        else:
            raise ValueError('Metric features has invalid dimensions')
        vn = cells[:, np.newaxis, :] / ee[np.newaxis, :, :]
        # And vn is cell S/N with shape (nCells,nTargets, nFeatures)

        # Here is our rescaling function:
        sum = np.zeros((vn.shape[0], vn.shape[1]), dtype=float)

        # Consider a range of rescaling options for the cells
        # and return the one with least distance.
        # Break the cells into bunches to avoid super-large 4d arrays

        chunk = 512
        # Here is the destination array for the results
        dsq = np.zeros((vn.shape[0], vf.shape[0]), dtype=float)
        df = np.arcsinh(vf)
        # weight for asinh vs geometric mean metrics
        w = np.minimum(np.exp(2 * (vf - 4)), 1000.)
        if np.any(np.isinf(w)):
            print('inf in w at', np.where(np.isinf(w)))
        if np.any(np.isnan(w)):
            print('nan in w at', np.where(np.isnan(w)))

        h = np.hypot(1, vf)
        for first in range(0, vn.shape[0], chunk):
            last = min(first + chunk, vn.shape[0])
            # Create rescaled cells, shape=(nS,nCells,nTargets,nFeatures)
            vnS = self.s[:, np.newaxis, np.newaxis, np.newaxis] * vn[first:last, :]
            dn = np.arcsinh(vnS)
            tmp = w * np.log(2 * vnS) + dn
            ####
            if np.any(np.isinf(tmp)):
                print("inf tmp at", np.where(np.isinf(tmp)))
                print(np.any(np.isinf(w)),
                      np.any(np.isinf(vnS)),
                      np.any(np.isinf(dn)),
                      np.any(vnS <= 0))
            if np.any(np.isnan(tmp)):
                print("nan tmp at", np.where(np.isnan(tmp)))
                print(np.any(np.isnan(w)),
                      np.any(np.isnan(vnS)),
                      np.any(np.isnan(dn)),
                      np.any(vnS <= 0))

            dn = tmp / (1 + w)
            d = (dn - df) * h
            dsq0 = np.sum(d * d, axis=3)  # Sum distance over features
            # Now add penalty for the scaling factor
            dsq0 += self.sPenalty[:, np.newaxis, np.newaxis]
            # Take minimum distance of all scaling factors
            dsq[first:last, :] = np.min(dsq0, axis=0)

        return dsq

    def update(self, cells, fractions, features, errors):
        if len(cells.shape) != 2:
            raise ValueError('Metric cells is wrong dimension')
        if len(fractions.shape) != 1 or fractions.shape[0] != cells.shape[0]:
            raise ValueError('Metric fractions array is wrong shape')
        if len(features.shape) > 1:
            raise ValueError('Metric gradient only works for single feature vector')
        if features.shape != errors.shape:
            raise ValueError('Metric features and errors do not match')
        if cells.shape[-1] != features.shape[-1]:
            raise ValueError('Metric cells and features have mismatched no. of features')

        # Write just an unscaled version first
        vf = features / errors
        vn = cells / errors

        factor = np.maximum(1., vf) / vn
        # Don't move if there's no information at all
        lowSN = np.maximum(vn, vf) < 2.  # Arbitrary choice of lower limits here
        factor[lowSN] = 1.

        cells *= np.power(factor, fractions[:, np.newaxis])
        return


class LinearMetric:
    """
    Metric interface implementation for error-scaled Euclidean distances,
    e.g. Gaussian probabilities.
    """

    def __init__(self, noise0=0, signalScale=None):
        """Create a distance metric between scale-smeared cells and some features
        noise0: when a move is requested, each dimension's move will be suppressed
            by a factor noise^2/(noise0^2+noise^2) as a means of deweighting bad
            measurements.
        signalScale: if given, it should be an array over features giving range of
            signals, and the move suppression for noise0 above will be
            calculated using noise/signalScale in each feature dimension.
        """
        self.noise0 = noise0
        if signalScale is None:
            self.scale = None
        else:
            self.scale = np.array(signalScale)
        return

    def __call__(self, cells, features, errors):
        if len(cells.shape) != 2:
            raise ValueError('Metric cells is wrong dimension')
        if features.shape != errors.shape:
            raise ValueError('Metric features and errors do not match')
        if cells.shape[-1] != features.shape[-1]:
            raise ValueError('Metric cells and features have mismatched no. of features')
        if len(features.shape) == 1:
            vf = (features / errors).reshape(1, features.shape[0])
            ee = errors.reshape(1, features.shape[0])
        elif len(features.shape) == 2:
            vf = features / errors
            ee = errors
            # vf is S/N of the features, with shape (nTargets,nFeatures)
        else:
            raise ValueError('Metric features has invalid dimensions')
        vn = cells[:, np.newaxis, :] / ee[np.newaxis, :, :]
        # And vn is cell S/N with shape (nCells,nTargets, nFeatures)

        # Simple Euclidean distance between error-scaled values:
        d = vn - vf
        return np.sum(d * d, axis=2)

    def update(self, cells, fractions, features, errors):
        if len(cells.shape) != 2:
            raise ValueError('Metric cells is wrong dimension')
        if len(fractions.shape) != 1 or fractions.shape[0] != cells.shape[0]:
            raise ValueError('Metric fractions array is wrong shape')
        if len(features.shape) > 1:
            raise ValueError('Metric gradient only works for single feature vector')
        if features.shape != errors.shape:
            raise ValueError('Metric features and errors do not match')
        if cells.shape[-1] != features.shape[-1]:
            raise ValueError('Metric cells and features have mismatched no. of features')

        # The shift is just difference scaled by fractions and
        # any suppression factor
        shift = features - cells  # Shape is (nCells, nFeatures)
        shift = shift * fractions[:, np.newaxis]
        if self.noise0 > 0:
            # Suppress moves in features with large errors
            noise = np.array(errors)
            if self.scale is not None:
                # Rescale noise levels by signal level
                noise /= self.scale
            noise *= noise
            factor = noise / ((self.noise0 * self.noise0) + noise)
            shift *= factor

        cells += shift
        return


def readCOSMOS():
    """Function to read the COSMOS input files.
    Returns arrays fluxes,errors,redshifts,counts giving for each unique object
    its ugrizJHK fluxes, flux errors, Laigle redshift, number of Balrog counts,
    and radec array.
    """
    # Read the master file.  Mag zeropoints are all 30.0
    cosmos = pandas.read_hdf('cosmos.hdf5', 'fluxes')

    # Pull out numpy arrays for our quantities of interest
    fluxes = np.vstack([cosmos['BDF_FLUX_DERED_U'],
                        cosmos['BDF_FLUX_DERED_G'],
                        cosmos['BDF_FLUX_DERED_R'],
                        cosmos['BDF_FLUX_DERED_I'],
                        cosmos['BDF_FLUX_DERED_Z'],
                        cosmos['BDF_FLUX_DERED_J'],
                        cosmos['BDF_FLUX_DERED_H'],
                        cosmos['BDF_FLUX_DERED_K']]).transpose()
    errors = np.vstack([cosmos['BDF_FLUX_ERR_DERED_U'],
                        cosmos['BDF_FLUX_ERR_DERED_G'],
                        cosmos['BDF_FLUX_ERR_DERED_R'],
                        cosmos['BDF_FLUX_ERR_DERED_I'],
                        cosmos['BDF_FLUX_ERR_DERED_Z'],
                        cosmos['BDF_FLUX_ERR_DERED_J'],
                        cosmos['BDF_FLUX_ERR_DERED_H'],
                        cosmos['BDF_FLUX_ERR_DERED_K']]).transpose()
    redshifts = np.array(cosmos['Z'])
    radec = np.vstack([cosmos['RA'],
                       cosmos['DEC']]).transpose()

    # Reduce to the unique COSMOS objects, keep track of number of Balrog detections
    junk, indices, counts = np.unique(fluxes[:, 0], return_index=True, return_counts=True)
    fluxes = fluxes[indices]
    errors = errors[indices]
    redshifts = redshifts[indices]
    radec = radec[indices]
    return fluxes, errors, redshifts, counts, radec


def somPlot3d(som, az=200., el=30.):
    # Make a 3d plot of cells weights in color space.  az/el are plot view angle
    mags = 30. - 2.5 * np.log10(som.weights)
    ug = mags[:, 0] - mags[:, 1]
    gi = mags[:, 1] - mags[:, 3]
    ik = mags[:, 3] - mags[:, 7]
    imag = mags[:, 3]
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.azim = az
    ax.elev = el
    ax.scatter(gi, ik, imag, c=ug, cmap='Spectral_r')
    # Draw the outline of the SOM edges
    xx = np.arange(som.shape[0], dtype=int)
    yy = np.arange(som.shape[1], dtype=int)
    xxx = np.hstack((xx,
                     np.ones(len(yy) - 2, dtype=int) * xx[-1],
                     xx[::-1],
                     np.zeros(len(yy) - 1, dtype=int)))
    yyy = np.hstack((np.zeros(len(xx), dtype=int),
                     yy[1:-1],
                     np.ones(len(xx) - 1, dtype=int) * yy[-1],
                     yy[-1::-1]))
    ii = np.ravel_multi_index((xxx, yyy), som.shape)
    ax.plot(gi[ii], ik[ii], imag[ii], 'k--')
    ax.set_title('Node locations')
    ax.set_aspect('equal')
    ax.set_xlabel('gi')
    ax.set_ylabel('ik')
    ax.set_zlabel('imag')
    return


def somPlot2d(som):
    # Make a 2d plot of cells weights in color-color diagram space.
    mags = 30. - 2.5 * np.log10(som.weights)
    ug = mags[:, 0] - mags[:, 1]
    gi = mags[:, 1] - mags[:, 3]
    ik = mags[:, 3] - mags[:, 7]
    imag = mags[:, 3]
    fig = pl.figure(figsize=(6, 7))
    # First a color-color plot of nodes
    pl.scatter(gi, ik, c=imag, alpha=0.3, cmap='Spectral')
    # Draw the outline of the SOM edges
    xx = np.arange(som.shape[0], dtype=int)
    yy = np.arange(som.shape[1], dtype=int)
    xxx = np.hstack((xx,
                     np.ones(len(yy) - 2, dtype=int) * xx[-1],
                     xx[::-1],
                     np.zeros(len(yy) - 1, dtype=int)))
    yyy = np.hstack((np.zeros(len(xx), dtype=int),
                     yy[1:-1],
                     np.ones(len(xx) - 1, dtype=int) * yy[-1],
                     yy[-1::-1]))
    ii = np.ravel_multi_index((xxx, yyy), som.shape)
    pl.plot(gi[ii], ik[ii], 'k-')
    pl.title('Node locations')
    pl.gca().set_aspect('equal')
    cb = pl.colorbar()
    cb.set_label('imag')
    pl.xlabel('gi')
    pl.ylabel('ik')
    return


def somDomainColors(som):
    # Make 4-panel plot colors and mag across SOM space
    mags = 30. - 2.5 * np.log10(som.weights)
    ug = mags[:, 0] - mags[:, 1]
    gi = mags[:, 1] - mags[:, 3]
    ik = mags[:, 3] - mags[:, 7]
    imag = mags[:, 3]
    fig = pl.figure(figsize=(6, 7))

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(8, 8))
    im = ax[0, 0].imshow(gi.reshape(som.shape), interpolation='nearest', origin='lower',
                         cmap='Spectral_r')
    ax[0, 0].set_title('gi')
    ax[0, 0].set_aspect('equal')
    pl.colorbar(im, ax=ax[0, 0])

    im = ax[1, 0].imshow(ug.reshape(som.shape), interpolation='nearest', origin='lower',
                         cmap='Spectral_r')
    ax[1, 0].set_title('ug')
    ax[1, 0].set_aspect('equal')
    pl.colorbar(im, ax=ax[1, 0])

    im = ax[0, 1].imshow(ik.reshape(som.shape), interpolation='nearest', origin='lower',
                         cmap='Spectral_r')
    ax[0, 1].set_title('ik')
    ax[0, 1].set_aspect('equal')
    pl.colorbar(im, ax=ax[0, 1])

    im = ax[1, 1].imshow(imag.reshape(som.shape), interpolation='nearest', origin='lower',
                         cmap='Spectral')
    ax[1, 1].set_title('imag')
    ax[1, 1].set_aspect('equal')
    pl.colorbar(im, ax=ax[1, 1])
    return


def plotSOMz(som, cells, zz, subsamp=1, figsize=(8, 8)):
    """Make 4-panel plot showing occupancy of SOM by a redshift sample and statistics
       of redshift distribution in each cell."""
    nbins = np.prod(som.shape)
    nn = np.histogram(cells, bins=nbins, range=(-0.5, nbins - 0.5))[0]
    zmean = np.histogram(cells, bins=nbins, range=(-0.5, nbins - 0.5), weights=zz[::subsamp])[0] / nn
    zvar = np.histogram(cells, bins=nbins, range=(-0.5, nbins - 0.5), weights=(zz * zz)[::subsamp])[0] / nn
    zrms = np.sqrt(zvar - zmean * zmean)
    zmed = np.array([np.median(zz[cells == i]) for i in range(nbins)])

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=figsize)

    im = ax[0, 0].imshow(np.log10(nn.reshape(som.shape)), interpolation='nearest', origin='lower')  # ,
    # cmap=cmr.heat)
    ax[0, 0].set_aspect('equal')
    ax[0, 0].set_title('Sources per cell')
    pl.colorbar(im, ax=ax[0, 0])

    useful = nn > 4
    im = ax[0, 1].imshow(zmed.reshape(som.shape), interpolation='nearest', origin='lower',
                         vmax=2.5, vmin=0., cmap='Spectral')
    ax[0, 1].set_aspect('equal')
    ax[0, 1].set_title('z_median')
    pl.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow((zrms / (1 + zmed)).reshape(som.shape), interpolation='nearest', origin='lower',
                         cmap='Spectral')
    ax[1, 0].set_aspect('equal')
    ax[1, 0].set_title('std(z)/(1+zmed)')
    pl.colorbar(im, ax=ax[1, 0])

    print('Median sig(ln(z)):', np.median((zrms / (1 + zmed))[useful]))

    # make another plot showing rms of neighbor cells
    tmp = zmed.reshape(som.shape)
    tmp2 = np.stack((tmp[:-2, :-2],
                     tmp[:-2, 1:-1],
                     tmp[:-2, 2:],
                     tmp[1:-1, :-2],
                     tmp[1:-1, 1:-1],
                     tmp[1:-1, 2:],
                     tmp[2:, :-2],
                     tmp[2:, 1:-1],
                     tmp[2:, 2:]), axis=0)
    grad = np.std(tmp2, axis=0) / tmp[1:-1, 1:-1] / 2.
    print('Median neighbor sig(ln(z)):', np.median(grad[~np.isnan(grad)]))
    tmp = (zrms / (1 + zmean)).reshape(som.shape)[1:-1, 1:-1] / grad
    im = ax[1, 1].imshow(tmp, interpolation='nearest', origin='lower', cmap='Spectral', vmin=0, vmax=5)
    ax[1, 1].set_aspect('equal')
    ax[1, 1].set_title('stddev / local slope')
    pl.colorbar(im, ax=ax[1, 1])

    return
