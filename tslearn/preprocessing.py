"""
The :mod:`tslearn.preprocessing` module gathers time series scalers.
"""

import numpy, sys
from sklearn.base import TransformerMixin
from scipy.interpolate import interp1d
from scipy.linalg import *
from tslearn.utils import *

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

#do this after separate_atributes_dataset
#running for several Attributes
# z-normaltion where each element of the ts is substract by the ts mean and divided by its variance
def z_normalize(dataset,variables_size):
    for u in range(0,variables_size):
        for j in range(0,len(dataset[0])):
            mean = ts_mean_single_var(dataset[u][j])
            variance = numpy.std(dataset[u][j])
            for i in range(0,len(dataset[u][j])):
                dataset[u][j][i] = numpy.array([(dataset[u][j][i] - mean)/variance])
    return dataset

#do this after separate_atributes_dataset
#running for several Attributes
#resolves dependencies between variables!
def multivariate_normalization(data,variables_size):
    for j in range(0,len(data[0])):
        means = []
        ts_s = []
        for u in range(0,variables_size):
            means.append(ts_mean_single_var(data[u][j]))
            ts_s.append(data[u][j])
            #print("mean1: ",mean1)
            #print("mean2: ",mean2)

            #not needed after all??
            #std_var1 = numpy.std(ts1)
            #std_var2 = numpy.std(ts2)
            #print("stdvar: ",std_var1)
            #print("stdvar: ",std_var2)
            #print("ts1: ", ts1)
            #print("ts2: ", ts2)

        #print("ts_with_several_variables:", ts_s)
        #print("variables_size:", len(ts_s))
        #print("ts_length:", len(ts_s[0]))

        #numpy.cov faz arredondamentos para zero o que leva a matrix nao ser semidefinitava pos
        #covariance_matrix = numpy.cov(ts_s)
        covariance_matrix = my_covariance_matrix(ts_s,variables_size)
        print("matrix: ", covariance_matrix)
        #w eigenvalues
        #v eigenvectores
        w, v= numpy.linalg.eig(covariance_matrix)
        #print("W: ", w)
        #print("v: ", v)
        diagonal = numpy.diag(w)
        #print("iden: ", diagonal)
        result = sqrtm(diagonal)
        #print("and squared: ", result)
        B = numpy.matmul(v,result)

        try:
            inverse_B = numpy.linalg.inv(B)
            #print("inverse matrix: ", inverse_B)
        except numpy.linalg.LinAlgError:
            # Not invertible. Skip this one.
            # Non invertable cases Uwave
            print("not invertible")
            sys.exit()

        for i in range(0,len(data[0][j])):
            atributes_together = []
            for u in range(0,variables_size):
                atributes_together.append(data[u][j][i])
            result = numpy.subtract(atributes_together, numpy.array(means))
            result = numpy.matmul(inverse_B,result)
            for u in range(0,variables_size):
                data[u][j][i] = result[u]
    return data

class TimeSeriesResampler(TransformerMixin):
    """Resampler for time series. Resample time series so that they reach the target size.

    Parameters
    ----------
    sz : int
        Size of the output time series.

    Example
    -------
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0. ],
            [ 1.5],
            [ 3. ],
            [ 4.5],
            [ 6. ]]])
    """
    def __init__(self, sz):
        self.sz_ = sz

    def fit_transform(self, X, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        X_ = to_time_series_dataset(X)
        n_ts, sz, d = X_.shape
        equal_size = check_equal_size(X_)
        X_out = numpy.empty((n_ts, self.sz_, d))
        for i in range(X_.shape[0]):
            xnew = numpy.linspace(0, 1, self.sz_)
            if not equal_size:
                sz = ts_size(X_[i])
            for di in range(d):
                f = interp1d(numpy.linspace(0, 1, sz), X_[i, :sz, di], kind="slinear")
                X_out[i, :, di] = f(xnew)
        return X_out


class TimeSeriesScalerMinMax(TransformerMixin):
    """Scaler for time series. Scales time series so that their span in each dimension is between ``min`` and ``max``.

    Parameters
    ----------
    min : float (default: 0.)
        Minimum value for output time series.
    max : float (default: 1.)
        Maximum value for output time series.

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Example
    -------
    >>> TimeSeriesScalerMinMax(min=1., max=2.).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1. ],
            [ 1.5],
            [ 2. ]]])
    """
    def __init__(self, min=0., max=1.):
        self.min_ = min
        self.max_ = max

    def fit_transform(self, X, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset.
        """
        X_ = to_time_series_dataset(X)
        for i in range(X_.shape[0]):
            for d in range(X_.shape[2]):
                cur_min = X_[i, :, d].min()
                cur_max = X_[i, :, d].max()
                cur_range = cur_max - cur_min
                X_[i, :, d] = (X_[i, :, d] - cur_min) * (self.max_ - self.min_) / cur_range + self.min_
        return X_


class TimeSeriesScalerMeanVariance(TransformerMixin):
    """Scaler for time series. Scales time series so that their mean (resp. standard deviation) in each dimension is
    mu (resp. std).

    Parameters
    ----------
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Example
    -------
    >>> TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[-1.22474487],
            [ 0. ],
            [ 1.22474487]]])
    """
    def __init__(self, mu=0., std=1.):
        self.mu_ = mu
        self.std_ = std

    def fit_transform(self, X, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        X_ = to_time_series_dataset(X)
        for i in range(X_.shape[0]):
            for d in range(X_.shape[2]):
                cur_mean = X_[i, :, d].mean()
                cur_std = X_[i, :, d].std()
                X_[i, :, d] = (X_[i, :, d] - cur_mean) * self.std_ / cur_std + self.mu_
        return X_
