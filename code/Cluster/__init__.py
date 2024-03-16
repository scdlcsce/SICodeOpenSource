# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
#
"""Cluster Analysis.

The Bio.Cluster provides commonly used clustering algorithms and was
designed with the application to gene expression data in mind. However,
this module can also be used for cluster analysis of other types of data.

Bio.Cluster and the underlying C Clustering Library is described in
M. de Hoon et al. (2004) https://doi.org/10.1093/bioinformatics/bth078
"""

import numbers

try:
    import numpy
except ImportError:
    from Bio import MissingPythonDependencyError

    raise MissingPythonDependencyError(
        "Please install numpy if you want to use Bio.Cluster. "
        "See http://www.numpy.org/"
    ) from None

from . import _cluster

__all__ = (
    "kcluster",
)


__version__ = _cluster.version()


def kcluster(
    data,
    nclusters=2,
    mask=None,
    weight=None,
    transpose=False,
    npass=1,
    method="a",
    dist="e",
    initialid=None,
):
    """Perform k-means clustering.

    This function performs k-means clustering on the values in data, and
    returns the cluster assignments, the within-cluster sum of distances
    of the optimal k-means clustering solution, and the number of times
    the optimal solution was found.

    Keyword arguments:
     - data: nrows x ncolumns array containing the data values.
     - nclusters: number of clusters (the 'k' in k-means).
     - mask: nrows x ncolumns array of integers, showing which data
       are missing. If mask[i,j]==0, then data[i,j] is missing.
     - weight: the weights to be used when calculating distances
     - transpose:
       - if False: rows are clustered;
       - if True: columns are clustered.
     - npass: number of times the k-means clustering algorithm is
       performed, each time with a different (random) initial
       condition.
     - method: specifies how the center of a cluster is found:
       - method == 'a': arithmetic mean;
       - method == 'm': median.
     - dist: specifies the distance function to be used:
       - dist == 'e': Euclidean distance;
       - dist == 'b': City Block distance;
       - dist == 'c': Pearson correlation;
       - dist == 'a': absolute value of the correlation;
       - dist == 'u': uncentered correlation;
       - dist == 'x': absolute uncentered correlation;
       - dist == 's': Spearman's rank correlation;
       - dist == 'k': Kendall's tau.
     - initialid: the initial clustering from which the algorithm
       should start.
       If initialid is None, the routine carries out npass
       repetitions of the EM algorithm, each time starting from a
       different random initial clustering. If initialid is given,
       the routine carries out the EM algorithm only once, starting
       from the given initial clustering and without randomizing the
       order in which items are assigned to clusters (i.e., using
       the same order as in the data matrix). In that case, the
       k-means algorithm is fully deterministic.

    Return values:
     - clusterid: array containing the number of the cluster to which each
       item was assigned in the best k-means clustering solution that was
       found in the npass runs;
     - error: the within-cluster sum of distances for the returned k-means
       clustering solution;
     - nfound: the number of times this solution was found.
     - centers: centers.
    """
    data = __check_data(data)
    shape = data.shape
    if transpose:
        ndata, nitems = shape
    else:
        nitems, ndata = shape
    mask = __check_mask(mask, shape)
    weight = __check_weight(weight, ndata)
    clusterid, centers, npass = __check_initialid(initialid, npass, nitems, nclusters, shape[1])
    error, nfound = _cluster.kcluster(
        data, nclusters, mask, weight, transpose, npass, method, dist, clusterid, centers
    )
    return clusterid, error, nfound, centers


# Everything below is private
#


def __check_data(data):
    if isinstance(data, numpy.ndarray):
        data = numpy.require(data, dtype="d", requirements="C")
    else:
        data = numpy.array(data, dtype="d")
    if data.ndim != 2:
        raise ValueError("data should be 2-dimensional")
    if numpy.isnan(data).any():
        raise ValueError("data contains NaN values")
    return data


def __check_mask(mask, shape):
    if mask is None:
        return numpy.ones(shape, dtype="intc")
    elif isinstance(mask, numpy.ndarray):
        return numpy.require(mask, dtype="intc", requirements="C")
    else:
        return numpy.array(mask, dtype="intc")


def __check_weight(weight, ndata):
    if weight is None:
        return numpy.ones(ndata, dtype="d")
    if isinstance(weight, numpy.ndarray):
        weight = numpy.require(weight, dtype="d", requirements="C")
    else:
        weight = numpy.array(weight, dtype="d")
    if numpy.isnan(weight).any():
        raise ValueError("weight contains NaN values")
    return weight


def __check_initialid(initialid, npass, nitems, ncluster, dim):
    if initialid is None:
        if npass <= 0:
            raise ValueError("npass should be a positive integer")
        clusterid = numpy.empty(nitems, dtype="intc")
        centers = numpy.empty(ncluster * dim, dtype="float64")
    else:
        npass = 0
        clusterid = numpy.array(initialid, dtype="intc")
        centers = numpy.array(ncluster * dim, dtype="float64")
    return clusterid, centers, npass


def __check_index(index):
    if index is None:
        return numpy.zeros(1, dtype="intc")
    elif isinstance(index, numbers.Integral):
        return numpy.array([index], dtype="intc")
    elif isinstance(index, numpy.ndarray):
        return numpy.require(index, dtype="intc", requirements="C")
    else:
        return numpy.array(index, dtype="intc")


def __check_distancematrix(distancematrix):
    if distancematrix is None:
        return distancematrix
    if isinstance(distancematrix, numpy.ndarray):
        distancematrix = numpy.require(distancematrix, dtype="d", requirements="C")
    else:
        try:
            distancematrix = numpy.array(distancematrix, dtype="d")
        except ValueError:
            n = len(distancematrix)
            d = [None] * n
            for i, row in enumerate(distancematrix):
                if isinstance(row, numpy.ndarray):
                    row = numpy.require(row, dtype="d", requirements="C")
                else:
                    row = numpy.array(row, dtype="d")
                if row.ndim != 1:
                    raise ValueError("row %d is not one-dimensional" % i) from None
                m = len(row)
                if m != i:
                    raise ValueError(
                        "row %d has incorrect size (%d, expected %d)" % (i, m, i)
                    ) from None
                if numpy.isnan(row).any():
                    raise ValueError("distancematrix contains NaN values") from None
                d[i] = row
            return d
    if numpy.isnan(distancematrix).any():
        raise ValueError("distancematrix contains NaN values")
    return distancematrix
