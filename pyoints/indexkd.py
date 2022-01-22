# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University,
# lamprecht@uni-trier.de
#
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""A generic spatial index.
"""

import bisect
import numpy as np
from numbers import Number

from scipy.spatial import cKDTree
import rtree.index as r_treeIndex
from rtree import Rtree

from . import (
    assertion,
    transformation,
)

from .misc import print_rounded


class IndexKD(object):
    """Wrapper class for several spatial indices to speed up spatial queries
    and ease the usage.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.
    T : optional, array_like(Number, shape=(k+1, k+1))
        Represents any kind of transformation matrix applied to the coordinates
        before index computation.
    leafsize : optional, positive int
        Leaf size of KD-Tree.
    quickbuild : optional, bool
        Indicates whether or not the spatial index shall be optimized for quick
        building (True) or quick spatial queries (False).
    copy : optional, bool
        Indicates whether or not to copy the coordinate array.

    Attributes
    ----------
    dim : positive int
        Number of coordinate dimensions `k`.
    t : np.matrix(Number, shape=(k+1, k+1))
        Transformation matrix.
    coords : np.ndarray(Number, shape=(n, k))
        Coordinates of the spatial index.
    kd_tree : `scipy.spatial.cKDTree`
        KD-tree for rapid neighbourhood queries. Generated on first demand.
    r_tree : `rtree.Rtree`
        R-tree for rapid box queries. Generated on first demand.

    Notes
    -----
    Most spatial index operations are time critical. So it is usually avoided
    to check each input parameter in detail.

    Examples
    --------

    Create a simple spatial index.

    >>> coords = np.indices((5, 10)).reshape((2, 50)).T
    >>> indexKD = IndexKD(coords)
    >>> len(indexKD)
    50
    >>> indexKD.dim
    2

    Query points within a sphere.

    >>> nids = indexKD.ball([0, 0], 1.1)
    >>> print_rounded(nids)
    [ 0  1 10]
    >>> print_rounded(indexKD.coords[nids, :])
    [[0 0]
     [0 1]
     [1 0]]

    Scale the coordinates using a transformation matrix to enable queries
    in the shape of an ellipsoid.

    >>> T = [(0.5, 0, 0), (0, 0.8, 0), (0, 0, 1)]
    >>> indexKD = IndexKD(coords, T)

    >>> nids = indexKD.ball([0, 0], 1.1)
    >>> print_rounded(nids)
    [ 0  1 20 10 11]
    >>> print_rounded(indexKD.coords[nids, :])
    [[ 0.   0. ]
     [ 0.   0.8]
     [ 1.   0. ]
     [ 0.5  0. ]
     [ 0.5  0.8]]

    """

    def __init__(
            self,
            coords,
            T=None,
            leafsize=16,
            quickbuild=True,
            copy=True):

        if copy:
            coords = assertion.ensure_coords(coords).copy()
        else:
            coords = assertion.ensure_coords(coords)

        if not isinstance(leafsize, int) and leafsize > 0:
            raise ValueError('"leafsize" needs to be an iteger greater zero')
        if not isinstance(quickbuild, bool):
            raise ValueError('"quickbuild" needs to be boolean')

        self._leafsize = leafsize
        self._balanced = not quickbuild
        self._compact = not quickbuild

        if T is None:
            self._coords = coords
            self._t = transformation.i_matrix(self._coords.shape[1])
        else:
            self._t = assertion.ensure_tmatrix(T)
            self._coords = transformation.transform(coords, self._t)

    def __len__(self):
        """Number of points of the spatial index.

        Returns
        -------
        positive int
            Number of points.

        """
        return self.coords.shape[0]

    def __iter__(self):
        """Iterates over the points of the spatial index.

        Yields
        ------
        np.ndarray(Number, shape=(self.dim))
            point.

        """
        return enumerate(self.coords)

    def _get_bulk(self, coords, bulk):
        """Internal function to get a bulk of coordinates.

        Parameters
        ----------
        coords : iterable of array_like(Number, shape=(k))
            Coordinates of at least `self.dim` dimensions.
        bulk : positive int
            Size of bulk.

        Yields
        ------
        array_like(Number, shape=(self.dim))

        """
        dim = self.dim
        try:
            for i in range(bulk):
                yield next(coords)[:dim]
        except StopIteration:
            pass

    @property
    def dim(self):
        return self.coords.shape[1]

    @property
    def coords(self):
        return self._coords

    @property
    def t(self):
        return self._t

    @property
    def kd_tree(self):
        if not hasattr(self, '_kd_tree'):
            self._kd_tree = cKDTree(
                self.coords,
                leafsize=self._leafsize,
                copy_data=False,
                balanced_tree=self._balanced,
                compact_nodes=self._compact
            )
        return self._kd_tree

    @property
    def r_tree(self):
        if not hasattr(self, '_r_tree'):

            # define properties
            p = r_treeIndex.Property()
            p.dimension = self.dim()
            p.variant = r_treeIndex.RT_Star
            index = np.concatenate((range(self.dim()), range(self.dim())))

            # Generator provides required point format
            def gen():
                for id, coord in self:
                    yield (id, coord[index], id)
            self._r_tree = Rtree(gen(), properties=p)

        return self._r_tree

    def ball(self, coords, r, bulk=100000, **kwargs):
        """Finds all points within distance `r` of point or points `coords`.

        Parameters
        ----------
        coords : array_like(Number, shape=(n, k))
            Represents `n` data points of `k` dimensions.
        r : positive float
            Radius of ball.
        bulk : optional, positive int
            Reduces required memory by performing bulk queries.
        \*\*kwargs : optional
            Additional parameters passed to
            `scipy.spatial.cKDTree.query_ball_point`

        Returns
        -------
        nIds: list or array of lists
            If coords is a single point, this returns a list of neighbours. If
            coords is a list of points, this returns a list containing the
            lists of neighbours.

        Examples
        --------

        >>> coords = np.indices((5, 10)).reshape((2, 50)).T
        >>> indexKD = IndexKD(coords)
        >>> indexKD.ball((0, 0), 1)
        [0, 1, 10]
        >>> indexKD.ball(np.array([(0, 0), (1, 1)]), 1)
        [[0, 1, 10], [1, 10, 11, 12, 21]]

        """
        if assertion.iscoord(coords):
            # single point
            return self.kd_tree.query_ball_point(
                coords[:self.dim], r, **kwargs)
        elif hasattr(r, '__iter__'):
            # query multiple radii
            return list(self.balls_iter(coords, r, **kwargs))
        else:
            # bulk queries
            return list(self.ball_iter(coords, r, bulk=bulk, **kwargs))

    def ball_iter(self, coords, r, bulk=10000, **kwargs):
        """Similar to `ball`, but yields lists of neighbours.

        Yields
        ------
        nIds : list of int
            Lists of indices of neighbouring points.

        See Also
        --------
        ball, balls_iter

        """
        if not isinstance(bulk, int) and bulk > 0:
            raise ValueError("bulk size has to be an integer greater zero")

        coords = iter(coords)
        while True:
            # bulk query
            bulk_coords = np.array(list(self._get_bulk(coords, bulk)))
            if len(bulk_coords) == 0:
                break
            nIds = self.kd_tree.query_ball_point(
                bulk_coords, r, n_jobs=-1, **kwargs)
            for nId in nIds:
                yield nId

    def balls_iter(self, coords, radii, **kwargs):
        """Similar to `ball_iter`, but with differing radii.

        Parameters
        ----------
        radii: iterable of float
            Radii to query.

        Yields
        ------
        nIds : list
            Lists of indices of neighbouring points.

        See Also
        --------
        ball, ball_iter

        """
        for coord, r in zip(coords, radii):
            nIds = self.kd_tree.query_ball_point(coord[:self.dim], r, **kwargs)
            yield nIds

    def ball_count(self, r, coords=None, **kwargs):
        """Counts numbers of neighbours within radius.

        Parameters
        ----------
        r : float or iterable of float
            Iterable radii to query.
        coords : optional, array_like(Number, shape=(n, k)) or iterable
            Represents `n` points of `k` dimensions. If none, it is set to
            `self.coords`.
        \*\*kwargs : optional
            Additional parameters passed to
            `scipy.spatial.cKDTree.query_ball_point`

        Returns
        -------
        numpy.ndarray(int, shape=(n))
            Number of neigbours for each point.

        See Also
        --------
        ball_count_iter, ball

        Examples
        --------

        >>> coords = [(0, 0), (0, 1), (1, 1), (2, 1), (1, 0.5), (0.5, 1)]
        >>> indexKD = IndexKD(coords)

        >>> counts = indexKD.ball_count(1)
        >>> print_rounded(counts)
        [2 4 5 2 3 4]

        >>> counts = indexKD.ball_count(1, coords=[0.5, 0.5])
        >>> print_rounded(counts)
        5

        """
        if coords is None:
            coords = self.coords

        if assertion.iscoord(coords):
            return len(self.ball(coords, r, **kwargs))
        else:
            gen = self.ball_count_iter(r, coords=coords, **kwargs)
            return np.array(list(gen), dtype=int)

    def ball_count_iter(self, r, coords=None, **kwargs):
        """Counts numbers of neighbours within radius.

        Parameters
        ----------
        r : float or iterable of float
            Iterable radii to query.
        coords : optional, array_like(Number, shape=(n, k)) or iterable
            Represents `n` points of `k` dimensions. If none, it is set to
            `self.coords`.
        \*\*kwargs : optional
            Additional parameters passed to
            `scipy.spatial.cKDTree.query_ball_point`

        Yields
        ------
        int
            Number of neigbours for each point.

        See Also
        --------
        ball_iter, balls_iter

        """
        if coords is None:
            coords = self.coords

        if hasattr(r, '__iter__'):
            nIdsGen = self.balls_iter(coords, r, **kwargs)
        else:
            nIdsGen = self.ball_iter(coords, r, **kwargs)
        return map(len, nIdsGen)

    def sphere(self, coord, r_min, r_max, **kwargs):
        """Counts numbers of neighbours within radius.

        Parameters
        ----------
        r_min : float
            Inner radius of the sphere.
        r_max : float
            Outer radius of the sphere.
        coord : array_like(Number, shape=(k))
            Center of sphere.
        \*\*kwargs : optional
            Additional parameters passed to
            `scipy.spatial.cKDTree.query_ball_point`

        Returns
        -------
        list of int
            Indices of points within sphere.

        Examples
        --------

        >>> coords = np.indices((5, 10)).reshape((2, 50)).T
        >>> indexKD = IndexKD(coords)
        >>> print_rounded(indexKD.ball((3, 3), 1))
        [32 34 33 43 23]
        >>> print_rounded(indexKD.ball((3, 3), 1.5))
        [22 42 32 34 33 43 44 23 24]
        >>> print_rounded(indexKD.sphere((3, 3), 1, 1.5))
        [23 32 33 34 43]

        """
        if not isinstance(r_min, Number) and r_min > 0:
            raise ValueError("r_min has to be numeric and greater zero")

        if not isinstance(r_max, Number) and r_max > r_min:
            raise ValueError("r_max has to be numeric and greater 'r_min'")

        inner = self.ball(coord[:self.dim], r_min, **kwargs)
        outer = self.ball(coord[:self.dim], r_max, **kwargs)
        return np.intersect1d(outer, inner)

    def knn(self, coords, k=1, bulk=100000, **kwargs):
        """Query for `k` nearest neighbours.

        Parameters
        ----------
        coords : array_like(Number, shape=(n, k))
            Represents `n` points of `k` dimensions.
        k : optional, positive int
            Number of nearest neighbours to return.
        \*\*kwargs : optional
            Additional parameters passed to
            `scipy.spatial.cKDTree.query_ball_point`

        Returns
        -------
        dists : np.ndarray(Number, shape=(n, k))
            Distances to nearest neighbours.
        indices : np.ndarray(int, shape=(n, k))
            Indices of nearest neighbours.

        Examples
        --------

        >>> coords = [(0, 0), (0, 1), (1, 1), (2, 1), (1, 0.5), (0.5, 1)]
        >>> indexKD = IndexKD(coords)

        >>> dists, nids = indexKD.knn((0.5, 1), 2)
        >>> print_rounded(dists)
        [ 0.   0.5]
        >>> print_rounded(nids)
        [5 2]

        >>> dists, nids = indexKD.knn([(0.5, 1), (1.5, 1)], 2)
        >>> print_rounded(dists)
        [[ 0.   0.5]
         [ 0.5  0.5]]
        >>> print_rounded(nids)
        [[5 2]
         [3 2]]

        >>> dists, nids = indexKD.knn([(0.5, 1), (1.5, 1), (1, 1)], [3, 1, 2])
        >>> print(dists)
        (array([ 0. ,  0.5,  0.5]), array([ 0.5]), array([ 0. ,  0.5]))
        >>> print(nids)
        (array([5, 1, 2]), array([2]), array([2, 4]))

        See Also
        --------
        knn, knns_iter, scipy.spatial.cKDTree.query

        """

        if assertion.iscoord(coords):
            # single point query
            dists, nIds = self.kd_tree.query(coords[:self.dim], k, **kwargs)
        elif hasattr(k, '__iter__'):
            # query multiple radii
            dists, nIds = zip(*self.knns_iter(coords, ks=k, **kwargs))
        else:
            # bulk queries
            dists, nIds = zip(*self.knn_iter(coords, k=k, bulk=bulk, **kwargs))
            dists = np.array(dists)
            nIds = np.array(nIds)
        return dists, nIds

    def knn_iter(self, coords, k=1, bulk=100000, **kwargs):
        """Similar to `knn`, but yields lists of neighbours.

        Yields
        -------
        dists : np.ndarray(Number, shape=(k))
            List of distances to nearest neighbours.
        indices : np.ndarray(int, shape=(k))
            List of ??? and corresponding point indices.

        See Also
        --------
        knn, knns_iter

        """
        if not isinstance(bulk, int) and bulk > 0:
            raise ValueError("bulk size has to be an integer greater zero")
        coords = iter(coords)
        while True:
            bulk_coords = list(self._get_bulk(coords, bulk))
            if len(bulk_coords) == 0:
                break
            dists_list, nIds_list = self.kd_tree.query(
                bulk_coords, k=k, **kwargs)
            for dists, nIds in zip(dists_list, nIds_list):
                yield dists, nIds

    def knns_iter(self, coords, ks, **kwargs):
        """Similar to `knn`, but yields lists of neighbours.

        Parameters
        ----------
        ks : iterable of int
            Iterable numbers of neighbours to query.

        Yields
        -------
        dists : np.ndarray(Number, shape=(k))
            List of distances to nearest neighbours.
        indices : np.ndarray(int, shape=(k))
            List of ??? and corresponding point indices.

        See Also
        --------
        knn, knn_iter

        """
        for coord, k in zip(coords, ks):
            dists, nIds = self.kd_tree.query(coord[:self.dim], k=k, **kwargs)
            if k == 1:
                dists = np.array([dists])
                nIds = np.array([nIds])
            yield dists, nIds

    @property
    def nn(self):
        """Provides the nearest neighbours for each point.

        Returns
        -------
        distances : np.ndarray(Number, shape=(n))
            Distances to each nearest neighbour.
        indices : np.ndarray(int, shape=(n))
            Indices of nearest neighbours.

        """
        if not hasattr(self, '_NN'):
            dists, ids = self.knn(self.coords, k=2, p=2)
            self._NN = dists[:, 1], ids[:, 1]
        return self._NN

    def closest(self, ids, **kwargs):
        """Provides nearest neighbour of a point.

        Parameters
        ----------
        ids : int or array_like(int, shape=(m))
            Index of point or indices of points in `self.coords`.
        \*\*kwargs : optional
            Additional parameters similar to `scipy.spatial.cKDTree.query`.

        Returns:
        --------
        distances : np.ndarray(Number, shape=(n))
            Distance to closest point.
        indices : np.ndarray(int, shape=(n))
            Index of closest point.

        See Also
        --------
        knn

        Examples
        --------

        >>> coords = np.indices((5, 10)).reshape((2, 50)).T
        >>> indexKD = IndexKD(coords)
        >>> indexKD.closest(3)
        (1.0, 4)
        >>> print_rounded(indexKD.closest([0, 2, 5, 3])[1])
        [1 1 6 4]

        """
        dists, nIds = self.knn(self.coords[ids, :], k=2, **kwargs)
        if hasattr(ids, '__len__'):
            return dists[:, 1], nIds[:, 1]
        else:
            return dists[1], nIds[1]

    def cube(self, coords, r, **kwargs):
        """Provides points within a cube.

        Notes
        -----
        Wrapper for `self.ball` with `p=np.inf`.

        See Also
        --------
        ball

        """
        return self.ball(coords, r, p=np.inf, **kwargs)

    def box(self, extent):
        """Selects points within a given extent.

        Parameters
        ----------
        extent : array_like(Number, shape=(2 * self.dim))
            Specifies the points to return. A point p is returned, if
            `np.all(p <= extent[0: dim])` and
            `np.all(p >= extent[dim+1: 2*dim])`

        Returns
        -------
        list of int
            Indices of points within the extent.

        See Also
        --------
        ball, cube, slice

        """
        return self.r_tree.intersection(extent, objects='raw')

    def box_count(self, extent):
        """Counts all points within a given extent.

        Parameters
        ----------
        extent : array_like(Number, shape=(2*self.dim))
            Specifies the points to return. A point p is returned, if
            p <= extent[0:dim] and p >= extent[dim+1:2*dim]

        Returns
        -------
        list of int
            Number of points within the extent.

        See Also
        --------
        box

        """
        return self.r_tree.count(extent)

    def slice(self, min_th=-np.inf, max_th=np.inf, axis=-1):
        """Selects points with coordinate value of axis `axis` within the range
        of [`min_th`, `max_th`].

        Parameters
        ----------
        min_th,max_th : Number
            A point `p` is returned if `min_th <= p[axis] <= max_th`.
        axis : int
            Axis to evaluate.

        Returns
        -------
        list of int
            Indices of points within the slice.

        See Also
        --------
        box, cube, slice

        """
        if not isinstance(min_th, Number) and min_th > 0:
            raise ValueError("'min_th' has to be numeric and greater zero")
        if not isinstance(max_th, Number) and max_th > min_th:
            raise ValueError("'max_th' has to be numeric and greater 'min_th'")
        if not isinstance(axis, int) and abs(axis) < self.dim:
            raise ValueError("'axis' has to be an integer and smaller 'dim'")

        values = self.coords[:, axis]
        order = np.argsort(values)
        iMin = bisect.bisect_left(values[order], min_th) - 1
        iMax = bisect.bisect_left(values[order], max_th)

        # keep original order (for performance reasons of later operations)
        ids = np.sort(order[iMin:iMax])
        return ids
