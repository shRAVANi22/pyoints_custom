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
"""Handles spatial extents.
"""

import numpy as np

from . import assertion

from . misc import print_rounded


class Extent(np.recarray, object):
    """Specifies spatial extent (or bounding box) of coordinates in `k`
    dimensions.

    Parameters
    ----------
    ext : array_like(Number, shape=(2 * k)) or array_like(Number, shape=(n, k))
        Defines spatial extent of `k` dimensions as either minimum corner and
        maximum corner or as a set of `n` points. If a set of points is given,
        the bounding box of these coordinates is calculated.

    Attributes
    ----------
    dim : positive int
        Number of coordinate dimensions.
    ranges : np.ndarray(Number, shape=(self.dim))
        Ranges between each coordinate dimension.
    min_corner,max_corner : array_like(Number, shape=(self.dim))
        Minimum and maximum values in each coordinate dimension.
    center : array_like(Number, shape=(self.dim))
        Focal point of the extent.

    Examples
    --------

    Derive the extent of a list of points.

    >>> points = [(0, 0), (1, 4), (0, 1), (1, 0.5), (0.5, 0.7)]
    >>> ext = Extent(points)
    >>> print(ext)
    [ 0.  0.  1.  4.]

    Create a extent based on minimum and maximum values.

    >>> ext = Extent([-1, 0, 1, 4, ])
    >>> print(ext)
    [-1  0  1  4]

    Get some properties.

    >>> print_rounded(ext.dim)
    2
    >>> print_rounded(ext.min_corner)
    [-1  0]
    >>> print_rounded(ext.max_corner)
    [1 4]
    >>> print_rounded(ext.ranges)
    [2 4]
    >>> print(ext.center)
    [ 0.  2.]
    >>> print_rounded(ext.corners)
    [[-1  0]
     [ 1  0]
     [ 1  4]
     [-1  4]]

    """
    def __new__(cls, ext):

        ext = assertion.ensure_numarray(ext)

        if not len(ext.shape) <= 2:
            raise ValueError('vector or coordinates needed')

        if len(ext.shape) == 2:
            # coordinates
            min_ext = np.amin(ext, axis=0)
            max_ext = np.amax(ext, axis=0)
            ext = np.concatenate((min_ext, max_ext))
        else:
            # vector
            dim = len(ext) // 2
            if not dim * 2 == len(ext):
                raise ValueError('malformed extent vector')
            if not np.all(ext[:dim] <= ext[dim:]):
                raise ValueError('minima must not be greater than maxima')

        return ext.view(cls)

    @property
    def dim(self):
        return len(self) // 2

    @property
    def ranges(self):
        return self.max_corner - self.min_corner

    @property
    def min_corner(self):
        return self[:self.dim]

    @property
    def max_corner(self):
        return self[self.dim:]

    @property
    def center(self):
        return (self.max_corner + self.min_corner) * 0.5

    def split(self):
        """Splits the extent into the minimum and maximum corners.

        Returns
        -------
        min_corner,max_corner : np.ndarray(Number, shape=(self.dim))
            Minimum and maximum values in each coordinate dimension.
        """
        return self.min_corner, self.max_corner

    @property
    def corners(self):
        """Provides each corner of the extent box.

        Returns
        -------
        corners : np.ndarray(Number, shape=(2\*\*self.dim, self.dim))
            Corners of the extent.

        Examples
        --------

        Two dimensional case.

        >>> ext = Extent([-1, -2, 1, 2])
        >>> print_rounded(ext.corners)
        [[-1 -2]
         [ 1 -2]
         [ 1  2]
         [-1  2]]

        Three dimensional case.

        >>> ext = Extent([-1, -2, -3, 1, 2, 3])
        >>> print_rounded(ext.corners)
        [[-1 -2 -3]
         [ 1 -2 -3]
         [ 1  2 -3]
         ..., 
         [ 1  2  3]
         [ 1 -2  3]
         [-1 -2  3]]

        """
        def combgen(dim):
            # generates order of corners
            if dim == 1:
                return np.array([[0, 1]], dtype=int).T
            else:
                comb = combgen(dim - 1)
                col = np.array([np.hstack((
                    np.zeros(len(comb)),
                    np.ones(len(comb)),
                ))], dtype=int).T
                comb = np.vstack((comb, comb[::-1, :]))
                return np.hstack((comb, col))

        combs = combgen(self.dim)
        combs = combs * self.dim + range(self.dim)
        return self[combs]

    def intersection(self, coords, dim=None):
        """Tests if coordinates are located within the extent.

        Parameters
        ----------
        coords : array_like(Number, shape=(n, k)) or
        array_like(Number, shape=(k))
            Represents `n` data points of `k` dimensions.
        dim : positive int
            Desired number of dimensions to consider.

        Returns
        -------
        indices : np.ndarray(int, shape=(n)) or np.ndarray(bool, shape=(n))
            Indices of coordinates which are within the extent. If just a
            single point is given, a boolean value is returned.

        Examples
        --------

        Point within extent?

        >>> ext = Extent([0, 0.5, 1, 4])
        >>> print(ext.intersection([(0.5, 1)]))
        True

        Points within extent?

        >>> print_rounded(ext.intersection([(1, 2), (-1, 1), (0.5, 1)]))
        [0 2]

        Corners are located within the extent.

        >>> print_rounded(ext.intersection(ext.corners))
        [0 1 2 3]

        """
        # normalize data
        coords = assertion.ensure_numarray(coords)
        if len(coords.shape) == 1:
            coords = np.array([coords])

        # set desired dimension
        dim = self.dim if dim is None else dim
        if not dim > 0:
            raise ValueError('dimension "dim" needs to be greater zero')

        # check
        n, c_dim = coords.shape
        if not c_dim <= self.dim:
            m = 'expected %i dimensions, but got %i'
            raise ValueError(m % (self.dim, c_dim))

        min_ext, max_ext = self.split()

        # Order axes by range to speed up the process (heuristic)
        order = np.argsort(self.ranges[0:dim])
        mask = np.any(
            (np.abs(min_ext[order]) < np.inf, np.abs(max_ext[order]) < np.inf),
            axis=0
        )
        axes = order[mask]

        indices = np.arange(n)
        for axis in axes:
            values = coords[indices, axis]

            # Minimum
            mask = values >= min_ext[axis]
            indices = indices[mask]
            if len(indices) == 0:
                break
            values = values[mask]

            # Maximum
            mask = values <= max_ext[axis]
            indices = indices[mask]
            if len(indices) == 0:
                break
            values = values[mask]

        if n == 1:
            return len(indices) == 1

        return indices
