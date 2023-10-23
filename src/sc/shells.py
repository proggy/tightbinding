#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2013-2023 Daniel Jung
# Contact: proggy-contact@mailbox.org
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
"""Calculate characteristic vectors of the n-th shell.

A shell is the set of vectors of a simple hypercubic lattice that have the
same length (distance to the chosen origin) and the same characteristic
vector.

For small n, the shell of order n consists of the n-th next neighbors of a
chosen lattice site, but there are deviations from this statement for higher
n because there are shells whose sites have the same distance to the origin,
but have different characteristic vectors. Examples for a 2D lattice are the
shells with the characteristic vectors (5, 0) and (4, 3).

The lattice sites that belong to a certain shell can be calculated by
obtaining all permutations of its characteristic vector including all
possible sign changes. To do that, use the function :func:`signperms`.
"""

import itertools
import numpy


def cvects(order, dim=1):
    """Return characteristic vectors of the shells of a *dim*-dimensional
    simple hypercubic lattice up to the given *order*. The result is a list
    with length *order*+1 of tuples with length *dim*, so the 0-th order is
    always included. All found vectors will be from the "irreducible wedge" of
    the hyperdimensional lattice.

    This iterative algorithm is more intuitive and memory-efficient than the
    common "trial-and-error" method (calculate "far too many" characteristic
    vectors and then sort them afterwards). The shell orders are found in
    ascending sequence, the algorithm can be stopped at any time. Furthermore,
    it works for any dimensionality.

    Possible future improvements:

        - avoid multiple calculation of the same distances
        - break on certain condition, not on fixed order

    Example usage:

        >>> cvects(3, dim=2)  # find neighbors in a 2D lattice up to order 3
        [(0, 0), (1, 0), (1, 1), (2, 0)]
    """
    # check order
    order = int(order)
    if order < 0:
        raise ValueError('illegal order: %i. Must be non-negative integer'
                         % order)

    # check number of dimensions
    dim = int(dim)
    if dim < 1:
        raise ValueError('illegal number of dimensions: %i. Must be ' +
                         'positive integer' % dim)

    # initialize list of vectors with the trivial case of 0th order (origin)
    vects = [(0,)*dim]

    # initialize list holding the current "surface sites" of the wedge
    surface = set(vects)  # in the beginning, the surface consists only of the
                        # origin

    # calculate characteristic vectors up to the requested shell oder
    while len(vects) <= order:
        # cycle the current surface sites and find all outward neighbors
        neighs = set()
        neighcount = dict()
        neighof = dict()
        for surfsite in surface:
            # initialize neighbor count for this surface site
            neighcount[surfsite] = 0

            # find neighbors (try to increment all dimensions by one)
            for d in range(dim):
                # do not leave the irreducibe wedge
                if d > 0 and surfsite[d]+1 > surfsite[d-1]:
                    continue

                # try a possible neighbor site
                trysite = list(surfsite)
                trysite[d] += 1
                trysite = tuple(trysite)

                # exclude surface sites
                if trysite in surface:
                    continue

                # apparently, the site is a valid neighbor site
                # so, add the site to the list of neighbors
                neighs.add(trysite)

                # count number of found neighbors for each surface site
                neighcount[surfsite] += 1

                # remember the neighbors of this site among the surface sites
                if trysite not in neighof:
                    neighof[trysite] = set()
                neighof[trysite].add(surfsite)

        # find the neighbor with the shortest distance to the origin
        next = min(neighs,
                   key=lambda vect: sum(vect[i]**2 for i in range(dim)))
        # it would be nice if all the distances would not have to be calculated
        # again everytime the surface has changed
        vects.append(next)

        # also add the found neighbor to the surface
        surface.add(next)

        # delete sites from the surface that are not needed anymore
        for surfsite in neighof[next]:
            # this is the condition that a site can be deleted. Only then it is
            # not needed anymore as part of the surface
            if neighcount[surfsite] == 1:
                surface.remove(surfsite)

    # return resulting vectors
    return vects


def signperms(tup):
    """Return an (unordered) list of all possible permutations of a given tuple
    *tup* of values, including **all possible sign flips** of the values.  In
    other words, if the characteristic vector of a shell is given, the relative
    vectors pointing to all the lattice sites of the shell are returned. Also
    the number of shell members can obtained easily as the length of the
    result, i.e. ``len(signperms(characteristic_vector))``.

    Example usage:
        >>> signperms((1, 0))
        [(0, 1), (0, -1), (1, 0), (-1, 0)]
        >>> n100 = cvects(100, dim=3)[-1]
        >>> n100  # characteristic vector of 100th shell in a 3D lattice
        (7, 5, 1)
        >>> perms = signperms(n100)
        >>> len(perms)  # number of shell members
        48
        >>> perms[:5]  # just show a few here
        [(-7, -5, -1), (-1, -7, 5), (7, -5, 1), (-5, 7, 1), (7, 5, -1)]
    """
    tup = tuple(tup)
    dim = len(tup)

    # get all possible permutations of input
    perm = numpy.array(list(itertools.permutations(tup)))

    # get all possible sign changes
    signlist = [(-1, 1)]*dim
    signperm = numpy.array(list(itertools.product(*signlist)))

    # calculate all possible permutations including sign changes
    allperm = list((signperm * perm[:, None]).reshape(-1, dim))

    # make a list of tuples out of it
    for ind in range(len(allperm)):
        allperm[ind] = tuple(allperm[ind])

    # return list of tuples, filtering double entries
    return list(set(allperm))


#def in_sphere(radius, dim=1):
  #"""Find the number of points of a uniform grid that are within a sphere
  #with the given radius, if the sphere itself is centred on one of the grid
  #points.
  #"""
  ## get characteristic vectors
  #maxvect = math.ceil(radius)
  #vectors = cvects(maxvect, dim=dim)
