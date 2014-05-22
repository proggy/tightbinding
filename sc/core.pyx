#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2013-2014 Daniel Jung
# Contact: djungbremen@gmail.com
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
"""Core algorithms of the supercell package, using Cython. To accelerate
and parallelize certain parts of the code."""
# 2012-05-01 - 2012-05-04
import numpy as _np
cimport numpy as _np
from cython.parallel import *
#cimport openmp
import cython

# select my preferred data types
FLT = _np.float64  # my floating point type
ctypedef _np.float64_t FLTt  # compile-time version
INT = _np.int32  # my integer type
ctypedef _np.int32_t INTt  # compile-time version
COM = _np.complex128  # my complex type
ctypedef _np.complex128_t COMt
#BOO = _np.bool8 # my boolean type
#ctypedef _np.bool8_t # compile-time version
#OBJ = object # abbreviation for the Python object type


# in the future, instead of
# <double*>array.data,
# write
# <double*>np.PyArray_DATA(array)! The further is deprecated! At least it is
# probably removed from the Python-C API soon, where ".data" is a field of
# the C struct that is representing the Numpy array.
# (compare http://comments.gmane.org/gmane.comp.python.cython.user/5620)
# Or even better:
# "To access the buffer in Cython, take the address of the first element,
# i.e. &array[0]."
# And in 2D arrays respectively: &array[0, 0]

# Another example:
# cdef np.ndarray[np.uint8_t] x
# cdef void *buffer
# x = array.view(dtype=np.uint8)
# buffer = <void*> &x[0]


@cython.boundscheck(False)
def findinds(positions, vectors):
    """Search for the given vectors in the list of vectors "positions" and
    return the indices they have in that list. Only the first index is returned
    (it is assumed that they are unique in "positions"). If a vector has not
    been found in the list of positions, return -1 for that index.  All
    involved data types are integer. positions and vectors are expected to be
    Numpy arrays with shape n x d and m x d, where d is the number of
    dimensions of the given vectors, n is the number of positions, and m is the
    number of vectors to probe. The result will be a 1D Numpy array of length
    m."""
    # 2012-05-01 - 2012-05-04

    # get and check shapes
    if len(positions.shape) != 2:
        raise ValueError('positions must be a 2D array')
    if len(vectors.shape) != 2:
        raise ValueError('vectors must be a 2D array')
    if positions.shape[1] != vectors.shape[1]:
        raise ValueError('second shape components of positions and vectors ' +
                         'do not match')

    cdef:
        _np.ndarray[int, ndim=2, mode="c"] pos = \
            _np.ascontiguousarray(positions, dtype=_np.int32)
        _np.ndarray[int, ndim=2, mode="c"] vects = \
            _np.ascontiguousarray(vectors, dtype=_np.int32)
        _np.ndarray[int, ndim=1, mode="c"] indices = \
            _np.zeros(vectors.shape[0], dtype=_np.int32)
        int m, n, d, i, j, k, rowsequal

    n = positions.shape[0]
    m = vectors.shape[0]
    d = vectors.shape[1]

    # cycle vectors to probe
    for i in prange(m, nogil=True):
        # cycle positions
        for j in xrange(n):
            # check this position
            #if equalrows(pos, i, vects, j):
            rowsequal = 1
            for k in xrange(d):
                if pos[j, k] != vects[i, k]:
                    rowsequal = 0
                    break
            if rowsequal:
                indices[i] = j
                break
        else:
            indices[i] = -1

    # return resulting indices
    return indices
