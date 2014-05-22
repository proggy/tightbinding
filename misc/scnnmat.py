#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2011-2014 Daniel Jung
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
"""Implements a standalone version of the function scnnmat.  The original is
included in the submodule tb.sc of the tb package."""
# 2011-01-03 -2014-02-08
import numpy as np
import scipy.sparse as spa


def scnnmat(dims, pot=0., hop=1., bcond='p', format='lil'):
    """Return Hamiltonian matrix of a simple dim-dimensional tight-binding
    system with dimensions "dims" (tuple of positive integers of length dim),
    site potentials "pot" (float or 1D-array of floats with length size =
    prod(dims)), constant isotropic next-neighbor hopping "hop" (float) and
    boundary conditions "bcond" (string consisting of characters "s", "p" and
    "a").  Return matrix in LIL-sparse format.

    bcond may be at most of length dim. In this way, each dimension can have
    it's own boundary condition."""
    # 2011-02-28

    # check arguments
    assert format \
        in ['lil', 'dok', 'csr', 'csc', 'dia', 'coo', 'bsr', 'dense'], \
        'unknown format. Must be one of lil, dok, coo, dia, csr, csc, bsr ' + \
        'or dense'
    if isiterable(dims):
        dims = tuple(dims)
    else:
        dims = (dims,)
    #dims = dims[::-1]
    size = np.prod(dims, dtype=int)
    dim = len(dims)
    assert dim > 0, 'bad dimensions tuple: Must have at least one element'

    # Check boundary condition string
    assert len(bcond) <= dim, \
        'bad boundary condition: %s. Number of given boundary conditions ' + \
        'is greater than dimensionality of the system (%i)' % (bcond, dim)
    assert len(bcond) > 0, \
        'bad boundary condition: May not be empty string. At least one ' + \
        'character (s, p or a) has to be given'
    if len(bcond) < dim:
        bcond += bcond[-1]*(dim-len(bcond))

    # Method:
    # 1. Build matrix of a system with dimensions dims[:-1] (recursive call of
    #    the function), set the dims[-1] blocks with it using kronecker product
    #    (scipy.sparse.kron)
    # 2. Add the matrix elements of the remaining 1D-subsystem using setdiag
    # That's it!

    # Step 1
    subdims = dims[:-1]
    subsize = np.prod(subdims, dtype=int)
    if subsize > 1:
        # calculate submatrix recursevely
        mat = spa.kron(spa.eye(dims[-1], dims[-1]),
                       scnnmat(subdims, hop=hop, bcond=bcond[:-1]),
                       format='lil')
    else:
        # initialize matrix
        mat = spa.lil_matrix((size, size))

    # Step 2
    # If periodic or anti-periodic boundary conditions are given, set
    # respective elements as well. Do this first, because they may be
    # overwritten by the direct hoppings in the next step.
    if bcond[-1] in ['p', 'a']:
        if bcond[-1] == 'p':
            bcondhoparray = np.ones((subsize))*(-hop)
        else:
            bcondhoparray = np.ones((subsize))*hop
        mat.setdiag(bcondhoparray, size-subsize)
        mat.setdiag(bcondhoparray, subsize-size)
    elif bcond[-1] == 's':
        # do not set anything
        pass
    else:
        raise ValueError('bad boundary condition: %s. Expecting either "s" ' +
                         '(static), "p" (periodic) or "a" (antiperiodic)'
                         % bcond[-1])

    # set off-diagonals with hopping
    hoparray = np.ones((size-subsize))*(-hop)
    mat.setdiag(hoparray, subsize)
    mat.setdiag(hoparray, -subsize)

    # set diagonal elements with potential
    if isiterable(pot):
        pot = np.array(pot)
        assert len(pot.shape) == 1, \
            'potentials have wrong shape. Must be 1D-array-like'
        assert len(pot) == size, \
            'wrong number of potentials: Expecting %i, but got %i' \
            % (size, len(pot))
        mat.setdiag(pot)
    else:
        if pot != 0.:  # this step can be omitted if potentials are zero anyway
            mat.setdiag(np.ones((size))*pot)

    # convert matrix if needed
    if format != 'lil':
        mat = getattr(mat, 'to'+format)()

    # return matrix
    return mat


def isiterable(obj):
    """Checks if an object is iterable. Returns True for lists, tuples and
    dictionaries. Returns False for scalars (float, int, etc.), strings, bool
    and None."""
    # 2011-01-27
    # Inicial idea from:
    # http://bytes.com/topic/python/answers/514838-how-test-if-object-sequence-
    # iterable
    #return isinstance(obj, basestring) or getattr(obj, '__iter__', False)
    # I found this to be better:
    return not getattr(obj, '__iter__', False) is False


# example call, if this module is executed directly
if __name__ == '__main__':
    print """Create Hamiltonian matrix of a 2D simple-cubic tight-binding
    system with all site-diagonal entries (potentials) pot=2 and hopping
    parameter hop=1. System dimensions: 4 x 3."""
    hamilt = scnnmat((4, 3), pot=2, hop=1)

    print 'Convert to dense format and print to screen:'
    print hamilt.todense()

    print """Create the same matrix, but with anti-periodic boundary conditions
    in both dimensions. Note that some of the hopping matrix elements have
    switched their sign:"""
    print scnnmat((4, 3), hop=1, pot=2, bcond='a', format='dense')
