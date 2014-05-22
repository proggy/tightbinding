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
"""Check matrices for symmetry."""
__created__ = '2013-07-25'
__modified__ = '2013-07-25'
import numpy
import scipy.sparse
import dummy

try:
    from frog import Frog
except ImportError:
    Frog = dummy.Decorator


# common frog configuration for all frogs defined here
prolog = 'This frog wrapper of the function uses the supercell definition ' + \
         '(dataset "scell") of the given file and creates a tight-binding ' + \
         'matrix on-the-fly using the method "tbmat".'


@Frog(inmap=dict(mat='$0/scell'), prolog=prolog,
      preproc=dict(mat=lambda scell: scell.tbmat()))
def symmetric(mat):
    """Check if given matrix is symmetric."""
    if scipy.sparse.base.isspmatrix(mat):
        return mat.transpose().todok() == mat.todok()
    else:
        mat = numpy.array(mat)
        return numpy.all(mat.transpose() == mat)


@Frog(inmap=dict(mat='$0/scell'), prolog=prolog,
      preproc=dict(mat=lambda scell: scell.tbmat()))
def hermitian(mat):
    """Check if given matrix is hermitian."""
    if scipy.sparse.base.isspmatrix(mat):
        return mat.transpose().conjugate().todok() == mat.todok()
    else:
        mat = numpy.array(mat)
        return numpy.all(mat.transpose().conjugate() == mat)
