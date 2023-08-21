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
"""Tight binding package. Define tight binding systems (see submodule
:mod:`tightbinding.define`) as instances of the class :class:`tightbinding.sc.SuperCell` and create
the dynamic matrix (tight-binding matrix) using its method
:py:meth:`tightbinding.sc.SuperCell.tbmat`.
"""
__version__ = '0.1.0'

from tightbinding import dummy

try:
    from comliner import Comliner
except ImportError:
    Comliner = dummy.Decorator


@Comliner(inmap=dict(scell='$0/scell'))
def tbmat(scell, format=None):
    """Calculate the tight binding matrix of the given supercell definition
    *scell*.
    """
    return scell.tbmat(format=format)
