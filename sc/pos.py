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
"""Utilities to choose a list of random positions with the given number of
dimensions (dim) and of the given length (size) from a uniform grid with the
given shape (shape)."""
__created__ = '2012-05-02'
__modified__ = '2014-01-26'
import math
import numpy
import tb
import dist
import progress


class PosRule(dist.Distribution):
    # 2012-05-02 - 2012-05-02
    ### is there anything extra to be done? Or is it just the same as a
    ### distribution object?
    pass


class all(PosRule):
    """Get all positions of a lattice with the given shape. So, there is no
    randomness included, hence this class is (as you may think) very simple in
    nature."""
    # 2012-05-02 - 2014-01-26

    def __init__(self, shape=None, mix=None, num=None):
        """Initialize homogeneous position generator. Set either mixing ratio
        (mix) or fixed number of positions (num). Set lattice shape (shape)."""
        # 2012-05-03 - 2014-01-26

        # set attributes
        self.shape = shape
        self.num = num
        self.mix = mix

        # set additional attributes for __repr__
        self.args = []
        self.kwargs = dict(shape=shape, num=num, mix=mix)

    def __call__(self, distinguish=False):
        """Get all positions as a list. If distinguish is True, nest the
        position list inside a 1-tuple (as the positions cannot be
        distinguished any further in this case)."""
        # 2012-05-03 - 2012-09-03
        return (list(numpy.ndindex(self.shape)),) if distinguish \
            else list(numpy.ndindex(self.shape))


class hom(PosRule):
    """Generate random positions (integer tuples) of a lattice with the given
    shape. It is made sure that no position is generated twice. The positions
    are distributed homogeneously, i.e. every possible position is chosen with
    the same probability.

    Call an instance of this class, providing the shape of the lattice (shape)
    and the number of positions needed (size)."""
    # 2012-05-03 - 2012-09-03

    def __init__(self, shape=None, mix=None, num=None):
        """Initialize homogeneous position generator. Set either mixing ratio
        (mix) or fixed number of positions (num). Set lattice shape (shape)."""
        # 2012-05-03

        # check that either mix or num is set
        if mix is not None and num is not None or mix is None and num is None:
            raise ValueError('exactly one of "mix" and "num" must be None')

        # process mixing ratio
        if mix is not None:
            num = int(mix*numpy.prod(shape))

        # make sure that the lattice is large enough (shape) for the requested
        # number of positions (num)
        if num > numpy.prod(shape):
            raise ValueError('not enough lattice positions for the ' +
                             'requested sample size')

        # set attributes
        self.shape = shape
        self.num = num
        self.mix = mix

        # set additional attributes for __repr__
        self.args = []
        self.kwargs = dict(shape=shape, num=num, mix=mix)

    def __call__(self, distinguish=False):
        """Return random positions (get a new realization of the disordered
        system). If distinguish is True, nest the position list inside a
        1-tuple (as the positions cannot be distinguished any further in this
        case)."""
        # 2012-05-03 - 2012-09-03

        # choose random positions one by one
        ### One could also choose many random indices at once, put them into a
        ### set, and repeat this until the requested number of positions is
        ### reached.
        ### That would presumably have by far better performance
        ### However, it seems to be fast enough in all test cases...
        positions = []
        dim = len(self.shape)
        while len(positions) < self.num:
            # get a random position
            newpos = []
            for d in xrange(dim):
                newpos.append(numpy.random.randint(self.shape[d]))
            newpos = tuple(newpos)

            # check if this position is already in the list
            if newpos not in positions:
                positions.append(newpos)

        # return sorted list of positions
        ### one could also write a generator and use yield
        positions.sort()
        return (positions,) if distinguish else positions


class spheres(PosRule):
    """Generate random positions (integer tuples) of a lattice with the given
    shape under the existence of spherical imhomogeneities.

    It is made sure that no position is generated twice.

    Call an instance of this class, providing the shape of the lattice (shape)
    and the number of positions needed (size)."""
    # 2012-05-03 - 2012-07-06

    def __init__(self, shape=None, rad=1., space=0., sconc=None,
                 iconc=None, iconcin=None, iconcout=None, timeout='10s'):
        """Initialize position generator for spherical inhomogeneities. Set
        lattice shape (shape), radius of spheres (rad) and minimum space
        between the spheres (space). Set exactly 3 of the 4 concentrations
        sconc, iconc, iconcin and iconcout:

        sconc    : concentration of spheres throughout the system
        iconc    : total concentration of impurities within the system
        iconcin  : concentration of impurities inside the spheres
        iconcout : concentration of impurities outside the spheres

        Instead of concentrations (floats between 0 and 1 or strings with
        character %), also numbers (integers) can be given.

        If the given timeout is exceeded and still no valid configuration could
        be found, the program exits.

        Possible extensions:
        --> allow spheres to overlap boundaries (with periodic boundary
            conditions)?"""
        # 2012-07-05 - 2012-07-06

        # check arguments
        if sum([arg is not None
                for arg in [sconc, iconc, iconcin, iconcout]]) != 3:
            raise ValueError('exactly three of sconc, iconc, iconcin and ' +
                             'iconcout must be given, the fourth must be None')

        # calculate how many sites a sphere contains ("sites per sphere")
        sps = self.in_sphere(rad, dim=len(shape))

        # calculate lattice size
        size = numpy.prod(shape)

        # interprete concentrations
        if iconcout is None:
            snum = tb.misc.get_num_from_ratio(sconc, total=size/float(sps),
                                              roundfunc=math.ceil)
            inum = tb.misc.get_num_from_ratio(iconc, total=size)
            inumin = tb.misc.get_num_from_ratio(iconcin, total=sps*snum)
            inumout = inum-inumin
        elif iconcin is None:
            snum = tb.misc.get_num_from_ratio(sconc, total=size/float(sps),
                                              roundfunc=math.ceil)
            inum = tb.misc.get_num_from_ratio(iconc, total=size)
            inumout = tb.misc.get_num_from_ratio(iconcout, total=size-sps*snum)
            inumin = inum-inumout
        elif iconc is None:
            snum = tb.misc.get_num_from_ratio(sconc, total=size/float(sps),
                                              roundfunc=math.ceil)
            inumin = tb.misc.get_num_from_ratio(iconcin, total=sps*snum)
            inumout = tb.misc.get_num_from_ratio(iconcout, total=size-sps*snum)
            inum = inumin+inumout
        elif sconc is None:
            # deducing snum from impurity concentrations not yet possible
            # (solution difficult... but possible?)
            raise NotImplementedError('sphere concentration cannot be None ' +
                                      'at the moment (solution not yet found)')
        else:
            raise ValueError('at least one of sconc, iconc, iconcin, or ' +
                             'iconcout must be None')

        # check integrity of the numbers
        if inum != inumin+inumout:
            raise ValueError('impurity numbers don\'t add up (%i != %i+%i)'
                             % (inum, inumin, inumout))

        # set attributes
        self.shape = shape
        self.rad = rad
        self.space = space
        self.snum = snum
        self.inum = inum
        self.inumin = inumin
        self.inumout = inumout
        self.timeout = timeout

        # set additional attributes for __repr__
        self.args = []
        self.kwargs = dict(shape=shape, rad=rad, space=space, sconc=sconc,
                           iconc=iconc, iconcin=iconcin, iconcout=iconcout,
                           timeout=timeout)

        # provide the "num" attribute
        self.num = inum

    def __call__(self, distinguish=False):
        """Return random positions (get a new realization of the disordered
        system). If distinguish is True, return the lists of positions inside/
        outside the spheres separately (list of two lists)."""
        # 2012-07-05 - 2012-09-03

        # Plan:
        # 1. select the centres of the spheres
        # 2. select random impurity positions inside the spheres
        # 3. select random impurity positions outside the spheres

        # calculate lattice size
        size = numpy.prod(self.shape)

        # select sphere centers (one by one)
        centers = []
        tries = 0  # current try counter (per start with empty centers list)
        started = 1  # how many times it has been started with empty centers
                     # list
        probed = 0  # how many positions have been probed in total
        dim = len(self.shape)
        tout = progress.Until(self.timeout)  # timeout handler
        while len(centers) < self.snum:
            # check timeout
            if tout.check():
                print 'found only %i of %i centers' % (len(centers),
                                                       self.snum),
                print 'at %s' % str(centers),
                print '(started %ix, now at try #%i,' % (started, tries),
                print 'probed %i positions in total)' % probed
                raise ValueError('sphere PosRule timed out (%s) '
                                 % self.timeout)
                # could not find suitable sphere centers... maybe the system is
                # too small for the chosen sphere concentration?

            # check try counter
            if tries > size*dim:  # hopefully a good estimate
                # then start all over
                centers = []
                tries = 0
                started += 1
                continue

            # try counter
            tries += 1
            probed += 1

            # get a random position
            newpos = []
            for d in xrange(dim):
                newpos.append(numpy.random.randint(self.shape[d]))
            newpos = tuple(newpos)

            # exclude positions that are too near to the boundaries
            skip = False
            for d in xrange(dim):
                if newpos[d] < self.rad+self.space/2. \
                        or newpos[d] > self.shape[d]-self.rad-1-self.space/2.:
                    skip = True
                    break
            if skip:
                continue

            # if the sphere overlaps with another sphere, find a new center
            skip = False
            for center in centers:
                if self.distance(newpos, center) <= self.rad*2+self.space:
                    skip = True
                    break
            if skip:
                continue

            # if the new center is valid, append it to the list of centers
            centers.append(newpos)

        centers.sort()
        #print 'the centers are %s' % str(centers)
        #print '(started %ix, now at try #%i,' % (started, tries),
        #print 'probed %i positions in total)' % probed

        # identify all lattice sites that belong to the spheres (index list)
        allsites = list(numpy.ndindex(self.shape))
        spheresites = []
        bulksites = []  # index lists
        for sind, site in enumerate(allsites):
            for center in centers:
                if self.distance(site, center) <= self.rad:
                    spheresites.append(sind)
                    break
            else:
                bulksites.append(sind)
        assert len(spheresites)+len(bulksites) == len(allsites), 'strange...'

        # select impurities inside and outside the spheres
        ininds = numpy.random.permutation(len(spheresites))[:self.inumin]
        inside = list(numpy.array(spheresites)[ininds])
        outinds = numpy.random.permutation(len(bulksites))[:self.inumout]
        outside = list(numpy.array(bulksites)[outinds])

        if distinguish:
            # distinguish different classes of positions, namely those that are
            # inside and those that are outside the spheres
            positions_inside = [allsites[i] for i in inside]
            positions_outside = [allsites[i] for i in outside]

            # return sorted lists of positions
            positions_inside.sort()
            positions_outside.sort()
            return positions_inside, positions_outside
        else:
            # collect all impurity positions
            positions = [allsites[i] for i in inside+outside]

            # return sorted list of positions
            positions.sort()
            return positions

    @staticmethod
    def distance(coord1, coord2):
        """Calculate euclidian distance between the two given coordinates."""
        # 2012-07-06
        coord1 = numpy.array(coord1)
        coord2 = numpy.array(coord2)
        return math.sqrt(numpy.sum((coord1-coord2)**2))

    @staticmethod
    def in_sphere(radius, dim=1):
        """Find the number of points of a dim-dimensional uniform grid that are
        located within a sphere with the given radius, assuming the sphere
        itself is centred on one of the grid points."""
        # 2012-07-05
        gridsize = math.ceil(radius)
        grid = numpy.mgrid[(slice(-gridsize, gridsize+1),)*dim]
        return numpy.sum(numpy.sum(grid**2, axis=0) <= radius**2)
