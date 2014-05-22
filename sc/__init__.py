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
# with this prograk; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
"""Package to define tight binding supercells and calculate their tight
binding matrices (hamiltonians) in site-occupation basis. Start by creating an
instance of the SuperCell class. The methods SuperCell.add_scnn and
SuperCell.add_fccnn (and others) give good examples how to define lattices
within the supercell.

The submodule "dist" contains everything needed to generate random numbers from
various probability distributions, including a few predefined distributions.

Future visions:
--> supercell itself may have (anti)periodic boundary conditions (until now,
    just lattices have boundary conditions)
--> improve Lattice.ndindex2index (increase performance even more by using a
    lookup table)
--> add special algorithms for special cases (e.g., Lattice.scnnmat)
--> add automatic performance checks (compare times for finding the connections
    to the setting of matrix elements, random and constant)
--> rename "dim" to "ndim", be consistent with numpy.ndarray"""
__created__ = '2011-08-17'
__modified__ = '2014-01-26'
import math
import itertools
import scipy
import string
import core
import dist
import pos
import shells
from frog import Frog


class SuperCellObject(object):
    """Abstract base class for almost all other classes defined in this module.
    There are certain methods that are shared by all classes used to define
    the supercell, so to avoid redundancy, those methods are collected here in
    this abstract base class."""
    # 2011-11-19

    def __repr__(self):
        """Return short string representation."""
        # 2011-09-12
        # former sc.SuperCell.__repr__ from 2011-08-20 - 2011-08-26
        return '<'+self.label+'>'

    def __str__(self):
        """Return detailed string representation."""
        # 2011-09-12
        # former sc.SuperCell.__repr__ from 2011-08-20 - 2011-08-29
        attr = ', '.join('%s=%s' % (key, repr(value))
                         for key, value in self.__dict__.iteritems())
        return '<'+self.__class__.__name__+'('+attr+')>'

    def infertype(self):
        """Return data type (complex or float) that is needed due to the
        defined potentials and hoppings within this object. Depends on the
        method is_complex that has to be defined within each child class."""
        # 2011-09-12
        # former sc.Lattice.infertype from 2011-08-25
        return complex if self.is_complex() else float

    def _ucletters(self, s):
        """Return all uppercase letters of the given string."""
        # 2011-09-12 - 2012-06-19
        # former sc.SuperCell._ucletters from 2011-08-26
        result = ''
        for letter in s:
            if letter in string.uppercase:
                result += letter
        return result

    @staticmethod
    def ismath(string):
        """Check if the given string is a mathematical expression (containing
        only mathematical operators like '+', '-', '*', or '/', and of course
        digits). Can be used before using "eval" on some string to evaluate a
        given expression.

        Note: This function does not check if the numerical expression is
        actually valid. It just gives a hint if the given string should be
        passed to eval or not."""
        # 2011-09-13 - 2011-10-12
        # former tb.mathexpr from 2011-06-12
        if '+' in string or '*' in string or '/' in string:
            return True

        # special handling of the minus sign
        if string.count('-') == 1 and string[0] == '-':
            return False
        if '-' in string:
            return True
        return False

    def evalmath(self, value, dtype=float):
        """Cast value to the given data type (dtype). If value is a string,
        assume that it contains a mathematical expression, and evaluate it with
        eval before casting it to the specified type. Functions are passed
        unmodified.

        The function could always use eval, but this is assumed to be slower
        for values that do not have to be evaluated."""
        # 2011-10-12 - 2011-12-14
        if hasattr(value, '__call__'):
            # bypass function or callable class objects (defining a certain
            # probability distribution)
            return value
        elif isinstance(value, str):
            if self.ismath(value):
                return dtype(eval(value))
            else:
                return dtype(value)
        else:
            return dtype(value)


class SuperCell(SuperCellObject):
    """Define tight binding supercell, using an object-oriented user interface.
    Create multiple sites or whole lattices of sites (repeat sites within a
    given unitcell periodically). Each site may hold several entities
    (potentials)."""
    # 2011-08-17 - 2012-09-04
    # former tb.SuperCell from 2011-03-05 - 2011-08-02

    # initialize instance counter
    count = 0

    def __init__(self, dim=1, margin=None, label=None, sites=None,
                 lats=None, hops=None, eps=.1):  # bcond=None
        """Initialize supercell."""
        # 2011-08-17 - 2011-10-12

        # number of dimensions
        self.dim = self.evalmath(dim, dtype=int)
        assert self.dim > 0, 'number of dimensions must be positive integer'

        # boundary conditions
        # yet to come for the supercell itself. So far, boundary conditions are
        # only supported for single lattices within the supercell
        #if bcond is None:
            #self.bcond = 'p'*self.dim
        #else:
            #self.bcond = str(bcond)
            #assert bcond != '', 'no boundary conditions given'
            #while len(self.bcond) < self.dim:
                #self.bcond += self.bcond[-1]

        # set label
        self.__class__.count += 1
        if label is None:
            label = '%s%i' % (self._ucletters(self.__class__.__name__),
                              self.__class__.count)
        self.label = label

        # set margin
        if margin is None:
            margin = (0.)*self.dim
        self.margin = margin

        # set tolerance in finding the distances between sites
        self.eps = self.evalmath(eps)

        # data structures
        self.sites = [] if sites is None else sites
        self.lats = [] if lats is None else lats
        self.hops = {} if hops is None else hops

    def ents(self):
        """Return generator of all entities of all the sites of all the
        lattices and all the single sites of this supercell."""
        # 2011-08-31
        for lat in self.lats:
            for ent in lat.unitcell.ents():
                yield ent
        for site in self.sites:
            for ent in site.ents:
                yield ent

    def add_site(self, *sites, **kwargs):
        """Add a site to the supercell at the given coordinates (coord). For a
        complete list of possible arguments, look at the definition of
        Site.__init__.

        Also, predefined site objects may be given as positional arguments.
        Then, no keyword arguments may be given."""
        # 2011-08-23 - 2012-05-01

        kwargs.update(dim=self.dim)
        if len(sites) == 0:
            site = Site(**kwargs)
            self.sites.append(site)
            return site
        else:
            if len(kwargs) > 0:
                raise KeyError('no keyword arguments allowed if ' +
                               'positional arguments are given')
            for site in sites:
                if site not in self.sites:
                    self.sites.append(site)

    def add_lat(self, *lats, **kwargs):
        """Add a lattice to the supercell by defining a unitcell that may
        contain multiple sites and that is repeated periodically in space a
        given number of times in each dimension (shape).

        Set the origin and the basis vectors (bvects) to control where and how
        the lattice stretches within the supercell.

        Also, predefined lattice objects may be given as positional arguments.
        Then, no keyword arguments may be given."""
        # 2011-08-17 - 2012-05-01

        kwargs.update(dim=self.dim)
        if len(lats) == 0:
            lat = Lattice(**kwargs)
            self.lats.append(lat)
            return lat
        else:
            if len(kwargs) > 0:
                raise KeyError('no keyword arguments allowed if positional ' +
                               'arguments are given')
            for lat in lats:
                if lat not in self.lats:
                    self.lats.append(lat)

    def add_sparselat(self, *lats, **kwargs):
        """Add a sparse lattice to the supercell by defining a unitcell that
        may contain multiple sites and that is placed on only a few of the
        lattice sites within the given shape of the lattice.

        Set the origin and the basis vectors (bvects) to control where and how
        the lattice stretches within the supercell.

        Also, predefined lattice objects may be given as positional arguments.
        Then, no keyword arguments may be given."""
        # 2012-05-01
        # based on tb.sc.add_lat from 2011-08-17 - 2012-05-01

        kwargs.update(dim=self.dim)
        if len(lats) == 0:
            lat = SparseLattice(**kwargs)
            self.lats.append(lat)
            return lat
        else:
            if len(kwargs) > 0:
                raise KeyError('no keyword arguments allowed if positional ' +
                               'arguments are given')
            for lat in lats:
                if lat not in self.lats:
                    self.lats.append(lat)

    def add_hop(self, *ents, **kwargs):
        """Add a hopping between the given entities, additionally restricted to
        a certain distance (delta) that must exist between the coordinates of
        the sites (within a certain tolerance) where the given entities are
        located.  The hopping parameter (hop) can either be given by a constant
        or be drawn from a random distribution. If iso is True, also hopping in
        the opposite direction will be allowed."""
        # 2011-08-31 - 2012-04-05

        # make sure all entities exist at this site
        for ent in ents:
            assert ent in self.ents(), \
                'Entity %s does not exist at this site' % repr(ent)

        # handle multiple entities
        assert len(ents) > 1, 'at least two entities have to be given'
        if len(ents) > 2:
            for ent1, ent2 in itertools.combinations(ents, 2):
                self.add_hop(ent1, ent2, **kwargs)
        else:
            ent1, ent2 = ents

            # get keyword arguments
            delta = kwargs.get('delta', 1.)
            hop = kwargs.get('hop', -1.)
            iso = kwargs.get('iso', False)

            # check if the two entities are identical
            if ent1 == ent2 and delta == 0.:
                # then just redefine the potential of that entity
                ent1.pot = hop
                return

            # constant or random?
            if hasattr(hop, '__call__'):
                # random hopping
                self.hops[ent1, ent2, delta] = hop
            else:
                # constant hopping
                if hop == 0. and (ent1, ent2, delta) in self.hops:
                    del(self.hops[ent1, ent2, delta])
                else:
                    self.hops[ent1, ent2, delta] = hop

            # isotropic hopping?
            if iso:
                self.add_hop(ent2, ent1, delta=delta, hop=hop.conjugate(),
                             iso=False)
                ### What happens if hop is random and iso is True?

    def tbmat(self, format=None, distinguish=False):
        """Return tight binding matrix of this supercell, convert to the
        specified format (one of dense, csr, csc, dok, coo, dia, bsr, or lil).
        If all is True, recalculate the whole matrix, do not return a cached
        version. If random is True, recalculate only random matrix elements,
        i.e.  get a new realization of disorder. If cache is True, always
        return the cached version of the matrix (if there is no cached version,
        raise an exception). If all, random and cache are all False, calculate
        the matrix only if there is no cached version, otherwise just return
        the cache."""
        # 2011-09-01 - 2012-09-04

        # first, fill all blocks on the main diagonal, containing all
        # potentials and intra-lattice and intra-site hoppings
        #print time.ctime(), \
        #'create matrix of supercell and set diagonal blocks...'
        stateclasses = None
        nblocks = len(self.lats)+len(self.sites)  # number of blocks
        blocks = scipy.array([None]*nblocks**2).reshape(nblocks, -1).tolist()
        for ind, obj in enumerate(self.lats+self.sites):
            if isinstance(obj, SparseLattice):
                # only SparseLattice.tbmat knows the keyword "distinguish"
                blocks[ind][ind], stateclasses = obj.tbmat(distinguish=True)
            else:
                blocks[ind][ind] = obj.tbmat()
        mat = scipy.sparse.bmat(blocks).astype(self.infertype()).todok()
        #print time.ctime(), 'matrix of supercell created, diagonal blocks set'

        # add hoppings, cycle all hopping definitions
        #print time.ctime(), 'add supercell hoppings...'
        for (ent1, ent2, delta), hop in self.hops.iteritems():
            # get all indices and coordinates of the two entities
            einds1, coords1 = self.find_ent(ent1, coord=True)
            einds2, coords2 = self.find_ent(ent2, coord=True)
            ### change find_ent, improve performance even more

            # instanciate kd-trees
            tree1 = scipy.spatial.KDTree(coords1)
            tree2 = scipy.spatial.KDTree(coords2)

            # calculate distance matrix (sparse, dictionary-of-keys format)
            dismat = tree1.sparse_distance_matrix(tree2,
                                                  max_distance=delta+self.eps)

            # filter distances, find those that are greater than delta-eps
            inds = scipy.flatnonzero(scipy.array(dismat.values()) >
                                     delta-self.eps)
            if len(inds) == 0:
                continue
            inds1, inds2 = scipy.array(dismat.keys())[inds].T

            if hasattr(hop, '__call__'):
                # generate random variates
                data = hop(size=len(inds1))
            else:
                data = scipy.ones(len(inds1))*hop

            # add this hopping to the big matrix
            helper = scipy.sparse.coo_matrix((data,
                                              (einds1[inds1], einds2[inds2])),
                                             shape=mat.shape)
            mat.update(helper.todok())

            # another, presumably slower approach, using nested Python
            # for-loops select entities, together with their coordinates and
            # indices
            #for inds1, coords1 in self.loop_ent(ent1, coord=True):
                #for inds2, coords2 in self.loop_ent(ent2, coord=True):
                    #dist2 = sum((coords1-coords2)**2)
                    #if dist2 >= delta-tol/2 and dist <= delta+tol/2:
                    ### better: allclose
        #print time.ctime(), 'supercell hoppings added'

        # return the matrix
        if distinguish:
            if stateclasses is None:
                stateclasses = (range(mat.shape[0]),)
            return mat.asformat(format), stateclasses
        else:
            return mat.asformat(format)

    def is_symmetric(self):
        """Check if tight binding matrix is symmetric."""
        # 2012-02-02
        # Convert the matrix to the dok-format before comparison
        mat = self.tbmat()  # cache=True
        return mat.transpose().todok() == mat.todok()

    def is_hermitian(self):
        """Check if tight binding matrix is hermitian."""
        # 2012-02-02
        # Convert the matrix to the dok-format before comparison
        mat = self.tbmat()  # cache=True
        return mat.transpose().conjugate().todok() == mat.todok()

    def find_ent(self, ent, coord=False):
        """Return indices (within the big matrix) that belong to the given
        entity.  If coords is True, also return the absolute coordinates
        (within the supercell)."""
        # 2011-09-02 - 2011-09-05

        # initialize offset for matrix index
        offset = 0

        # initialize data collectors
        inds = []
        coords = []

        # search lattices
        for lat in self.lats:
            # get number of entities specified within the unitcell of the
            # lattice
            enum = len(list(lat.unitcell.ents()))

            for eind, (e, site) in enumerate(lat.unitcell.ents(sites=True)):
                if e == ent:
                    for ucind, uccoord in enumerate(scipy.ndindex(lat.shape)):
                        inds.append(offset+ucind*enum+eind)
                        if coord:
                            origin = scipy.array(lat.origin)
                            bvects = scipy.array(lat.bvects)
                            uccoord = scipy.array(uccoord)
                            sitecoord = scipy.array(site.coord)
                            coords.append(tuple(origin+scipy.dot(bvects,
                                                                 uccoord +
                                                                 sitecoord)))
                            #yield offset+ucind*enum+eind, \
                                    #origin+scipy.dot(bvects, uccoord+coord)
                        #else:
                            #yield offset+ucind*enum+eind

                    # double entries would not make any sense
                    break

            # increase offset
            offset += enum*lat.size()

        # search single sites
        for site in self.sites:
            # Get number of entities specified on the site
            enum = len(site.ents)

            for eind, e in enumerate(site.ents):
                if e == ent:
                    inds.append(offset+eind)
                    if coord:
                        coords.append(site.coord)
                        #yield offset+eind, scipy.array(site.coord)
                    #else:
                        #yield offset+eind

                    # Double entries would not make any sense
                    break

            # increase offset
            offset += enum

        # return results
        if coord:
            return scipy.array(inds), scipy.array(coords)
        else:
            return scipy.array(inds)

    def add_scnn(self, pot=0., hop=-1., shape=None, bcond=None, origin=None,
                 bvects=None, label=None):
        """Create a dim-dimensional 1-band simple cubic lattice with isotropic
        next-neighbor hopping (hop) only."""
        # 2011-08-28 - 2011-11-09
        # former tb.SuperCell.add_scnn
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)
        site = lat.unitcell.add_site()
        ent = site.add_ent(pot=pot)
        nei = lat.add_neigh()
        nei.add_vect(permall=True)
        nei.add_hop(ent, ent, hop=hop)
        return lat

    def add_triang(self, pot=0., hop=-1., shape=None, bcond=None, origin=None,
                   bvects=None, label=None):
        """Create a 2-dimensional 1-band triangular lattice with isotropic
        next-neighbor hopping (hop) only."""
        # 2011-12-11
        ### bvects should be chosen to get lattice with equilateral triangles
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)
        site = lat.unitcell.add_site()
        ent = site.add_ent(pot=pot)
        nei = lat.add_neigh()
        nei.add_vect((1, 0), permall=True)
        nei.add_vect((1, -1))
        nei.add_vect((-1, 1))
        nei.add_hop(ent, ent, hop=hop)
        return lat

    def add_fccnn(self, pot=0., hop=-1., shape=None, bcond=None, origin=None,
                  bvects=None, label=None):  # pdist=None, hdist=None,
                                             # ploc=0., pscale=1., hloc=-1.,
                                             # hscale=1.
        """Create a 3-dimensional 1-band face-centered cubic lattice with
        isotropic next-neighbor hopping (hop) only."""
        # 2011-08-28 - 2011-11-09
        # former tb.SuperCell.add_fccnn
        assert self.dim == 3, 'a fcc lattice can only be created in a ' + \
                              '3-dimensional supercell'

        # create lattice
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)

        # add the four sites that exist in the conventional unitcell of a fcc
        # lattice
        siteA = lat.unitcell.add_site(coord=(0., 0., 0.), label='Site A')
        siteB = lat.unitcell.add_site(coord=(.5, .5, 0.), label='Site B')
        siteC = lat.unitcell.add_site(coord=(.5, 0., .5), label='Site C')
        siteD = lat.unitcell.add_site(coord=(0., .5, .5), label='Site D')

        # add a potential to each of the sites
        A = siteA.add_ent(pot=pot, label='A')  # dist=pdist, loc=ploc,
                                               # scale=pscale
        B = siteB.add_ent(pot=pot, label='B')
        C = siteC.add_ent(pot=pot, label='C')
        D = siteD.add_ent(pot=pot, label='D')

        # define intra-unitcell hoppings
        lat.unitcell.add_hop(A, B, hop=hop, iso=True)  # dist=hdist, loc=hloc,
                                                       # scale=hscale
        lat.unitcell.add_hop(A, C, hop=hop, iso=True)
        lat.unitcell.add_hop(A, D, hop=hop, iso=True)
        lat.unitcell.add_hop(B, C, hop=hop, iso=True)
        lat.unitcell.add_hop(B, D, hop=hop, iso=True)
        lat.unitcell.add_hop(C, D, hop=hop, iso=True)

        # define neighbors
        n001 = lat.add_neigh(label='N001')
        n010 = lat.add_neigh(label='N010')
        n100 = lat.add_neigh(label='N100')
        n011 = lat.add_neigh(label='N011')
        n101 = lat.add_neigh(label='N101')
        n110 = lat.add_neigh(label='N110')
        n01_1 = lat.add_neigh(label='N01_1')  # the underscore stands for minus
        n10_1 = lat.add_neigh(label='N10_1')
        n1_10 = lat.add_neigh(label='N1_10')

        # set the vectors for each neighbor that lead to the respective
        # neighboring unitcells
        n001.add_vect((0, 0, 1), iso=True)
        n010.add_vect((0, 1, 0), iso=True)
        n100.add_vect((1, 0, 0), iso=True)
        n011.add_vect((0, 1, 1), iso=True)
        n101.add_vect((1, 0, 1), iso=True)
        n110.add_vect((1, 1, 0), iso=True)
        n01_1.add_vect((0, 1, -1), iso=True)
        n10_1.add_vect((1, 0, -1), iso=True)
        n1_10.add_vect((1, -1, 0), iso=True)

        # define neighbor hoppings between A and B
        n010.add_hop(B, A, hop=hop)  # dist=hdist, loc=hloc, scale=hscale
        n100.add_hop(B, A, hop=hop)
        n110.add_hop(B, A, hop=hop)

        # define neighbor hoppings between A and C
        n100.add_hop(C, A, hop=hop)
        n001.add_hop(C, A, hop=hop)
        n101.add_hop(C, A, hop=hop)

        # define neighbor hoppings between A and D
        n010.add_hop(D, A, hop=hop)
        n001.add_hop(D, A, hop=hop)
        n011.add_hop(D, A, hop=hop)

        # define neighbor hoppings between B and C
        n001.add_hop(C, B, hop=hop)
        n010.add_hop(B, C, hop=hop)
        n01_1.add_hop(B, C, hop=hop)

        # define neighbor hoppings between B and D
        n100.add_hop(B, D, hop=hop)
        n10_1.add_hop(B, D, hop=hop)
        n001.add_hop(D, B, hop=hop)

        # define neighbor hoppings between C and D
        n100.add_hop(C, D, hop=hop)
        n1_10.add_hop(C, D, hop=hop)
        n010.add_hop(D, C, hop=hop)

        # return reference to this newly created lattice object
        return lat

    def add_bccnn(self, pot=0., hop=-1., shape=None, bcond=None, origin=None,
                  bvects=None, label=None):  # pdist=None, ploc=0., pscale=1.,
                                             # hdist=None, hloc=-1., hscale=1.
        """Create a 3-dimensional 1-band body-centered cubic lattice with
        isotropic next-neighbor hopping (hop) only."""
        # 2011-08-28 - 2011-11-09
        # former tb.SuperCell.add_bccnn
        assert self.dim == 3, 'a bcc lattice can only be created in a ' + \
                              '3-dimensional supercell'

        # create lattice
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)

        # add the two sites that exist in the conventional unitcell of a bcc
        # lattice
        siteA = lat.unitcell.add_site(coord=(0., 0., 0.), label='Site A')
        siteB = lat.unitcell.add_site(coord=(.5, .5, .5), label='Site B')

        # add a potential to each of the sites
        A = siteA.add_ent(pot=pot, label='A')  # dist=pdist, loc=ploc,
                                               # scale=pscale
        B = siteB.add_ent(pot=pot, label='B')

        # define intra-unitcell hoppings
        lat.unitcell.add_hop(A, B, hop=hop, iso=True)  # dist=hdist, loc=hloc,
                                                       # scale=hscale

        # define next-neighbor hoppings
        nei = lat.add_neigh()
        nei.add_vect((1, 0, 0), iso=True, perm=True)
        nei.add_vect((1, 1, 0), iso=True, perm=True)
        nei.add_vect((1, 1, 1), iso=True)
        nei.add_hop(B, A, hop=hop, iso=True)  # dist=hdist, loc=hloc,
                                              # scale=hscale

        # return reference to this newly created lattice object
        return lat

    def add_scsnn(self, pot=0., hop1=-1., hop2=-1., shape=None, bcond=None,
                  origin=None, bvects=None, label=None):  # pdist=None,
                                                          # ploc=0., pscale=1.,
                                                          # h1dist=None,
                                                          # h1loc=-1.,
                                                          # h1scale=1.,
                                                          # h2dist=None,
                                                          # h2loc=-1.,
                                                          # h2scale=1.
        """Create a dim-dimensional 1-band simple cubic lattice with isotropic
        next-neighbor (hop1) and second-next-neighbor hopping (hop2)."""
        # 2011-08-28 - 2011-11-09

        # create lattice
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)

        # add a site to the unitcell of the lattice
        site = lat.unitcell.add_site()

        # add a potential to the site
        ent = site.add_ent(pot=pot)  # dist=pdist, loc=ploc, scale=pscale

        # define next neighbor hoppings
        nei1 = lat.add_neigh()
        nei1.add_vect(permall=True)
        nei1.add_hop(ent, ent, hop=hop1)  # dist=h1dist, loc=h1loc,
                                          # scale=h1scale

        # define second-next neighbor hoppings
        nei2 = lat.add_neigh()
        if self.dim == 1:
            nei2.add_vect((2,), iso=True)
        else:
            nei2.add_vect((1, 1)+(0,)*(self.dim-2), permall=True)
        nei2.add_hop(ent, ent, hop=hop2)  # dist=h2dist, loc=h2loc,
                                          # scale=h2scale

        # Return reference to this newly created lattice object
        return lat

    def add_honey(self, pot=0., hop=-1., shape=None, bcond=None, origin=None,
                  label=None):  # pdist=None, ploc=0., pscale=1., hdist=None,
                                # hloc=-1., hscale=1.
        """Create a 2-dimensional 1-band honeycomb lattice with isotropic
        next-neighbor hopping (hop) only."""
        # 2011-08-28 - 2011-11-09
        # former tb.SuperCell.add_honey
        assert self.dim == 2, 'a honeycomb lattice can only be created in ' + \
                              'a 2-dimensional supercell'

        # create lattice
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           label=label)
        lat.bvects[0] = [3, 0]
        lat.bvects[1] = [0, math.sqrt(3)]

        # add the four sites that exist in the conventional unitcell of a
        # honeycomb lattice
        siteA = lat.unitcell.add_site(coord=(0.,    0.), label='Site A')
        siteB = lat.unitcell.add_site(coord=(1./3., 0.), label='Site B')
        siteC = lat.unitcell.add_site(coord=(.5,    .5), label='Site C')
        siteD = lat.unitcell.add_site(coord=(5./6., .5), label='Site D')

        # add a potential to each of the sites
        A = siteA.add_ent(pot=pot, label='A')  # dist=pdist, loc=ploc,
                                               # scale=pscale
        B = siteB.add_ent(pot=pot, label='B')
        C = siteC.add_ent(pot=pot, label='C')
        D = siteD.add_ent(pot=pot, label='D')

        # define intra-unitcell hoppings
        lat.unitcell.add_hop(A, B, hop=hop, iso=True)  # dist=hdist, loc=hloc,
                                                       # scale=hscale
        lat.unitcell.add_hop(B, C, hop=hop, iso=True)
        lat.unitcell.add_hop(C, D, hop=hop, iso=True)

        # define neighbor interactions
        n10 = lat.add_neigh(label='N10')
        n01 = lat.add_neigh(label='N01')
        n11 = lat.add_neigh(label='N11')

        # add the vectors to the neighbor objects
        n10.add_vect(vect=(1, 0), iso=True)
        n01.add_vect(vect=(0, 1), iso=True)
        n11.add_vect(vect=(1, 1), iso=True)

        # define hoppings to neighboring unitcells
        n10.add_hop(D, A, hop=hop)  # dist=hdist, loc=hloc, scale=hscale
        n01.add_hop(C, B, hop=hop)
        n11.add_hop(D, A, hop=hop)

        # return reference to this newly created lattice object
        return lat

    def add_diam(self, pot=0., hop=-1., shape=None, bcond=None, origin=None,
                 bvects=None, label=None):  # pdist=None, ploc=0., pscale=1.,
                                            # hdist=None, hloc=-1., hscale=1.
        """Create a 3-dimensional 1-band diamond lattice with isotropic
        next-neighbor hopping only (hop)."""
        # 2011-08-28 - 2011-11-09
        # former tb.SuperCell.add_diam
        assert self.dim == 3, 'a diamond lattice can only be created in a ' + \
                              '3-dimensional supercell'

        # create lattice
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)

        # add the eight sites that exist in the conventional unitcell of a
        # diamond lattice
        siteA1 = lat.unitcell.add_site(coord=(0., 0., 0.), label='Site A1')
        siteA2 = lat.unitcell.add_site(coord=(0., .5, .5), label='Site A2')
        siteA3 = lat.unitcell.add_site(coord=(.5, 0., .5), label='Site A3')
        siteA4 = lat.unitcell.add_site(coord=(.5, .5, 0.), label='Site A4')
        siteB1 = lat.unitcell.add_site(coord=(.25, .25, .25), label='Site B1')
        siteB2 = lat.unitcell.add_site(coord=(.25, .75, .75), label='Site B2')
        siteB3 = lat.unitcell.add_site(coord=(.75, .25, .75), label='Site B3')
        siteB4 = lat.unitcell.add_site(coord=(.75, .75, .25), label='Site B4')

        # add a potential to each of the sites
        A1 = siteA1.add_ent(pot=pot, label='A1')  # dist=pdist, loc=ploc,
                                                  # scale=pscale
        A2 = siteA2.add_ent(pot=pot, label='A2')
        A3 = siteA3.add_ent(pot=pot, label='A3')
        A4 = siteA4.add_ent(pot=pot, label='A4')
        B1 = siteB1.add_ent(pot=pot, label='B1')
        B2 = siteB2.add_ent(pot=pot, label='B2')
        B3 = siteB3.add_ent(pot=pot, label='B3')
        B4 = siteB4.add_ent(pot=pot, label='B4')

        # define neighbor interaction objects
        n100 = lat.add_neigh(label='N100')
        n010 = lat.add_neigh(label='N010')
        n001 = lat.add_neigh(label='N001')
        n110 = lat.add_neigh(label='N110')
        n101 = lat.add_neigh(label='N101')
        n011 = lat.add_neigh(label='N011')

        # define vectors that lead to the respective neighboring unitcells
        n100.add_vect((1, 0, 0), iso=True)
        n010.add_vect((0, 1, 0), iso=True)
        n001.add_vect((0, 0, 1), iso=True)
        n110.add_vect((1, 1, 0), iso=True)
        n101.add_vect((1, 0, 1), iso=True)
        n011.add_vect((0, 1, 1), iso=True)

        # define hoppings from and to A1
        lat.unitcell.add_hop(A1, B1, hop=hop, iso=True)  # dist=hdist,
                                                         # loc=hloc,
                                                         # scale=hscale
        n110.add_hop(B4, A1, hop=hop)
        n101.add_hop(B3, A1, hop=hop)
        n011.add_hop(B2, A1, hop=hop)

        # define hoppings from and to A2
        lat.unitcell.add_hop(A2, B2, hop=hop, iso=True)  # dist=hdist,
                                                         # loc=hloc,
                                                         # scale=hscale
        lat.unitcell.add_hop(A2, B1, hop=hop, iso=True)
        n100.add_hop(B4, A2, hop=hop)
        n100.add_hop(B3, A2, hop=hop)

        # define hoppings from and to A3
        lat.unitcell.add_hop(A3, B3, hop=hop, iso=True)  # dist=hdist,
                                                         # loc=hloc,
                                                         # ascale=hscale
        lat.unitcell.add_hop(A3, B1, hop=hop, iso=True)
        n010.add_hop(B4, A3, hop=hop)
        n010.add_hop(B2, A3, hop=hop)

        # define hoppings from and to A4
        lat.unitcell.add_hop(A4, B4, hop=hop, iso=True)  # dist=hdist,
                                                         # loc=hloc,
                                                         # scale=hscale
        lat.unitcell.add_hop(A4, B1, hop=hop, iso=True)
        n001.add_hop(B3, A4, hop=hop)
        n001.add_hop(B2, A4, hop=hop)

        # return reference to this newly created lattice object
        return lat

    def add_heis(self, mix=0., mom=1., range=1., coup=1., spin=1., shell=1,
                 shape=None, bcond=None, origin=None, bvects=None, label=None):
        """Create a dilute (homogeneous) Heisenberg system. Do not allow long
        hoppings by default (longhops=False), but make sure that hoppings cause
        a contribution to the diagonal matrix elements (diaghops=True)."""
        # 2012-03-14 - 2012-08-01

        # create lattice
        lat = self.add_sparselat(positions=pos.hom(mix=mix, shape=shape),
                                 shape=shape, longhops=False, diaghops=True)
        site = lat.unitcell.add_site()
        ent = site.add_ent()

        # calculate characteristic vectors of all shells
        cvects = shells.cvects(order=shell, dim=self.dim)

        # define couplings
        def couplings(vect, coup=1., range=1.):
            distance = scipy.sqrt(scipy.sum(scipy.array(vect)**2))
            return coup*scipy.exp(-distance/range)

        # add potentials and hoppings, cycle shells
        # create a Neighbor object for each shell
        neighs = []
        for cvect in cvects[1:]:
            neigh = lat.add_neigh()
            neigh.add_vect(cvect, permall=True)
            neigh.add_hop(ent, ent,
                          hop=-spin*couplings(cvect, coup=coup, range=range))
            neighs.append(neigh)

        # return reference to this newly created sparse lattice object
        return lat

    def add_spheres(self, mom=1., range=1., coup=1., spin=1., shell=1,
                    sconc=None, iconc=None, iconcin=None, iconcout=None,
                    rad=1., space=0., timeout='30s',
                    shape=None, bcond=None, origin=None, bvects=None,
                    label=None):
        """Create a dilute Heisenberg system with spherical inhomogeneities. Do
        not allow long hoppings by default (longhops=False), but make sure that
        hoppings cause a contribution to the diagonal matrix elements
        (diaghops=True)."""
        # 2012-07-06 - 2012-08-01

        # create lattice
        positions = pos.spheres(shape=shape, sconc=sconc,
                                iconc=iconc, space=space,
                                iconcin=iconcin, rad=rad,
                                iconcout=iconcout, timeout=timeout)
        lat = self.add_sparselat(positions=positions, shape=shape,
                                 longhops=False, diaghops=True)
        site = lat.unitcell.add_site()
        ent = site.add_ent()

        # calculate characteristic vectors of all shells
        cvects = shells.cvects(order=shell, dim=self.dim)

        # define couplings
        def couplings(vect, coup=1., range=1.):
            distance = scipy.sqrt(scipy.sum(scipy.array(vect)**2))
            return coup*scipy.exp(-distance/range)

        # add potentials and hoppings, cycle shells
        # create a Neighbor object for each shell
        neighs = []
        for cvect in cvects[1:]:
            neigh = lat.add_neigh()
            neigh.add_vect(cvect, permall=True)
            neigh.add_hop(ent, ent,
                          hop=-spin*couplings(cvect, coup=coup, range=range))
            neighs.append(neigh)

        # return reference to this newly created sparse lattice object
        return lat

    def add_andisp(self, pot=0., hop=-1., coup=1., mom=1., mix=.1, shape=None,
                   bcond=None, origin=None, bvects=None, label=None):
        """Implement Anderson-Ising model, polarized version (impurity spins
        always point up).

        coup: exchange couplings J between electron and local magnetic moment
        mom:  magnetic moment of the impurity
        mix:  concentration of the impurities"""
        # 2012-06-18 - 2012-06-19
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)
        site = lat.unitcell.add_site()

        # make sure that in case of a probability distribution, each random
        # number is used twice (put the same random values on spin-up and
        # spin-down sector)
        ### only works with box distribution so far

        # evaluate mathematical expressions
        coup = self.evalmath(coup)
        mix = self.evalmath(mix)
        mom = self.evalmath(mom)

        # create binary distribution for the impurities
        binary = dist.binary(mix=mix, a=0, b=coup*mom, copies=2)
        if hasattr(pot, '__call__'):  # if isinstance(pot, dist.Distribution)
            pot.copies(2)
        up = site.add_ent(pot=-binary+pot)  # V_i+J_i*S
        down = site.add_ent(pot=binary+pot)  # V_i-J_i*S

        # define next-neighbor interaction
        nei = lat.add_neigh()
        nei.add_vect(permall=True)
        nei.add_hop(up, up, hop=hop)
        nei.add_hop(down, down, hop=hop)

        # return reference to the lattice object
        return lat

    def add_andis(self, pot=0., hop=-1., coup=1., mom=1., mix=.1, shape=None,
                  bcond=None, origin=None, bvects=None, label=None):
        """Implement Anderson-Ising model, unpolarized version (impurity spins
        can point up or down, isotropic distribution).

        coup: exchange couplings J between conduction electrons and local
              magnetic moments
        mom:  magnetic moment of each impurity
        mix:  impurity concentration"""
        # 2013-06-27 - 2013-06-27
        # based on tb.sc.SuperCell.add_andis (2012-06-18 - 2012-06-19)
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)
        site = lat.unitcell.add_site()

        # make sure that in case of a probability distribution, each random
        # number is used twice (put the same random values on spin-up and
        # spin-down sector)
        ### only works with box distribution so far

        # evaluate mathematical expressions
        coup = self.evalmath(coup)
        mix = self.evalmath(mix)
        mom = self.evalmath(mom)

        # create binary distribution for the impurities
        binary = dist.binary(mix=mix, a=0, b=coup*mom, copies=2)
        costheta = dist.binary(mix=.5, a=-1, b=1, copies=2)
        if hasattr(pot, '__call__'):  # if isinstance(pot, dist.Distribution)
            pot.copies(2)
        up = site.add_ent(pot=-binary*costheta+pot)  # V_i+J_i*S*cos(theta)
        down = site.add_ent(pot=binary*costheta+pot)  # V_i-J_i*S*cos(theta)

        # define next-neighbor interaction
        nei = lat.add_neigh()
        nei.add_vect(permall=True)
        nei.add_hop(up, up, hop=hop)
        nei.add_hop(down, down, hop=hop)

        # return reference to the lattice object
        return lat

    def add_andheisp(self, pot=0., hop=-1., coup=1., mom=1., mix=.1,
                     shape=None, bcond=None, origin=None, bvects=None,
                     label=None):
        """Implement Anderson-Heisenberg model with classical impurity spins.
        This is the anisotropic version, somewhat preferring spins pointing up
        or down (along z-axis) due to a "wrong choice" of probability
        distribution.

        coup: exchange couplings J between conduction electrons and local
              magnetic moments
        mom:  magnetic moment of each impurity
        mix:  impurity concentration"""
        # 2012-08-04 - 2012-08-21

        # create lattice
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)  # diaghops=True
        site = lat.unitcell.add_site()

        # evaluate mathematical expressions
        coup = self.evalmath(coup)
        mix = self.evalmath(mix)
        mom = self.evalmath(mom)

        # create distribution objects
        binary = dist.binary(mix=mix, a=0, b=coup*mom, copies=4)
        theta = dist.box(loc=math.pi/2, scale=math.pi, copies=4)
        phi = dist.box(loc=math.pi, scale=2*math.pi, copies=2)
        if hasattr(pot, '__call__'):  # if isinstance(pot, dist.Distribution)
            pot.copies(2)

        # define potentials (V_i +- J_i*S*cos(theta_i))
        up = site.add_ent(pot=-binary*dist.cos(theta)+pot)
        down = site.add_ent(pot=binary*dist.cos(theta)+pot)

        # define on-site interaction (spin-flip)
        site.add_hop(down, up, hop=binary*dist.sin(theta)*dist.exp(phi*1j))
        site.add_hop(up, down, hop=binary*dist.sin(theta)*dist.exp(-phi*1j))

        # define next-neighbor interaction
        nei = lat.add_neigh()
        nei.add_vect(permall=True)
        nei.add_hop(up, up, hop=hop)
        nei.add_hop(down, down, hop=hop)
        #print 'sizeof(neigh):', asizeof.asizeof(neigh)

        # return reference to the lattice object
        return lat

    def add_andheis(self, pot=0., hop=-1., coup=1., mom=1., mix=.1,
                    shape=None, bcond=None, origin=None, bvects=None,
                    label=None):
        """Implement Anderson-Heisenberg model with classical impurity spins.
        This is the isotropic version, where all spin directions have equal
        probability (SU2-invariant).

        coup: exchange couplings J between conduction electrons and local
              magnetic moments
        mom:  magnetic moment of each impurity
        mix:  impurity concentration"""
        # 2013-06-27 - 2013-06-27
        # based on tb.sc.SuperCell.add_andheis (2012-08-04 - 2012-08-21)

        # create lattice
        lat = self.add_lat(shape=shape, bcond=bcond, origin=origin,
                           bvects=bvects, label=label)  # diaghops=True
        site = lat.unitcell.add_site()

        # evaluate mathematical expressions
        coup = self.evalmath(coup)
        mix = self.evalmath(mix)
        mom = self.evalmath(mom)

        # create distribution objects
        binary = dist.binary(mix=mix, a=0, b=coup*mom, copies=4)
        costheta = dist.box(loc=0, scale=2, copies=4)
        phi = dist.box(loc=math.pi, scale=2*math.pi, copies=2)
        if hasattr(pot, '__call__'):  # if isinstance(pot, dist.Distribution)
            pot.copies(2)

        # define potentials (V_i +- J_i*S*cos(theta_i))
        up = site.add_ent(pot=-binary*costheta+pot)
        down = site.add_ent(pot=binary*costheta+pot)

        # define on-site interaction (spin-flip)
        site.add_hop(down, up,
                     hop=binary*dist.sin(dist.arccos(costheta)) *
                     dist.exp(phi*1j))
        site.add_hop(up, down, hop=binary*dist.sin(dist.arccos(costheta))
                     * dist.exp(-phi*1j))

        # define next-neighbor interaction
        nei = lat.add_neigh()
        nei.add_vect(permall=True)
        nei.add_hop(up, up, hop=hop)
        nei.add_hop(down, down, hop=hop)
        #print 'sizeof(neigh):', asizeof.asizeof(neigh)

        # return reference to the lattice object
        return lat

    def is_complex(self):
        """Check whether this supercell contains any complex potentials or
        hoppings."""
        # 2011-08-22 - 2012-08-01
        for lat in self.lats:
            if lat.is_complex():
                return True
        for site in self.sites:
            if site.is_complex():
                return True
        for hop in self.hops.values():
            if scipy.iscomplexobj(hop):
                return True
        return False

    def size(self):
        """Return number of unitcells of all lattices."""
        # 2011-11-29
        total = 0
        for lat in self.lats:
            total += lat.size()
        return total

    def nsites(self):
        """Return number of sites, including single sites as well as those of
        all the lattices."""
        # 2011-11-29 - 2012-02-24
        total = len(self.sites)
        for lat in self.lats:
            total += lat.nsites()
        return total

    def nents(self):
        """Return number of entities of all lattices and all sites. Equals the
        rank of the tight binding matrix."""
        # 2012-02-24
        total = 0
        for site in self.sites:
            total += len(site.ents)
        for lat in self.lats:
            uctotal = 0
            for site in lat.unitcell.sites:
                uctotal += len(site.ents)
            total += uctotal*scipy.prod(lat.shape)
        return total

    def get_lat(self, label):
        """Get lattice by label."""
        # 2011-12-18
        out = None
        for lat in self.lats:
            if lat.label == label:
                out = lat
                break
        if out is None:
            raise KeyError('lattice with label %s not found' % label)
        return out

    def coords(self):
        """Return the coordinates of all the sites (also from all lattice
        sites)."""
        # 2012-04-05
        for site in self.sites:
            yield site.coord
        for lat in self.lats:
            for coord in lat.coords():
                yield coord


class LatticeObject(SuperCellObject):
    """Define shared methods and attributes of the lattice classes (so far:
    Lattice and SparseLattice."""
    # 2012-05-01
    # copied from tb.sc.Lattice from 2011-08-11 - 2012-04-05
    # former tb.Lattice from 2011-03-05 - 2011-03-31

    def is_complex(self):
        """Check whether this lattice has complex potentials or hoppings."""
        # 2011-08-20 - 2011-08-23
        if self.unitcell.is_complex():
            return True
        for neigh in self.neighs:
            if neigh.is_complex():
                return True
        return False

    def is_symmetric(self):
        """Check if tight binding matrix is symmetric."""
        # 2011-08-25 - 2012-03-02
        # Convert the matrix to the dok-format before comparison
        mat = self.tbmat()  # cache=True
        return mat.transpose().todok() == mat.todok()

    def is_hermitian(self):
        """Check if tight binding matrix is hermitian."""
        # 2011-08-25 - 2012-03-02
        # Convert the matrix to the dok-format before comparison
        mat = self.tbmat()  # cache=True
        return mat.transpose().conjugate().todok() == mat.todok()

    def nsites(self):
        """Return number of sites."""
        # 2011-11-29
        return self.size()*self.unitcell.nsites()

    def get_neigh(self, label):
        """Get neighbor by label."""
        # 2011-12-18
        out = None
        for neigh in self.neighs:
            if neigh.label == label:
                out = neigh
                break
        if out is None:
            raise KeyError('neighbor with label %s not found' % label)
        return out

    def add_neigh(self, *neighs, **kwargs):
        """Add a neighbor interaction object to the lattice that may connect
        several unitcells defined by relative vectors. For a complete list of
        possible arguments, look at the definition of Neighbor.__init__.

        Also, predefined neighbor objects may be given as positional arguments.
        Then, no keyword arguments may be given."""
        # 2011-08-20 - 2012-05-01

        kwargs.update(dim=self.dim)
        if len(neighs) == 0:
            kwargs.update(unitcell=self.unitcell)
            neigh = Neighbor(**kwargs)
            self.neighs.append(neigh)
            return neigh
        else:
            if len(kwargs) > 0:
                raise KeyError('no keyword arguments allowed if positional ' +
                               'arguments are given')
            for neigh in neighs:
                if neigh not in self.neighs:
                    neigh.unitcell = self.unitcell
                    self.neighs.append(neigh)


class Lattice(LatticeObject):
    """Define a regular lattice, using a unitcell that may contain multiple
    sites and is repeated periodically in space a given number of times in each
    dimension."""
    # 2011-08-11 - 2013-07-07
    # former tb.Lattice from 2011-03-05 - 2011-03-31

    # initialize instance counter
    count = 0

    def __init__(self, dim=1, shape=None, bcond=None, origin=None, bvects=None,
                 label=None, neighs=None, diaghops=False):
        """Initialize lattice object."""
        # 2011-08-11 - 2013-07-07

        # number of dimensions
        self.dim = self.evalmath(dim, dtype=int)
        assert self.dim > 0, 'number of dimensions must be positive integer'

        # shape
        if shape is None:
            self.shape = (1,)*self.dim
        elif isinstance(shape, basestring):
            try:
                self.shape = eval(shape)
            except:
                self.shape = tuple(shape.split(','))
        else:
            self.shape = tuple(shape)
            for val in self.shape:
                assert val > 0, 'shape must consist of positive integers'

        # boundary conditions
        if bcond is None:
            self.bcond = 'p'*self.dim
        else:
            self.bcond = str(bcond)
            assert bcond != '', 'no boundary conditions given'
            while len(self.bcond) < self.dim:
                self.bcond += self.bcond[-1]

        # origin
        if origin is None:
            self.origin = (0.,)*self.dim
        else:
            self.origin = tuple(origin)

        # basis vectors that span the lattice within the supercell
        if bvects is None:
            self.bvects = scipy.eye(self.dim)
        else:
            self.bvects = scipy.array(bvects)

        # list of inter-cell hoppings
        self.neighs = [] if neighs is None else neighs

        # create unitcell
        self.unitcell = UnitCell(dim=self.dim)
        self.uc = self.unitcell  # shortcut

        # set label
        self.__class__.count += 1
        if label is None:
            label = '%s%i' % (self._ucletters(self.__class__.__name__),
                              self.__class__.count)
        self.label = label

        # initialize switches
        #self.longhops = longhops  # allow long hoppings circling the lattice
        self.diaghops = diaghops  # put real part of negative hoppings on
                                  # diagonal

    def size(self):
        """Return number of unitcells."""
        # 2011-08-21 - 2011-10-12
        return scipy.prod(self.shape)

    def ndindex2index(self, ndindex, shape=None):
        """Return index of the given n-dimensional index that it would have in
        the n-dimensional index list provided by scipy.ndindex. If ndindex is a
        2D array, each row is treated as an n-dimensional index, and the result
        will be a 1D array of indices.  Can also be given a list of nd-indices
        (2d-array)."""
        # 2011-08-23 - 2012-09-03
        if shape is None:
            shape = self.shape
        return scipy.dot(ndindex, scipy.cumprod((1,)+shape[::-1])[-2::-1])
        ### The look-up table could better be calculated once, and not be
        ### re-calculated for every vector, because it is universal!

    def indmat(self, vects):
        """Return index matrix. Format will be coordinate sparse format (coo).
        It will contain ones for those blocks that have to be set to enable
        hopping in the direction of the given vector (vect), and zeros
        elsewhere."""
        # 2011-08-25

        # get all unitcell coordinates (integer)
        coords = scipy.array(tuple(scipy.ndindex(self.shape)))

        # initialize lists that hold the indices of the blocks where the
        # submatrix of this neighbor interaction will be put
        rowind = []
        colind = []
        signchanges = []

        # determine the indices of the blocks where to put the hopping matrix
        # of the neighbor interaction object
        for vect in vects:
            # Get coordinates shifted by this vector
            shifted = coords+vect

            # determine number of sign changes (if antiperiodic boundary
            # conditions are used)
            cdiv = scipy.absolute(scipy.divide(shifted, self.shape))

            # wrap shifted coordinates around so that they do not lie outside
            # the lattice
            shifted = scipy.mod(shifted, self.shape)

            # calculate which indices the shifted coordinates would have in the
            # normal coordinates list
            newrowinds = scipy.arange(self.size(), dtype=int)
            newcolinds = self.ndindex2index(shifted)

            # Debugging checks
            assert len(newrowinds) == self.size()
            assert len(newcolinds) == self.size()

            # respect boundary conditions
            donotuse = scipy.zeros(self.size(), dtype=bool)
            nosc = scipy.zeros(self.size(), dtype=int)  # number of sign
                                                        # changes
            for dind, bc in enumerate(self.bcond):
                if bc == 's':
                    # if a dimension has a static boundary condition and cdiv
                    # is not zero, do not set those blocks
                    donotuse += cdiv[:, dind] != 0
                elif bc == 'a':
                    # For every dimension with antiperiodic boundary condition,
                    # change the sign of the blocks a certain number of times
                    # according to cdiv
                    nosc += cdiv[:, dind]

            # append the indices and sign changes to the big index lists
            rowind += list(newrowinds[donotuse == False])  # "is False" does
            colind += list(newcolinds[donotuse == False])  # not work
            signchanges += list(nosc[donotuse == False])

        # debugging checks
        assert len(rowind) == len(colind)
        assert len(rowind) == len(signchanges)

        # construct index matrix. It defines where and with which sign the
        # hopping submatrix of this neighbor interaction will go.
        return scipy.sparse.coo_matrix(((-1)**scipy.array(signchanges),
                                        (rowind, colind)),
                                       shape=(self.size(),)*2)

    def tbmat(self, format=None):
        """Return tight binding matrix of this lattice, convert to the
        specified format (one of dense, csr, csc, dok, coo, dia, bsr, or
        lil)."""
        # 2011-08-20 - 2012-08-01

        # get index of each entity
        inds = {}
        for ind, ent in enumerate(self.unitcell.ents()):
            inds[ent] = ind

        # create matrix and set blocks on the main diagonal
        #print time.ctime(), \
                #'create matrix of lattice and set diagonal blocks...'
        mat = scipy.sparse.kron(scipy.sparse.eye(*((self.size(),)*2)),
                                self.unitcell.tbmat()).astype(self.infertype())
        #print time.ctime(), 'matrix of lattice created, diagonal blocks set'

        # set random potentials and hoppings within the blocks on the main
        # diagonal
        #print time.ctime(), \
                #'set diagonal random matrix elements of lattice...'
        mat = mat.todok()
        for (ent1, ent2), hop in self.unitcell.random.iteritems():  # new=False
            # define some shortcuts
            ne = len(list(self.unitcell.ents()))  # number of entities
            size = self.size()  # number of cells

            # generate random variates
            variates = hop(size=size)

            # add these random hopping parameters to the big matrix
            helper = scipy.sparse.coo_matrix(([1], ([inds[ent1]],
                                              [inds[ent2]])),
                                             shape=(ne, ne), dtype=int)
            diag = scipy.sparse.spdiags([variates], [0], size, size)
            mat.update(scipy.sparse.kron(diag, helper).todok())

            ### old method. Problem: "+=" only works with lil
            #self._tbmat[inds[ent1]::ne, inds[ent2]::ne] += spdiags([variates],
                                                                    #[0], size,
                                                                    #size)
        #print time.ctime(), 'diagonal random matrix elements of lattice set'

        # set off-diagonal blocks with submatrix
        #print time.ctime(), 'set off-diagonal blocks of lattice...'
        for neigh in self.neighs:
            # get index matrix, indicating the blocks that have to be set to
            # enable hopping for this neighbor interaction
            indmat = self.indmat(neigh.vects)
            isoindmat = self.indmat(neigh.isovects)

            # set certain blocks of the big matrix with the hopping matrix of
            # this neighbor interaction object
            submat = neigh.tbmat()
            helper3 = scipy.sparse.kron(indmat, submat).todok()
            mat.update(helper3)

            # set certain blocks of the big matrix with the adjoint of this
            # hopping matrix
            adjoint = submat.transpose().conjugate()
            #mat.update(scipy.sparse.kron(isoindmat, adjoint).todok())
            mat = mat + scipy.sparse.kron(isoindmat, adjoint).todok()

            # add the negative real part of each off-diagonal block to the
            # corresponding diagonal block in the same row
            if self.diaghops:
                add2diag = scipy.array(helper3.sum(axis=0)).flatten().real
                add2mat = scipy.sparse.dok_matrix(mat.shape)
                add2mat.setdiag(add2diag)
                mat = mat - add2mat

            # set random matrix elements for this hopping matrix
            for (ent1, ent2), hop in neigh.random.iteritems():
                # generate random variates
                if indmat.nnz > 0:
                    indmat.data = hop(size=indmat.nnz)
                if isoindmat.nnz > 0:
                    isoindmat.data = hop(size=isoindmat.nnz)

                # define shortcuts
                ne = len(list(self.unitcell.ents()))  # number of entities

                # add these random hopping parameters to the big matrix
                helper = scipy.sparse.coo_matrix(([1], ([inds[ent1]],
                                                  [inds[ent2]])),
                                                 shape=(ne, ne), dtype=int)
                helper2 = scipy.sparse.kron(indmat, helper).todok()
                #mat.update(helper2)
                #mat.update(scipy.sparse.kron(isoindmat, helper).todok())
                mat = mat + helper2
                mat = mat + scipy.sparse.kron(isoindmat, helper).todok()

                # add the negative real part of each off-diagonal block to the
                # corresponding diagonal block in the same row
                if self.diaghops:
                    add2diag = \
                        scipy.array(helper2.sum(axis=0)).flatten().real/2
                    add2mat = scipy.sparse.dok_matrix(mat.shape)
                    add2mat.setdiag(add2diag)
                    mat = mat-add2mat
        #print time.ctime(), 'off-diagonal blocks of lattice set'

        # convert to CSR sparse format by default
        if format is None:
            format = 'csr'

        # return matrix
        return mat.asformat(format)

    def coords(self):
        """Return the coordinates of all the sites."""
        # 2012-04-05 - 2012-05-01
        for vect in scipy.ndindex(self.shape):
            for site in self.unitcell.sites():
                raise NotImplementedError
                #yield tuple(self.origin+scipy.dot(self.bvects, vect))


class UnitCell(SuperCellObject):
    """Define the unitcell of a lattice. May contain several sites."""
    # 2011-08-21 - 2012-03-02

    # initialize instance counter
    count = 0

    def __init__(self, dim=1, label=None, sites=None, hops=None):
        """Initialize unitcell."""
        # 2011-08-21 - 2012-03-02

        # number of dimensions
        self.dim = self.evalmath(dim, dtype=int)
        assert self.dim > 0, 'number of dimensions must be positive integer'

        # data structures
        self.sites = [] if sites is None else sites
        self.hops = {} if hops is None else hops
        self.random = None

        # set label
        self.__class__.count += 1
        if label is None:
            label = '%s%i' % (self._ucletters(self.__class__.__name__),
                              self.__class__.count)
        self.label = label

    def ents(self, sites=False):
        """Return all entities of all sites that exist in this unitcell. If
        sites is True, also return the site where the entity is located."""
        # 2011-08-22 - 2011-09-02
        for site in self.sites:
            for ent in site.ents:
                if sites:
                    yield ent, site
                else:
                    yield ent

    def add_site(self, *sites, **kwargs):
        """Add a site to the unitcell at the given coordinates (coord). The
        coordinates should be given as relative values from the range [0, 1].
        For a complete list of possible arguments, look at the definition of
        Site.__init__.

        Also, predefined site objects may be given as positional arguments.
        Then, no keyword arguments may be given."""
        # 2011-08-20 - 2012-05-01

        kwargs.update(dim=self.dim)
        if len(sites) == 0:
            site = Site(**kwargs)
            self.sites.append(site)
            return site
        else:
            if len(kwargs) > 0:
                raise KeyError('no keyword arguments allowed if positional ' +
                               'arguments are given')
            for site in sites:
                if site not in self.sites:
                    self.sites.append(site)

    def add_hop(self, *ents, **kwargs):
        """Add a hopping between the given entities. All entities must already
        exist in the list of entities of this unitcell. The hopping parameter
        can either be a constant or drawn from a probability distribution. The
        interaction may be defined as isotropic (iso), meaning that also
        hopping in the opposite direction will be allowed.

        Notes:
        --> Should check if both entities are of the same site. If so, hand
            task off to Site.add_hop."""
        # 2011-09-06 - 2011-11-09

        # make sure all entities exist at this site
        for ent in ents:
            assert ent in self.ents(), \
                'Entity %s does not exist at this site' % repr(ent)

        # handle multiple entities
        assert len(ents) > 1, 'at least two entities have to be given'
        if len(ents) > 2:
            for ent1, ent2 in itertools.combinations(ents, 2):
                self.add_hop(ent1, ent2, **kwargs)
        else:
            ent1, ent2 = ents

            # get keyword arguments
            hop = kwargs.get('hop', -1.)
            #dist = kwargs.get('dist', None)
            #loc = kwargs.get('loc', 0.)
            #scale = kwargs.get('scale', 1.)
            iso = kwargs.get('iso', False)

            # check if the two entities are identical
            if ent1 == ent2:
                # then just redefine the potential of that entity
                ent1.pot = self.evalmath(hop)
                return

            # just delete if zero is given
            if hop == 0. and (ent1, ent2) in self.hops:
                del(self.hops[ent1, ent2])
            else:
                self.hops[ent1, ent2] = self.evalmath(hop)
                # dict(dist=dist, loc=self.evalmath(loc),
                # scale=self.evalmath(scale))

            # isotropic hopping?
            if iso:
                self.add_hop(ent2, ent1, hop=hop.conjugate(), iso=False)
                # dist=dist, loc=loc, scale=scale
                ### What happens if iso is True and hop is random?

    def tbmat(self, format=None):  # new=False, cache=False
        """Return submatrix of the unitcell, convert to the specified format
        (one of dense, csr, csc, dok, coo, dia, bsr, or lil).  If new is True,
        recalculate the whole matrix, including the random hopping dictionary.
        If cache is True, always return the cached version of the matrix (if
        there is no cached version, raise an exception). If new and cache are
        False, calculate the submatrix only if there is no cached version,
        otherwise just return the cache."""
        # 2011-08-22 - 2012-01-24

        # initialize data structures
        self.random = {}

        # get index of each entity
        inds = {}
        for ind, ent in enumerate(self.ents()):
            inds[ent] = ind

        # first, fill all diagonal elements (potentials)
        blocks = scipy.array([None]*len(self.sites)**2) \
            .reshape(len(self.sites), -1).tolist()
        for ind, site in enumerate(self.sites):
            blocks[ind][ind] = site.tbmat()  # new=new
            self.random.update(site.random)
        mat = scipy.sparse.bmat(blocks, format='lil').astype(self.infertype())

        # then, fill all off-diagonal elements, containing hopping between the
        # sites of the unitcell (intra-unitcell hopping)
        for (ent1, ent2), hop in self.hops.iteritems():
            if hasattr(hop, '__call__'):
                self.random[ent1, ent2] = hop
            else:
                mat[inds[ent1], inds[ent2]] = hop

        # return the matrix
        return mat.asformat(format)

    def is_complex(self):
        """Check whether this unitcell object contains complex potentials or
        hoppings."""
        # 2011-08-22 - 2012-08-01
        for site in self.sites:
            if site.is_complex():
                return True
        for hop in self.hops.values():
            if hasattr(hop, '__call__') and hop.is_complex():
                return True
            elif scipy.iscomplexobj(hop):
                return True
        return False

    def nsites(self):
        """Return number of sites."""
        # 2011-11-29
        return len(self.sites)

    def get_site(self, label):
        """Get site by label."""
        # 2011-12-18
        out = None
        for site in self.sites:
            if site.label == label:
                out = site
                break
        if out is None:
            raise KeyError('site with label %s not found' % label)
        return out


class Neighbor(SuperCellObject):
    """Define interaction between different unitcells of a lattice."""
    # 2011-08-20 - 2012-05-01
    # former tb.Neighbor from 2011-03-06 - 2011-03-31
    # former tb.Neighbor from 2011-03-06 - 2011-03-31

    # initialize instance counter
    count = 0

    def __init__(self, dim=1, unitcell=None, label=None, hops=None, vects=None,
                 isovects=None):
        """Initialize neighbor interaction."""
        # 2011-08-20 - 2012-03-02

        # number of dimensions
        self.dim = self.evalmath(dim, dtype=int)
        assert self.dim > 0, 'number of dimensions must be positive integer'

        # data structures
        self.unitcell = unitcell
        self.hops = {} if hops is None else hops
        self.vects = [] if vects is None else vects  # list of integer vectors,
                                                     # leading to those
                                                     # neighbor unitcells that
                                                     # should be set with the
                                                     # hopping matrix
        self.isovects = [] if isovects is None else isovects  # list of integer
                                                              # vectors,
                                                              # leading to
                                                              # those neighbor
                                                              # unitcells that
                                                              # should be set
                                                              # with the
                                                              # adjoint hopping
                                                              # matrix
        self.random = None

        # set label
        self.__class__.count += 1
        if label is None:
            label = '%s%i' % (self._ucletters(self.__class__.__name__),
                              self.__class__.count)
        self.label = label

    def ents(self):
        """Return the entities of all the sites of the unitcell."""
        # 2011-08-20 - 2011-08-22
        if self.unitcell is None:
            raise ValueError('no unitcell given')
        for site in self.unitcell.sites:
            for ent in site.ents:
                yield ent

    def add_hop(self, *ents, **kwargs):
        """Add a hopping from entity 1 of the unitcell to entity 2 of each of
        the specified neighbor cells (which are specified by the given relative
        vectors). All entities must be defined within the unitcell of the
        lattice.  The hopping parameter can either be a constant or drawn from
        a probability distribution. The interaction may be defined as isotropic
        (iso), also allowing hopping from entity 2 to entity 1 (but still in
        the same direction, from this unitcell to the neighbor cell).

        If more than two entities are given, all combinations are added. If iso
        is True, even all permutations are added. Example with three entities
        e1, e2, e3: The possible combinations are (e1, e2), (e1, e3), (e2, e3).
        If iso is True, there will also be the hoppings (e3, e2), (e3, e1),
        (e2, e1)."""
        # 2011-08-20 - 2012-05-01

        # check if unitcell object can be accessed
        if self.unitcell is None:
            raise ValueError('no unitcell given')

        # make sure all given entities exist within the unitcell
        for ent in ents:
            assert ent in self.ents(), \
                'entity %s does not exist in the unitcell' % repr(ent)

        # handle multiple entities
        assert len(ents) > 1, 'at least two entities have to be given'
        if len(ents) > 2:
            for ent1, ent2 in itertools.combinations(ents, 2):
                self.add_hop(ent1, ent2, **kwargs)
        else:
            ent1, ent2 = ents

            # get keyword arguments
            hop = kwargs.pop('hop', -1.)
            iso = kwargs.pop('iso', False)
            assert len(kwargs) == 0, 'there are unknown keyword arguments'

            # just delete if zero is given
            if hop == 0. and (ent1, ent2) in self.hops:
                del(self.hops[ent1, ent2])
            else:
                self.hops[ent1, ent2] = self.evalmath(hop)

            # isotropic hopping?
            if iso:
                self.add_hop(ent2, ent1, hop=hop, iso=False)

    def add_vect(self, vect=None, iso=False, perm=False, permall=False):
        """Add a vector to this neighbor interaction object, leading to a
        specific neighbor cell.

        If iso is set to True, also the neighbor of the negative vector is set,
        with the adjoint (complex conjugated and transposed) of the hopping
        matrix of this neighbor.

        If perm is set to True, the neighbors of all permutations of the given
        vector are set with the same neighbor hopping matrix. Of course, if
        both iso and perm are set to True, the neighbors of the negative
        vectors of all permutations are set with the adjoint of the matrix.
        This case could be useful for bcc-like structures.

        If permall is set to True, the blocks of all permutations of the given
        vector are set with the neighbor's hopping matrix, including all
        negative values, i.e. covering all possible directions. Not the
        adjoint, but the matrix itself is always used. USE THIS OPTION WITH
        CAUTION! It should only be used in systems with only one site per
        unitcell. Otherwise, unwanted hoppings will probably occur. This option
        overrides the options iso and perm."""
        # 2011-08-20
        # former tb.Lattice.add_vect

        # vector
        if vect is None:
            vect = (1,)+(0,)*(self.dim-1)

        # add this vector to the list of vectors of this neighbor interaction
        # Respect automatic symmetry features (permutations and isotropy)
        if permall:
            # add all permutations of the given vector (including all possible
            # sign changes of the elements)
            self.vects += shells.signperms(vect)
        else:
            if perm:
                # add all permutations of the given vector (without sign
                # changes)
                self.vects += list(set(itertools.permutations(vect)))
                if iso:
                    # also add all permutations of the negative vector
                    self.isovects += \
                        list(set(itertools.permutations(tuple(
                            -scipy.array(vect)))))
            else:
                # just add this vector
                self.vects.append(vect)
                if iso:
                    # also add the negative vector
                    self.isovects.append(tuple(-scipy.array(vect)))

    def tbmat(self, format=None):
        """Return submatrix of this neighbor interaction object, containing all
        constant hopping parameters. Convert to the specified format (one of
        dense, csr, csc, dok, coo, dia, bsr, or lil)."""
        # 2012-05-01
        # based on tb.sc.Neighbor.tbmat from 2011-08-20 - 2012-03-02

        # assure that a reference to the unitcell object and the position list
        # are available
        if self.unitcell is None:
            raise AttributeError('no unitcell given')

        # initialize data structures
        mat = scipy.sparse.lil_matrix((len(list(self.ents())),)*2,
                                      dtype=(complex if self.is_complex()
                                             else float))
        self.random = {}

        # get index of each entity
        inds = {}
        for ind, ent in enumerate(self.ents()):
            inds[ent] = ind

        # fill all matrix elements
        for (ent1, ent2), hop in self.hops.iteritems():
            if hasattr(hop, '__call__'):
                self.random[ent1, ent2] = hop
            else:
                mat[inds[ent1], inds[ent2]] = hop

        # return the matrix
        return mat.asformat(format)

    def is_complex(self):
        """Check whether this neighbor object contains complex potentials or
        hoppings."""
        # 2011-08-20 - 2012-08-01
        for ent in self.ents():
            if ent.is_complex():
                return True
        for hop in self.hops.values():
            if hasattr(hop, '__call__') and hop.is_complex():
                return True
            elif scipy.iscomplexobj(hop):
                return True
        return False

    def checkhops(self):
        """Check if all the entities of the defined hoppings exist in the
        unitcell."""
        # 2011-08-22

        # check if a unitcell definition is given
        if self.unitcell is None:
            raise ValueError('no unitcell given')

        # cycle all defined hoppings and perform the check
        for ent1, ent2 in self.hops:
            if ent1 not in self.unitcell.ents():
                return False
            if ent2 not in self.unitcell.ents():
                return False

        # otherwise, return positive check result
        return True


class Site(SuperCellObject):
    """Define a site. May be part of the unitcell of a lattice, or a single
    site within the supercell. May contain several entities (potentials)."""
    # 2011-08-19 - 2012-03-02

    # initialize instance counter
    count = 0

    def __init__(self, dim=1, coord=None, label=None, ents=None, hops=None):
        """Initialize site."""
        # 2011-08-19 - 2012-03-02

        # number of dimensions
        self.dim = self.evalmath(dim, dtype=int)
        assert self.dim > 0, 'number of dimensions must be positive integer'

        # coordinates
        if coord is not None:
            self.coord = tuple(coord)
        else:
            self.coord = (0.,)*self.dim

        # data structures
        self.ents = [] if ents is None else ents
        self.hops = {} if hops is None else hops
        self.random = None

        # set label
        self.__class__.count += 1
        if label is None:
            label = '%s%i' % (self._ucletters(self.__class__.__name__),
                              self.__class__.count)
        self.label = label

    def add_ent(self, *ents, **kwargs):
        """Add an entity to the site. The entity is characterized by a constant
        or random potential (pot).

        Also, predefined site objects may be given as positional arguments.
        Then, no keyword arguments may be given."""
        # 2011-08-19 - 2012-05-01

        if len(ents) == 0:
            ent = Entity(**kwargs)
            self.ents.append(ent)
            return ent
        else:
            if len(kwargs) > 0:
                raise KeyError('no keyword arguments allowed if positional ' +
                               'arguments are given')
            for ent in ents:
                if ent not in self.ents:
                    self.ents.append(ent)

    def add_hop(self, *ents, **kwargs):
        """Add a hopping between the given entities. All entities must already
        exist in the list of entities of this site. The hopping parameter can
        either be constant or drawn from a probability distribution. The
        interaction may be defined as isotropic (iso), meaning that also
        hopping in the opposite direction will be allowed."""
        # 2011-08-19 - 2011-11-09

        # make sure all entities exist at this site
        for ent in ents:
            assert ent in self.ents, \
                'Entity %s does not exist at this site' % repr(ent)

        # handle multiple entities
        assert len(ents) > 1, 'at least two entities have to be given'
        if len(ents) > 2:
            for ent1, ent2 in itertools.combinations(ents, 2):
                self.add_hop(ent1, ent2, **kwargs)
        else:
            ent1, ent2 = ents

            # get keyword arguments
            hop = kwargs.get('hop', -1.)
            #dist = kwargs.get('dist', None)
            #loc = kwargs.get('loc', 0.)
            #scale = kwargs.get('scale', 1.)
            iso = kwargs.get('iso', False)

            # check if the two entities are identical
            if ent1 == ent2:
                # then just redefine the potential of that entity
                ent1.pot = self.evalmath(hop)
                return

            # just delete entry if zero was given
            if hop == 0. and (ent1, ent2) in self.hops:
                del(self.hops[ent1, ent2])
            else:
                self.hops[ent1, ent2] = self.evalmath(hop)

            # isotropic hopping?
            if iso:
                self.add_hop(ent2, ent1, hop=hop.conjugate(), iso=False)
                ### What will happen if hop is random and iso is True?

    def tbmat(self, format=None):
        """Return submatrix of this site, containing all constant hoppings and
        potentials. Convert to the specified format (one of dense, csr, csc,
        dok, coo, dia, bsr, or lil).  If new is True, recalculate the whole
        matrix, including the random hopping dictionary. If cache is True,
        always return the cached version of the matrix (if there is no cached
        version, raise an exception). If new and cache are False, calculate the
        submatrix only if there is no cached version, otherwise just return the
        cache."""
        # 2011-08-19 - 2012-08-04

        # initialize data structures
        mat = scipy.sparse.lil_matrix((len(self.ents),)*2,
                                      dtype=self.infertype())
        self.random = {}

        # get index of each entity
        inds = {}
        for ind, ent in enumerate(self.ents):
            inds[ent] = ind

        # fill the diagonal entries with the potentials of the entities
        pots = []
        for ent in self.ents:
            if hasattr(ent.pot, '__call__'):
                pots.append(0.)
                self.random[ent, ent] = ent.pot
            else:
                pots.append(ent.pot)
        mat.setdiag(pots)

        # fill off-diagonal entries with hoppings
        for (ent1, ent2), hop in self.hops.iteritems():
            if hasattr(hop, '__call__'):
                self.random[ent1, ent2] = hop
            else:
                mat[inds[ent1], inds[ent2]] = hop

        # return the matrix
        return mat.asformat(format)

    def is_complex(self):
        """Check whether this site has complex potentials or hoppings."""
        # 2011-08-19 - 2012-08-01
        for ent in self.ents:
            if ent.is_complex():
                return True
        for hop in self.hops.values():
            if hasattr(hop, '__call__') and hop.is_complex():
                return True
            elif scipy.iscomplexobj(hop):
                return True
        return False

    def checkhops(self, unitcell=None):
        """Check if all the entities of the hoppings exist in the list of
        entities. If a unitcell object is given, also checks if all the hopping
        entities exist in that unitcell."""
        # 2011-08-22

        for ent1, ent2 in self.hops:
            # Always check the entities of this site
            if ent1 not in self.ents:
                return False
            if ent2 not in self.ents:
                return False

            # if a unitcell was given, check all the entities of all the sites
            # of that unitcell, too
            if unitcell is not None:
                if ent1 not in unitcell.ents():
                    return False
                if ent2 not in unitcell.ents():
                    return False

        # otherwise, return positive check result
        return True

    def get_ent(self, label):
        """Get entity by label."""
        # 2011-12-18
        out = None
        for ent in self.ents:
            if ent.label == label:
                out = ent
                break
        if out is None:
            raise KeyError('entity with label %s not found' % label)
        return out


class Entity(SuperCellObject):
    """Define an entity, characterized by a constant or random potential
    (pot)."""
    # 2011-08-19 - 2011-11-09

    # initialize instance counter
    count = 0

    def __init__(self, pot=0., label=None):
        """Initialize entity."""
        # 2011-08-19 - 2011-11-09

        # initialize data structures
        self.pot = self.evalmath(pot)

        # set label
        self.__class__.count += 1
        if label is None:
            label = '%s%i' % (self._ucletters(self.__class__.__name__),
                              self.__class__.count)
        self.label = label

    def is_complex(self):
        """Check whether this entity has a complex potential."""
        # 2011-08-19 - 2012-08-01
        if hasattr(self.pot, '__call__'):
            return self.pot.is_complex()
        else:
            return scipy.iscomplexobj(self.pot)


_dist = dist  # "dist" is already defined inside the function
@Frog(inmap=dict(shape='$0/param', pot='$0/param', hop='$0/param',
                 dist='$0/param', loc='$0/param', scale='$0/param'),
      preproc=dict(shape=lambda p: p.shape, pot=lambda p: p.pot,
                   hop=lambda p: p.hop, dist=lambda p: p.dist.__name__,
                   loc=lambda p: p.loc, scale=lambda p: p.scale),
      outmap={0: None},  # {0: '$0/mat'},
      wrapname='_scnnmatf')
@Frog(preproc=dict(shape=lambda s: [int(i) for i in s.split(',')]))
def scnnmat(shape, pot=0., hop=-1., bcond=None, dist=None, loc=0., scale=1.,
            format=None):
    """Return tight binding matrix of a 1-band simple cubic lattice with
    constant isotropic next-neighbor hopping (hop) and either constant
    potentials (pot) or random potentials, drawn from a certain probability
    distribution (dist) defined by certain parameters (loc, scale). The
    boundary conditions (bcond) can chosen to be static (s), periodic (p) or
    antiperiodic (a) and can be set differently for each dimension. The length
    of the tuple that is specifying the system dimensions (shape) is also
    defining the dimensionality of the system.

    Because of the restriction to this special case, this function reaches an
    optimum of efficiency. The matrix is returned in a sparse matrix format
    (format)."""
    # 2011-09-06 - 2014-01-26
    # based on former tb.scnnmat from 2011-02-28 to 2011-06-20

    try:
        shape = tuple(shape)
    except TypeError:
        shape = (shape,)

    size = scipy.prod(shape)
    dim = len(shape)

    # edit boundary conditions
    if bcond is None:
        bcond = 'p'
    #bcond = bcond[:dim]
    while len(bcond) < dim:
        bcond += bcond[-1]

    # general idea:
    # 1. Build the matrix of dims[:-1] recursevely, set the dims[-1] blocks
    #    with it.
    # 2. Build the matrix of dims[-1], set streched slices dims[:-1] times.
    ### This description sucks!

    # step 1
    subshape = shape[:-1]
    subsize = scipy.prod(subshape, dtype=int)

    # initialize matrix
    if subsize > 1:
        # Get submatrix recursevely and set the diagonal blocks with it
        mat = scipy.sparse.kron(scipy.sparse.eye(*(2*(shape[-1],))),
                                scnnmat(subshape,
                                        hop=hop, bcond=bcond[:-1])).tolil()
    else:
        # Initialize a new empty matrix
        mat = scipy.sparse.lil_matrix(2*(size,))

    # step 2a
    # if (anti)periodic boundary conditions are given, set respective elements
    # as well. Do this first, because they may be overwritten by the direct
    # hoppings in the next step
    if bcond[-1] in 'pa':
        if bcond[-1] == 'p':
            bcondhoparray = hop*scipy.ones((subsize))
        else:
            # sign-flip values in the case of antiperiodic boundary conditions
            bcondhoparray = -hop*scipy.ones((subsize))

        # set complete diagonals at once
        mat.setdiag(bcondhoparray, size-subsize)
        mat.setdiag(bcondhoparray, subsize-size)
    elif bcond[-1] == 's':
        # Do not set anything, but neither show an error message...
        pass
    else:
        raise ValueError('bad boundary condition: %s. ' % bcond[-1] +
                         'Possible values: s (static), p (periodic), a ' +
                         '(antiperiodic)')

    # step 2b
    # set off-diagonal elements with direct hoppings
    hoparray = hop*scipy.ones((size-subsize))
    mat.setdiag(hoparray, subsize)
    mat.setdiag(hoparray, -subsize)

    # step 2c
    # set diagonal elements with potential
    if dist is None:
        if pot != 0.:  # spare this step if potentials are zero
            mat.setdiag(scipy.ones((size))*pot)
    else:
        d = _dist.choose(dist)
        get_random_values = d(loc=loc, scale=scale)
        mat.setdiag(get_random_values(size=size))

    # return matrix
    return mat.asformat(format)


class SparseLattice(LatticeObject):
    """Define a sparse lattice, using a unitcell that may contain multiple
    sites, but in contrast to a regular lattice, is not repeated periodically
    in space, but exists only at given positions within the lattice grid.

    Allow "long" hoppings if "longhops" is set to True. "Long" hoppings are
    those that circle the whole lattice in at least one dimension due to
    periodic boundary conditions (dangerous, because hoppings from a site to
    itself could be possible)."""
    # 2012-04-27 - 2012-09-06
    # based on tb.sc.Lattice from 2011-08-11 - 2012-04-05
    # former tb.Lattice from 2011-03-05 - 2011-03-31

    # initialize instance counter
    count = 0

    def __init__(self, dim=1, shape=None, bcond=None, origin=None, bvects=None,
                 label=None, neighs=None, positions=None, longhops=False,
                 diaghops=False):
        """Initialize sparse lattice object."""
        # 2012-04-27 - 2012-09-06
        # based on tb.sc.Lattice.__init__ from 2011-08-11 - 2012-03-02

        # number of dimensions
        self.dim = self.evalmath(dim, dtype=int)
        assert self.dim > 0, 'number of dimensions must be positive integer'

        # shape
        if shape is None:
            self.shape = (1,)*self.dim
        else:
            self.shape = tuple(shape)
            for val in self.shape:
                assert val > 0, 'shape must consist of positive integers'

        # boundary conditions
        if bcond is None:
            self.bcond = 'p'*self.dim
        else:
            self.bcond = str(bcond)
            assert bcond != '', 'no boundary conditions given'
            while len(self.bcond) < self.dim:
                self.bcond += self.bcond[-1]

        # origin
        if origin is None:
            self.origin = (0.,)*self.dim
        else:
            self.origin = tuple(origin)

        # basis vectors that span the lattice within the real space of the
        # supercell
        if bvects is None:
            self.bvects = scipy.eye(self.dim)
        else:
            self.bvects = scipy.array(bvects)

        # list of inter-cell hoppings
        self.neighs = [] if neighs is None else neighs

        # create unitcell
        self.unitcell = UnitCell(dim=self.dim)
        self.uc = self.unitcell  # shortcut

        # set label
        self.__class__.count += 1
        if label is None:
            label = '%s%i' % (self._ucletters(self.__class__.__name__),
                              self.__class__.count)
        self.label = label

        # initialize list of positions where the unitcell exists
        # can also be set with a PosRule instance, to choose random positions
        # following a certain rule
        if positions is None:
            self.positions = set()
        elif isinstance(positions, pos.PosRule):
            self.positions = positions
        else:
            self.positions = set(positions)

        # initialize switches
        self.longhops = longhops  # allow long hoppings circling the lattice
        self.diaghops = diaghops  # put real part of negative hoppings on
                                  # diagonal

    def size(self):
        """Return number of cells."""
        # 2012-04-27
        # based on tb.sc.Lattice.size from 2011-08-21 - 2011-10-12
        #return scipy.prod(self.shape)
        if hasattr(self.positions, '__call__'):
            return self.positions.num
        else:
            return len(self.positions)

    def indmat(self, vects, positions):
        """Return index matrix. Format will be coordinate sparse format (coo).
        It will contain ones for those blocks that have to be set to enable
        hopping in the direction of the given vector (vect), and zeros
        elsewhere (and minus one where a sign flip should occur due to
        antiperiodic boundary conditions)."""
        # 2012-04-27 - 2012-05-06
        # based on tb.sc.Lattice.indmat from 2011-08-25
        positions = scipy.array(positions)

        # initialize lists that hold the indices of the blocks where the
        # submatrix of this neighbor interaction will be put
        rowind = []
        colind = []
        signchanges = []

        # determine the indices of the blocks where to put the hopping matrix
        # of the neighbor interaction object
        for vect in vects:
            # exclude "long" hopping vectors
            if not self.longhops:
                toolong = False
                for d in xrange(self.dim):
                    # is this vector element greater or equal than the lattice
                    # in this dimension?
                    if abs(vect[d]) >= self.shape[d]:
                        toolong = True
                if toolong:
                    continue

            # get coordinates shifted by the current vector
            shifted = positions+vect

            # determine number of sign changes (if antiperiodic boundary
            # conditions are used)
            cdiv = scipy.absolute(scipy.divide(shifted, self.shape))

            # wrap around the shifted coordinates so that they do not lie
            # outside the lattice
            shifted = scipy.mod(shifted, self.shape)

            # calculate which indices the shifted coordinates would have in the
            # normal coordinates list
            newrowinds = scipy.arange(self.size(), dtype=int)
            newcolinds = core.findinds(positions, shifted)

            # delete those entries whose shifted indices do not appear (where
            # newcolinds has value -1)
            select = newcolinds != -1
            #print 'select', select, len(select) # error for shape=10,10,10???
            #print 'newrowinds', newrowinds, len(newrowinds)
            #print 'newcolinds', newcolinds, len(newcolinds)
            #print 'positions', positions, len(positions)
            #print 'size', self.size()
            newrowinds = newrowinds[select]
            newcolinds = newcolinds[select]

            # respect boundary conditions
            donotuse = scipy.zeros(len(newcolinds), dtype=bool)
            nosc = scipy.zeros(len(newcolinds), dtype=int)  # number of sign
                                                            # changes
            for dind, bc in enumerate(self.bcond):
                if bc == 's':
                    # if a dimension has static boundary conditions and cdiv is
                    # not zero, do not set those blocks
                    donotuse += cdiv[:, dind] != 0
                elif bc == 'a':
                    # for every dimension with antiperiodic boundary
                    # conditions, change the sign of the blocks a certain
                    # number of times according to cdiv
                    nosc += cdiv[:, dind]

            # append the indices and sign changes to the big index lists
            rowind += list(newrowinds[donotuse == False])  # "is False" does
            colind += list(newcolinds[donotuse == False])  # not work
            signchanges += list(nosc[donotuse == False])

        # debugging checks
        assert len(rowind) == len(colind)
        assert len(rowind) == len(signchanges)

        # construct index matrix. It defines where and with which sign the
        # hopping submatrix of this neighbor interaction will go.
        return scipy.sparse.coo_matrix(((-1)**scipy.array(signchanges),
                                        (rowind, colind)),
                                       shape=(self.size(),)*2)

    def coords(self):
        """Return the coordinates of all the sites."""
        # 2012-05-01
        # based on tb.sc.Lattice from 2012-04-05
        for pos in self.positions:
            for site in self.unitcell.sites():
                raise NotImplementedError
                #yield tuple(self.origin+scipy.dot(self.bvects, site.coord))

    def tbmat(self, format=None, distinguish=False):
        """Return tight binding matrix of this sparse lattice, convert to the
        specified format (one of dense, csr, csc, dok, coo, dia, bsr, or lil).

        If distinguish is True, also return a tuple of index lists, specifying
        how the positions can be distinguished."""
        # 2012-05-06 - 2012-09-03
        # based on tb.sc.Lattice.tbmat from 2011-08-20 - 2012-03-02

        # get all cell positions, sort them to have a fixed order
        if hasattr(self.positions, '__call__'):
            # generate new realization of disorder
            if distinguish:
                # also return indices of impurities inside/outside the spheres
                position_classes = self.positions(distinguish=True)
                positions = list(set(itertools.chain(*position_classes)))
                positions.sort()

                # convert positions to indices
                #print position_classes, len(position_classes)
                position_classes_ind = []
                for position_class in position_classes:
                    #position_classes_ind.append(
                        #list(self.ndindex2index(position_class)))
                    indices = []
                    for p in position_class:
                        indices.append(positions.index(p))
                    position_classes_ind.append(indices)
            else:
                positions = self.positions()
        else:
            # use the given fixed list of positions
            if distinguish is True:
                raise ValueError('cannot distinguish different classes of ' +
                                 'positions if they are provided directly')
            positions = list(self.positions)
        positions.sort()
        self.positions_cache = positions  # remember positions for checks

        # get index of each entity
        inds = {}
        for ind, ent in enumerate(self.unitcell.ents()):
            inds[ent] = ind

        # create matrix and set blocks on the main diagonal
        mat = scipy.sparse.kron(scipy.sparse.eye(*((self.size(),)*2)),
                                self.unitcell.tbmat()).astype(self.infertype())

        # set random potentials and hoppings within the blocks on the main
        # diagonal
        for (ent1, ent2), hop in self.unitcell.random.iteritems():  # new=False
            # define some shortcuts
            ne = len(list(self.unitcell.ents()))  # number of entities
            size = self.size()                     # number of cells

            # generate random variates
            variates = hop(size=size)

            # add these random hopping parameters to the big matrix
            helper = scipy.sparse.coo_matrix(([1], ([inds[ent1]],
                                              [inds[ent2]])),
                                             shape=(ne, ne), dtype=int)
            diag = scipy.sparse.spdiags([variates], [0], size, size)
            mat.update(scipy.sparse.kron(diag, helper).todok())
            #mat = mat+scipy.sparse.kron(diag, helper)

            ### old method. Problem: "+=" only works with lil
            #self._tbmat[inds[ent1]::ne, inds[ent2]::ne] += spdiags([variates],
                                                                    #[0], size,
                                                                    #size)

        # set off-diagonal blocks with submatrix
        for neigh in self.neighs:
            # get index matrix, indicating the blocks that have to be set to
            # enable hopping for this neighbor interaction
            indmat = self.indmat(neigh.vects, positions)
            isoindmat = self.indmat(neigh.isovects, positions)

            # set certain blocks of the big matrix with the hopping matrix of
            # this neighbor interaction object
            submat = neigh.tbmat()
            helper3 = scipy.sparse.kron(indmat, submat).todok()  # need it
                                                                 # again later
            #mat.update(helper3)
            mat = mat + helper3

            # set certain blocks of the big matrix with the adjoint of this
            # hopping matrix
            adjoint = submat.transpose().conjugate()
            #mat.update(scipy.sparse.kron(isoindmat, adjoint).todok())
            mat = mat+scipy.sparse.kron(isoindmat, adjoint)

            # add the negative real part of each off-diagonal block to the
            # corresponding diagonal block in the same row
            if self.diaghops:
                add2diag = scipy.array(helper3.sum(axis=0)).flatten().real
                add2mat = scipy.sparse.dok_matrix(mat.shape)
                add2mat.setdiag(add2diag)
                mat = mat-add2mat

            # set random matrix elements for this hopping matrix
            for (ent1, ent2), hop in neigh.random.iteritems():  # new=False
                # generate random variates
                indmat.data = hop(size=indmat.nnz)
                isoindmat.data = hop(size=isoindmat.nnz)

                # define shortcut
                ne = len(list(self.unitcell.ents()))  # number of entities

                # add these random hopping parameters to the big matrix
                helper = scipy.sparse.coo_matrix(([1], ([inds[ent1]],
                                                  [inds[ent2]])),
                                                 shape=(ne, ne), dtype=int)
                helper2 = scipy.sparse.kron(indmat, helper).todok()  # need it
                                                                     # later
                #mat.update(helper2)
                #mat.update(scipy.sparse.kron(isoindmat, helper).todok())
                mat = mat + helper2
                mat = mat + scipy.sparse.kron(isoindmat, helper).todok()

                # add the negative real part of each off-diagonal block to the
                # corresponding diagonal block in the same row
                if self.diaghops:
                    add2diag = \
                        scipy.array(helper2.sum(axis=0)).flatten().real/2
                    add2mat = scipy.sparse.dok_matrix(mat.shape)
                    add2mat.setdiag(add2diag)
                    mat = mat - add2mat

        # convert to CSR sparse format by default
        if format is None:
            format = 'csr'

        # return matrix
        if distinguish:
            return mat.asformat(format), tuple(position_classes_ind)
        else:
            return mat.asformat(format)

    def _ndindex2index_old(self, ndindex):
        """Return index of the given n-dimensional index that it would have in
        the n-dimensional index list provided by scipy.ndindex. If ndindex is a
        2D array, each row is treated as an n-dimensional index, and the result
        will be a 1D array of indices.  Can also be given a list of nd-indices
        (2d-array)."""
        # 2012-09-03
        # copied from tb.sc.Lattice.ndindex2index (developed 2011-08-23)
        return scipy.dot(ndindex, scipy.cumprod((1,) +
                                                self.shape[::-1])[-2::-1])
        ### The look-up table could better be calculated once, and not be
        ### re-calculated for every vector, because it is universal!

    def ndindex2index(self, ndindex, shape=None):
        """Return index of the given n-dimensional index that it would have in
        the n-dimensional index list provided by scipy.ndindex. If ndindex is a
        2D array, each row is treated as an n-dimensional index, and the result
        will be a 1D array of indices."""
        # 2012-09-03
        # copied from tb.sc.Lattice.ndindex2index (developed 2011-08-23
        # - 2012-09-03)
        if shape is None:
            shape = self.shape
        return scipy.dot(ndindex, scipy.cumprod((1,)+shape[::-1])[-2::-1])
        ### The look-up table could better be calculated once, and not be
        ### re-calculated for every vector, because it is universal!


# for compatibility of old HDF5 data files where SuperCell objects were pickled
# can be erased in the future
uniform = dist.uniform
