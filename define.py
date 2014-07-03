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
"""Define tight binding systems using instances of :class:`tb.sc.SuperCell`."""
__created__ = '2013-07-12'
__modified__ = '2014-01-27'
import bundle
import dummy
import misc
import sc

try:
    from frog import Frog, eval_if_str
except ImportError:
    Frog = dummy.Decorator


# common frog configuration for all frogs defined in this module
shortopts = dict(scale='c', shape='s', mom='m', mix='x', iconcin='i',
                 iconcout='o', coup='j', rad='r', range='a', sconc='x',
                 iconc='t', space='d', shell='l')
longopts = dict(sconc='spheres', iconc='total', iconcin='inside',
                iconcout='outside', space='dist', rad='radius')
preproc = dict(shape=eval_if_str)
opttypes = dict(mix=str)
optdoc = dict(scale='scale of the continuous probability ' +
                    'distribution, e.g. the width of a uniform distribution',
              dist='disorder distribution ' +
                   '("uniform", "cauchy", "normal" or "triangular")',
              loc='location of the probability distribution, i.e. the ' +
                  'energy around which the disordered potentials scatter',
              hop='constant next-neighbor hopping parameter',
              bcond='boundary conditions. Must be a string ' +
                    'consisting only of the characters "s" (static), "p" ' +
                    '(periodic) and "a" (anti-periodic). The length of ' +
                    'the string must not be greater than the number of ' +
                    'dimensions (length of shape).',
              shape='shape of the lattice (number of unitcells ' +
                    'in each dimension, separated by commas). At the ' +
                    'same time, it defines the dimensionality of the ' +
                    'system',
              mom='magnetic moment of the impurities',
              mix='concentration of impurities',
              coup='exchange coupling strength between band electrons ' +
                   'and magnetic impurities',
              sconc='either sphere concentration or number of spheres',
              iconc='either total impurity concentration or total ' +
                    'number of impurities',
              iconcin='either impurity concentration inside the spheres ' +
                      'or number of impurities inside the spheres',
              iconcout='either impurity concentration outside the spheres ' +
                       'or number of impurities outside the spheres',
              space='minimum distance between the spheres ' +
                    '(less sphere diameter)',
              rad='radius of the spheres',
              timeout='timeout for finding a suitable ' +
                      'configuration of sphere positions',
              range='interaction range',
              shell='number of shells to consider for the couplings')
outmap = {0: '%0/scell', 1: '%0/param'}


@Frog(outmap=outmap, preproc=preproc, optdoc=optdoc, shortopts=shortopts)
def anderson(hop=-1., dist='uniform', loc=0., scale=1., shape=(10,),
             bcond='p'):
    """Define 1-band simple-cubic tight binding system with site-diagonal
    disorder and constant isotropic next-neighbor hopping, using a continuous
    probability distribution. Return supercell object (instance of
    :class:`tb.sc.SuperCell`) and parameter set (instance of
    :class:`bundle.Bundle`)."""
    # created 2013-07-07, modified 2013-07-12
    # former tb._anderson (developed 2011-09-15 - 2012-11-28)
    # former tb._Anderson from 2011-03-29
    # former anderson.py from 21/01/2010-03/09/2010
    dim = len(shape)
    scell = sc.SuperCell(dim=dim)
    dist = sc.dist.select(dist)
    scell.add_scnn(pot=dist(loc, scale), hop=hop, bcond=bcond,
                   shape=shape)
    pset = bundle.Bundle(dist=dist, dim=dim, loc=loc, scale=scale, shape=shape,
                         hop=hop, pot=loc)
    return scell, pset


@Frog(outmap=outmap, preproc=preproc, optdoc=optdoc, shortopts=shortopts)
def andisp(hop=-1., dist='uniform', loc=0., scale=1., shape=(10,), bcond='p',
           coup=1., mix=.1, mom=1.):
    """Define Anderson-Ising model with polarized impurity spins (spin-up)."""
    # created 2013-07-12, modified 2014-01-27
    # former tb._andis (developed 2012-09-10 - 2012-11-19)
    # former tb._Andis (developed 2012-06-15 - 2012-06-19)
    mix = misc.get_ratio(mix)
    dim = len(shape)
    scell = sc.SuperCell(dim=dim)
    dist = sc.dist.select(dist)
    scell.add_andisp(pot=dist(loc, scale), hop=hop, bcond=bcond,
                     shape=shape, coup=coup, mix=mix, mom=mom)
    pset = bundle.Bundle(dist=dist, dim=dim, loc=loc, scale=scale, shape=shape,
                         mix=mix, coup=coup, mom=mom, hop=hop, pot=loc)
    return scell, pset


@Frog(outmap=outmap, preproc=preproc, optdoc=optdoc, shortopts=shortopts)
def andis(hop=-1., dist='uniform', loc=0., scale=1., shape=(10,), bcond='p',
          coup=1., mix=.1, mom=1.):
    """Define Anderson-Ising model with unpolarized impurity spins (spin-up and
    spin-down)."""
    # created 2013-07-12, modified 2014-01-27
    # former tb._andisi (developed 2013-06-27 - 2013-06-27)
    # based on tb._andis (2012-09-10 - 2012-11-19)
    # former tb._Andis (developed 2012-06-15 - 2012-06-19)
    mix = misc.get_ratio(mix)
    dim = len(shape)
    scell = sc.SuperCell(dim=dim)
    dist = sc.dist.select(dist)
    scell.add_andis(pot=dist(loc, scale), hop=hop, bcond=bcond, shape=shape,
                    coup=coup, mix=mix, mom=mom)
    pset = bundle.Bundle(dist=dist, dim=dim, loc=loc, scale=scale, shape=shape,
                         mix=mix, coup=coup, mom=mom, hop=hop, pot=loc)
    return scell, pset


@Frog(outmap=outmap, preproc=preproc, optdoc=optdoc, shortopts=shortopts)
def andheisp(hop=-1., dist='uniform', loc=0., scale=1., shape=(10,),
             bcond='p', coup=1., mix=.1, mom=1.):
    """Define Anderson-Heisenberg model with random impurity spins (somewhat
    preferring the z-axis)."""
    # created 2013-07-12, modified 2014-01-27
    # former tb._andheis (developed 2012-09-10 - 2012-12-17)
    # based on tb._andis from 2012-09-10
    mix = misc.get_ratio(mix)
    dim = len(shape)
    scell = sc.SuperCell(dim=dim)
    dist = sc.dist.select(dist)
    scell.add_andheisp(pot=dist(loc, scale), hop=hop, bcond=bcond,
                       shape=shape, coup=coup, mix=mix, mom=mom)
    pset = bundle.Bundle(dist=dist, dim=dim, loc=loc, scale=scale, shape=shape,
                         mix=mix, coup=coup, mom=mom, hop=hop, pot=loc)
    return scell, pset


@Frog(outmap=outmap, preproc=preproc, optdoc=optdoc, shortopts=shortopts)
def andheis(hop=-1., dist='uniform', loc=0., scale=1., shape=(10,),
            bcond='p', coup=1., mix=.1, mom=1.):
    """Define Anderson-Heisenberg model with isotropic random impurity spins
    (conforming to the SU(2) group)."""
    # created 2013-07-12, modified 2014-01-27
    # former tb._andheisi (developed 2013-06-27 - 2013-06-27)
    # based on tb._andheis (2012-09-10 - 2012-12-17)
    # based on tb._andis from 2012-09-10
    mix = misc.get_ratio(mix)
    dim = len(shape)
    scell = sc.SuperCell(dim=dim)
    dist = sc.dist.select(dist)
    scell.add_andheis(pot=dist(loc, scale), hop=hop, bcond=bcond,
                      shape=shape, coup=coup, mix=mix, mom=mom)
    pset = bundle.Bundle(dist=dist, dim=dim, loc=loc, scale=scale, shape=shape,
                         mix=mix, coup=coup, mom=mom, hop=hop, pot=loc)
    return scell, pset


@Frog(outmap=outmap, preproc=preproc, optdoc=optdoc, shortopts=shortopts)
def heis(mix=.1, range=1., coup=1., mom=1., shell=1, bcond='p', shape=(10,)):
    """Define homogeneous dilute Heisenberg model."""
    # created 2013-07-12, modified 2013-07-18
    # former tb._Heis (developed 2012-03-14 - 2012-09-06)
    mix = misc.get_ratio(mix)
    dim = len(shape)
    scell = sc.SuperCell(dim=dim)
    scell.add_heis(range=range, coup=coup, mix=mix, mom=mom, bcond=bcond,
                   shape=shape, shell=shell)
    pset = bundle.Bundle(dim=dim, shape=shape, mix=mix, coup=coup, mom=mom,
                         shell=shell, range=range)
    return scell, pset


@Frog(outmap=outmap, preproc=preproc, optdoc=optdoc, shortopts=shortopts,
      longopts=longopts)
def spheres(mix=.1, range=1., coup=1., mom=1., shell=1, bcond='p',
            shape=(10,), sconc=None, iconc=None, iconcin=None, iconcout=None,
            space=.1, rad=1., timeout='30s'):
    """Define dilute Heisenberg model with sphere-like inhomogeneities."""
    # created 2013-07-12, modified 2013-07-18
    # former tb._Spheres (developed 2012-03-14 - 2012-09-06)
    # former tb._Heis (developed 2012-03-14 - 2012-09-06)
    dim = len(shape)
    scell = sc.SuperCell(dim=dim)
    scell.add_spheres(range=range, coup=coup, mom=mom, bcond=bcond,
                      shape=shape, shell=shell, sconc=sconc, iconc=iconc,
                      iconcin=iconcin, space=space, iconcout=iconcout,
                      rad=rad, timeout=timeout)
    pset = bundle.Bundle(dim=dim, range=range, coup=coup, mom=mom, bcond=bcond,
                         shape=shape, shell=shell, sconc=sconc, iconc=iconc,
                         iconcin=iconcin, space=space, iconcout=iconcout,
                         rad=rad, timeout=timeout)
    return scell, pset


##class _VJ(TBModelHDP):
  ##"""Define V-J model. Use 1-band tight binding model, simple cubic, constant
  ##isotropic next-neighbour interaction."""
  ### 2011-12-28
  ##version = '2011-12-28'
  ##usage = '%prog [options] filename'

  ##def __init__(self):
    ##TBModelHDP.__init__(self)

    ### Set options
    ##self.op.add_option('-x', '--mixing-ratio', dest='mix', default='',
                       ##help='set mixing ratio of magnetic impurities.')
    ##self.op.add_option('-V', dest='V', default=1., type=float,
                       ##help='set position of the bound state with respect '+\
                            ##'to the valence band edge')
    # better use normal site-potential option instead of V? Allow first site to
    # be different from zero?
    #self.op.add_option('-J', '--coup', dest='coup', default=1.,
                       ##type=float, help='set coupling strength')
    #self.op.add_option('-S', '--impurity-spin', dest='impspin', default='5/2',
                       ##type=str, help='set impurity spin (e.g. Mn: 5/2)')
    #self.op.add_option('-a', '--lattice-parameter', dest='latpam', type=float,
                      #default=3.55,
                      #help='set lattice parameter (e.g. GaAs: 3.55 Angstrom)')
    ##### GaAs (zinc-blende) has a0 = 5.65 Angstrom, hence in simple cubic it
    ##### becomes a = a0/4**(1/3) = 3.55 Angstrom
    ##### lattice parameter could be shifted to TBModelHDP

  ##def __call__(self, *args, **kwargs):
    ##TBModelHDP.__call__(self, *args, **kwargs)

    ### check options
    ##if len(self.opts.hop) != 1:
      ##self.op.error('exactly one hopping parameter must be given')
    ##if len(self.opts.pot) > 0:
      ##self.op.error('no potential may be directly specified in this model')
    ##if self.opts.mix < 0. or self.opts.mix > 1.:
      ##self.op.error('bad mixing ratio: %f. ' % self.opts.mix +\
                    ##'Must lie between 0.0 and 1.0.')

    ### define supercell
    ##dim = len(self.opts.shape)
    ##scell = sc.SuperCell(dim=dim)

    ### find probability distribution
    ##pot = (0., self.opts.V) ### include JS*s somehow
    ##### assume S = Sz (ferromagnetic groundstate)
    ##mix = (1.-self.opts.mix, self.opts.mix)

    ### define lattice
    ##lat = scell.add_scnn(pot=sc.multi(*zip(pot, mix)), hop=self.opts.hop[0],
                         ##bcond=self.opts.bcond, shape=self.opts.shape,
                         ##bvect=scipy.diag([self.opts.latpam,]*dim))

    ### create output parameter set
    ##self.pout[self.scdset] = scell
    ##self.pout.mix     = mix
    ##self.pout.shape   = self.opts.shape
    ##self.pout.dist    = multi
    ##self.pout.dim     = dim
    ##self.pout.V       = self.opts.V
    ##self.pout.coup    = self.opts.coup
    ##self.pout.impspin = self.opts.impspin

    ### store or return supercell definition
    ##if self.shellmode:
      ###self.save('/%s' % self.pgroup)
      ##self.save('param')

    ##else:
      ##return scell
    ##### create magic dataset name param to store or return the parameter
    ##### set (and to load the parameter sets from files or function arguments,
    ##### of course). OK, pout will be stored anyway, but also return it in
    ##### function mode!
    ##### return self.save('param')
