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
"""Tight binding package. Define tight binding systems (see submodule
:mod:`tb.define`) as instances of the class :class:`tb.sc.SuperCell` and create
the dynamic matrix (tight-binding matrix) using its method
:py:meth:`tb.sc.SuperCell.tbmat`."""
__created__ = '2013-07-07'
__modified__ = '2013-10-31'
# former tb (developed 2011-11-03 - 2013-06-27)

import bundle
import dummy

try:
    from frog import Frog
except ImportError:
    Frog = dummy.Decorator


#==========================#
# Propagate parameter sets #
#==========================#


@Frog(inmap=dict(params='$@/param'), outmap={0: '%0/param'}, overwrite=True,
      usage='%prog [options] [INPUT_FILE1 [INPUT_FILE2 [...]]] OUTPUT_FILE',
      prolog='This frog "propagates" the parameter sets of the input ' +
             'files to the given output file, removing those parameters ' +
             'whose value is not the same throughout the input files.')
def intersect(*params):
    """Return intersection of given parameter sets. Expect list of instances
    of :class:`bundle.Bundle`. Return instance of :class:`bundle.Bundle`."""
    return bundle.intersection(*params)


@Frog(inmap=dict(scell='$0/scell'))
def tbmat(scell, format=None):
    """Calculate the tight binding matrix of the given supercell definition
    *scell*."""
    __created__ = '2014-01-26'
    __modified__ = '2014-01-26'
    return scell.tbmat(format=format)


###========================================================================#
### Hybrid data processing units that display information about data files #
###========================================================================#

##class _Scell(hdp.HDP):
  ##"""Load supercell definition of the given file."""
  ### 2011-10-13 - 2012-01-31
  ### former tb._Supercell from 2011-04-04
  ##version = '2012-01-31'
  ##usage = '%prog [options] filename'
  ##usage = '%prog [options] filenames'
  ##nin    = 0
  ##nout   = 0
  ##ninout = None
  ##sfrom = ffrom = 'file'
  ##sto = 'stdout'
  ##fto = 'return'

  ##def __init__(self):
    ##hdp.HDP.__init__(self)

    ### Set options
    ##self.op.add_option('-l', '--lat', dest='lat', default=None, type=str,
                       ##help='select lattice (index or label)')
    ##self.op.add_option('-s', '--site', dest='site', default=None, type=str,
                       ##help='select site (index or label)')
    ##self.op.add_option('-e', '--ent', dest='ent', default=None, type=str,
                       ##help='select entity (index or label)')
    ##self.op.add_option('-n', '--neigh', dest='neigh', default=None, type=str,
                       ##help='select neighbor (index or label)')
    ##self.op.add_option('-u', '--ucell', dest='ucell', default=False,
                       ##action='store_true',
                       ##help='display unitcell of the selected lattice')
    ##self.op.add_option('-f', '--filter', dest='filter', default=None,
                       ##type=str,
                       ##help='filter supercell attributes')
    ##self.op.add_option('-t', '--table', dest='table', default=False,
                       ##action='store_true', help='display as table')
    ##self.op.add_option('-h', '--head', dest='head', default=False,
                       ##action='store_true', help='show table head line')
    ##self.op.add_option('-w', '--width', dest='width', default=None, type=int,
                       ##help='set with of the terminal window')
    ##self.op.add_option('-S', '--sep', dest='sep', default='  ', type=str,
                       ##help='set column separator')

  ##def __call__(self, *args, **kwargs):
    ##hdp.HDP.__call__(self, *args, **kwargs)

    ### initialize table
    ##if self.opts.table:
      ##self.table = misc.Tab(head=self.opts.head, titles=['filename'],
                            ##width=self.opts.width, sep=self.opts.sep)

    ### load supercells from all files
    ##self.load('__scell__')

    ### cycle files
    ##for (ind, data), filename in itertools.izip(enumerate(self.din),
                                                    #self.fin):
      ##scell = data.__scell__

      ##if self.opts.lat is not None:
        ##lat = self.trylat(scell)

        ##if self.opts.ucell:
          ##if self.opts.site is not None:
            ##site = self.trysite(lat.unitcell)
            ##if self.opts.ent is not None:
              ##ent = self.tryent(site)
              ##self.printobj(ent, filename, ind)
            ##else:
              ##self.printobj(site, filename, ind)
          ##else:
            ##self.printobj(lat.unitcell, filename, ind)
        ##elif self.opts.site is not None:
          ##site = self.trysite(lat.unitcell)

          ##if self.opts.ent is not None:
            ##ent = self.tryent(site)
            ##self.printobj(ent, filename, ind)
          ##else:
            ##self.printobj(site, filename, ind)
        ##elif self.opts.neigh is not None:
          ##neigh = self.tryneigh(lat)
          ##self.printobj(neigh, filename, ind)
        ##else:
          ##self.printobj(lat, filename, ind)
      ##else:
        ##self.printobj(scell, filename, ind)

    ### print table
    ##if self.shellmode:
      ##if self.opts.table:
        ##self.table.display()
      ##else:
        ### the supercell definitions were already printed out within the loop
        ##pass
    ##else:
      ### function mode. Return supercell definitions
      ##return self.save(self.scdset)

  ##def trylat(self, scell):
    ##"""Select lattice by given option string."""
    ### 2011-12-18
    ##try:
      ##lind = int(self.opts.lat)
      ##try:
        ##return scell.lats[lind]
      ##except KeyError:
        ##raise KeyError, 'lattice index out of bounds: %i' % lind
    ##except ValueError:
      ##return scell.get_lat(self.opts.lat)

  ##def trysite(self, uc):
    ##"""Select site by given option string."""
    ### 2011-12-18
    ##try:
      ##sind = int(self.opts.site)
      ##try:
        ##return uc.sites[sind]
      ##except KeyError:
        ##raise KeyError, 'site index out of bounds: %i' % sind
    ##except ValueError:
      ##return uc.get_site(self.opts.site)

  ##def tryneigh(self, lat):
    ##"""Select neighbor by given option string."""
    ### 2011-12-18
    ##try:
      ##nind = int(self.opts.neigh)
      ##try:
        ##return lat.neighs[nind]
      ##except KeyError:
        ##raise KeyError, 'neighbor index out of bounds: %i' % nind
    ##except ValueError:
      ##return lat.get_neigh(self.opts.neigh)

  ##def tryent(self, site):
    ##"""Select entity by given option string."""
    ### 2011-12-18
    ##try:
      ##eind = int(self.opts.ent)
      ##try:
        ##return site.ents[eind]
      ##except KeyError:
        ##raise KeyError, 'entity index out of bounds: %i' % eind
    ##except ValueError:
      ##return site.get_ent(self.opts.ent)

  ##def printobj(self, obj, filename, ind):
    ##"""Print selected object, apply filter, respect table option. If in
    ##function mode, append to self.dout instead."""
    ### 2011-12-18 - 2011-12-20

    ##if self.opts.filter is None:
      ##filtered = obj.__dict__.keys()
    ##else:
      ### filter attributes
      ##filtered = set()
      ##for pattern in self.opts.filter.split(','):
        ##filtered.update(fnmatch.filter(obj.__dict__.keys(), pattern))
      ##filtered = list(filtered)
    ##filtered.sort()
    ##for key in obj.__dict__.keys():
      ##if key not in filtered:
        ##del(obj.__dict__[key])

    ### decide what to do
    ##if self.shellmode:
      ##if self.opts.table:
        ##obj.__dict__.update(filename=filename)
        ##self.table.add(**obj.__dict__)
      ##elif len(obj.__dict__) > 1:
        ### print object
        ##print obj
      ##elif len(obj.__dict__) == 1:
        ### print single value (one for each file)
        ##print obj.__dict__.pop(obj.__dict__.keys()[0])
      ##else:
        ### print nothing
        ##pass
    ##else:
      ##self.dout[ind][self.scdset] = obj


##class _Erange(hdp.HDP):
  ##"""Calculate energy range (spectrum boundaries) of the given tight binding
  ##matrix using the Lanczos algorithm."""
  ### 2012-05-23
  ##version = '2012-05-29'
  ##usage = '%prog [options] filenames'
  ##nin = nout = 0
  ##ninout = None
  ##sfrom = ffrom = 'file'
  ##sto = 'stdout'
  ##fto = 'return'

  ##def options(self):
    ##self.add_option('-k', dest='k', default=12,
                    ##help='set number of eigenvalues to calculate')

  ##def main(self):
    ### load tight binding matrix
    ##self.load('__tbmat__')

    ##with progress.Bar(len(self.fin)*2, text='calculate energy range',
                      ##verbose=self.opts.verbose) as bar:
      ##for din, dout in itertools.izip(self.din, self.dout):
        ### calculate highest and lowest eigenvalues using Lanczos algorithm
        ##some_eigvals \
        ##= list(scipy.sparse.linalg.eigs(din.__tbmat__,
                                        ##k=self.opts.k/2,
                                        ##which='SR',
                                        ##return_eigenvectors=False).real)
        ##bar.step() # step 1/2
        ##some_eigvals \
        ##+= list(scipy.sparse.linalg.eigs(din.__tbmat__,
                                         ##k=self.opts.k-self.opts.k/2,
                                         ##which='LR',
                                         ##return_eigenvectors=False).real)

        ##emin = min(some_eigvals)
        ##emax = max(some_eigvals)

        ###emin -= abs(.1*emin) # add an extra 10% to be sure
        ###emax += abs(.1*emax) # add an extra 10% to be sure

        ### store result
        ##dout.erange = (emin, emax)
        ##bar.step() # step 2/2

    ### save data
    ##return self.save('erange')
