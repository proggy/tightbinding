#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2013 Daniel Jung
# Contact: d.jung@jacobs-university.de
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
"""Define tight-binding related plot functions."""
__created__ = '2013-07-25'
__modified__ = '2013-07-25'


##class _Pmperf(hdp.PlotHDP):
  ##"""Plot performance chart for generating tight binding matrices.
  ##Plot elapsed time versus system size.

  ##To do:
  ##--> implement options for (semi-)logarithmic plots (maybe already in
  ##    PlotHDP)
  ##--> make chart with colored areas stacked on top of each other
  ##--> average over files of equal system size, thus obtain better precision
      ##by just passing more and more files
  ##--> sort input data, to be able to connect the data points throuth lines
      ##correctly"""
  ### 2012-02-01 - 2012-02-06
  ##version = '2012-02-06'

  ##def __init__(self):
    ##hdp.PlotHDP.__init__(self)

    ### set options
    ##self.op.add_option('-x', '--xdata', dest='xdata', default='N', type=str,
                       ##help='set x-axis data source. N: system size (total '+
                            ##'number of unitcells), L: system edge length '+\
                            ##'(number of unitcells in first dimension')
    ##self.op.add_option('-y', '--ydata', dest='ydata', default='t', type=str,
                       ##help='set y-axis data. '+\
                            ##'c: calculation time, l: load time, '+\
                            ##'p: preparation time, t: total elapsed time, '+\
                            ##'s: save time')

  ##def __call__(self, *args, **kwargs):
    ##hdp.PlotHDP.__call__(self, *args, **kwargs)
    ##import matplotlib.pyplot as plt
    ##from itertools import izip
    ##from tb.misc import prod

    ### configure the plot
    ##self.propose(xlabel=self.opts.xdata, ylabel='T [s]',
                 ##title='', grid=True, legend='on', marker='D')

    ### define dictionary with nice string representations for the respective
    ### y-axis data characters used in the ydata option
    ##char2str = {'p': 'prepare',
                ##'l': 'load',
                ##'c': 'calculation',
                ##'s': 'save',
                ##'t': 'total'}

    ### load data, only comments and parameter sets are needed
    ##self.load()

    ### select data, cycle through input files
    ##xdata = []; ydata = []
    ##for char in self.opts.ydata:
      ##ydata.append([])
    ##for cin, pin in izip(self.cin, self.pin):
      ### get x-axis data
      ##if self.opts.xdata == 'N':
        ##xdata.append(prod(pin.shape))
      ##elif self.opts.xdata == 'L':
        ##xdata.append(pin.shape[0])
      ##else:
        ##self.op.error('bad xdata option: %s' % self.opts.xdata)

      ### get y-axis data
      ##for cind, char in enumerate(self.opts.ydata):
        ##if char == 'p':
          ##ydata[cind].append(cin.tbmat.get('elapsed_pre', None))
        ##elif char == 'l':
          ##ydata[cind].append(cin.tbmat.get('elapsed_load', None))
        ##elif char == 'c':
          ##ydata[cind].append(cin.tbmat.get('elapsed_calc', None))
        ##elif char == 's':
          ##ydata[cind].append(cin.tbmat.get('elapsed_save', None))
        ##elif char == 't':
          ##ydata[cind].append(cin.tbmat.get('elapsed', None))
        ##else:
          ##self.op.error('bad ydata option: %s' % self.opts.ydata)

    ### plot data
    ##for cind, char in enumerate(self.opts.ydata):
      ##plt.plot(xdata, ydata[cind], label=char2str.get(char, ''))

    ### finish the plot
    ##self.finish()
