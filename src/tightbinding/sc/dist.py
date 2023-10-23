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
"""This submodule contains everything needed to generate random numbers from
various probability distributions, including a few predefined
distributions. Also, mathematical expressions consisting of such abstract
distribution definitions are possible.
"""

import math
import scipy
import scipy.stats


#=========================#
# Distribution base class #
#=========================#


class Distribution(object):
    """All distributions will share certain properties that are defined in this
    base class.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the distribution object.
        """
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        """Return detailed string representation.
        """
        attr = ['%s' % repr(value) for value in self.args]
        attr += ['%s=%s' % (key, repr(value))
                 for key, value in self.kwargs.items()]
        attrstr = ', '.join(attr)
        return ''+self.__class__.__name__+'('+attrstr+')'

    def evalmath(self, value, dtype=float):
        """Cast value to the given data type *dtype*. If *value* is a string,
        assume that it contains a mathematical expression, and evaluate it with
        :func:`eval` before casting it to the specified type. Functions are
        passed unmodified.

        The function could always use :func:`eval`, but this is assumed to be
        slower for values that do not have to be evaluated.
        """
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

    @staticmethod
    def ismath(string):
        """Check if the given string is a mathematical expression (containing
        only mathematical operators like '+', '-', '*', or '/', and of course
        digits). Can be used before using :func:`eval` on some string to
        evaluate a given expression.

        Note: This function does not check if the numerical expression is
        actually valid. It just gives a hint if the given string should be
        passed to :func:`eval` or not.
        """
        if '+' in string or '*' in string or '/' in string:
            return True

        # special handling of minus sign
        if string.count('-') == 1 and string[0] == '-':
            return False
        if '-' in string:
            return True
        return False

    def __str__(self):
        """Return short string representation.
        """
        return self.__class__.__name__

    def __neg__(self):
        return _NegDistribution(self)

    def __add__(self, other):
        """Add two probability distributions.
        """
        # create new object that adds the random values produced by calling
        # this and the other distribution and returns them
        return _AddDistributions(self, other)

    def __sub__(self, other):
        """Subtract two probability distributions from one another.
        """
        # create new object that adds the random values produced by calling
        # this and the other distribution and returns them
        return _SubDistributions(self, other)

    def __mul__(self, other):
        """Multiply two probability distributions.
        """
        # create new object that multiplies the random values produced by
        # calling this and the other distribution and returns them
        return _MulDistributions(self, other)

    def __div__(self, other):
        """Divide two probability distributions.
        """
        # create new object that divides the random values produced by calling
        # this and the other distribution and returns them
        return _DivDistributions(self, other)

    def is_complex(self):
        """Return whether the random numbers drawn from this probability
        distribution are complex.
        """
        return scipy.iscomplexobj(self(count_copies=False))


#===========================================================================#
# Classes representing mathematical expressions consisting of distributions #
#===========================================================================#


class _NegDistribution(Distribution):
    """Represent the negative of a distribution. When called, the distribution
    is called and the negative random variates are returned. Any parameters are
    passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return -self.dist(*args, **kwargs)
        else:
            return -self.dist

    def __repr__(self):
        """Return string representation."""
        return '-%s' % repr(self.dist)

    def __str__(self):
        """Return short string representation."""
        return '-%s' % str(self.dist)


class _SinDistribution(Distribution):
    """Represent the sine of a distribution. When called, the distribution is
    called and the sine of the random variates are returned. Any parameters are
    passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return scipy.sin(self.dist(*args, **kwargs))
        else:
            return scipy.sin(self.dist)

    def __repr__(self):
        """Return string representation.
        """
        return 'sin(%s)' % repr(self.dist)

    def __str__(self):
        """Return short string representation.
        """
        return 'sin(%s)' % str(self.dist)


class _CosDistribution(Distribution):
    """Represent the cosine of a distribution. When called, the distribution is
    called and the cosine of the random variates are returned. Any parameters
    are passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return scipy.cos(self.dist(*args, **kwargs))
        else:
            return scipy.cos(self.dist)

    def __repr__(self):
        """Return string representation."""
        return 'cos(%s)' % repr(self.dist)

    def __str__(self):
        """Return short string representation."""
        return 'cos(%s)' % str(self.dist)


class _TanDistribution(Distribution):
    """Represent the cosine of a distribution. When called, the distribution is
    called and the cosine of the random variates are returned. Any parameters
    are passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return scipy.tan(self.dist(*args, **kwargs))
        else:
            return scipy.tan(self.dist)

    def __repr__(self):
        """Return string representation.
        """
        return 'tan(%s)' % repr(self.dist)

    def __str__(self):
        """Return short string representation.
        """
        return 'tan(%s)' % str(self.dist)


class _ArcsinDistribution(Distribution):
    """Represent the sine of a distribution. When called, the distribution is
    called and the sine of the random variates are returned. Any parameters are
    passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return scipy.arcsin(self.dist(*args, **kwargs))
        else:
            return scipy.arcsin(self.dist)

    def __repr__(self):
        """Return string representation.
        """
        return 'arcsin(%s)' % repr(self.dist)

    def __str__(self):
        """Return short string representation.
        """
        return 'arcsin(%s)' % str(self.dist)


class _ArccosDistribution(Distribution):
    """Represent the cosine of a distribution. When called, the distribution is
    called and the cosine of the random variates are returned. Any parameters
    are passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return scipy.arccos(self.dist(*args, **kwargs))
        else:
            return scipy.arccos(self.dist)

    def __repr__(self):
        """Return string representation.
        """
        return 'arccos(%s)' % repr(self.dist)

    def __str__(self):
        """Return short string representation.
        """
        return 'arccos(%s)' % str(self.dist)


class _ArctanDistribution(Distribution):
    """Represent the cosine of a distribution. When called, the distribution is
    called and the cosine of the random variates are returned. Any parameters
    are passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return scipy.arctan(self.dist(*args, **kwargs))
        else:
            return scipy.arctan(self.dist)

    def __repr__(self):
        """Return string representation.
        """
        return 'arctan(%s)' % repr(self.dist)

    def __str__(self):
        """Return short string representation.
        """
        return 'arctan(%s)' % str(self.dist)


class _ExpDistribution(Distribution):
    """Represent the exponential of a distribution. When called, the
    distribution is called and the exponential of the random variates are
    returned. Any parameters are passed to the distribution object.
    """
    def __init__(self, dist):
        self.dist = dist
        self.isdist = isinstance(dist, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist:
            return scipy.exp(self.dist(*args, **kwargs))
        else:
            return scipy.exp(self.dist)

    def __repr__(self):
        """Return string representation.
        """
        return 'exp(%s)' % repr(self.dist)

    def __str__(self):
        """Return short string representation.
        """
        return 'exp(%s)' % str(self.dist)


class _AddDistributions(Distribution):
    """Represent the sum of two distributions. When called, the two
    distributions are called and the returned random variates are summed and
    returned. Any parameters are passed to the two distribution objects.
    """
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2
        self.isdist1 = isinstance(dist1, Distribution)
        self.isdist2 = isinstance(dist2, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist1 and self.isdist2:
            return self.dist1(*args, **kwargs)+self.dist2(*args, **kwargs)
        elif self.isdist1 and not self.isdist2:
            return self.dist1(*args, **kwargs)+self.dist2
        elif not self.isdist1 and self.isdist2:
            return self.dist1+self.dist2(*args, **kwargs)
        else:
            return self.dist1+self.dist2

    def __repr__(self):
        """Return string representation.
        """
        return '%s+%s' % (repr(self.dist1), repr(self.dist2))

    def __str__(self):
        """Return short string representation.
        """
        return '%s+%s' % (str(self.dist1), str(self.dist2))


class _MulDistributions(Distribution):
    """Represent the product of two distributions. When called, the two
    distributions are called and the returned random variates are multiplied
    and returned. All parameters are passed to the two distribution objects.
    """
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2
        self.isdist1 = isinstance(dist1, Distribution)
        self.isdist2 = isinstance(dist2, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist1 and self.isdist2:
            return self.dist1(*args, **kwargs)*self.dist2(*args, **kwargs)
        elif self.isdist1 and not self.isdist2:
            return self.dist1(*args, **kwargs)*self.dist2
        elif not self.isdist1 and self.isdist2:
            return self.dist1*self.dist2(*args, **kwargs)
        else:
            return self.dist1*self.dist2

    def __repr__(self):
        """Return string representation.
        """
        return '%s*%s' % (repr(self.dist1), repr(self.dist2))

    def __str__(self):
        """Return short string representation.
        """
        return '%s*%s' % (str(self.dist1), str(self.dist2))


class _DivDistributions(Distribution):
    """Represent the quotient of two distributions. When called, the two
    distributions are called and the returned random variates are divided and
    returned. All parameters are passed to the two distribution objects.
    """
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2
        self.isdist1 = isinstance(dist1, Distribution)
        self.isdist2 = isinstance(dist2, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist1 and self.isdist2:
            return self.dist1(*args, **kwargs)/self.dist2(*args, **kwargs)
        elif self.isdist1 and not self.isdist2:
            return self.dist1(*args, **kwargs)/self.dist2
        elif not self.isdist1 and self.isdist2:
            return self.dist1/self.dist2(*args, **kwargs)
        else:
            return self.dist1/self.dist2

    def __repr__(self):
        """Return string representation.
        """
        return '%s/%s' % (repr(self.dist1), repr(self.dist2))

    def __str__(self):
        """Return short string representation.
        """
        return '%s/%s' % (str(self.dist1), str(self.dist2))


class _SubDistributions(Distribution):
    """Represent the difference of two distributions. When called, the two
    distributions are called and the returned random variates are subtracted
    from each other and returned. Any parameters are passed to the two
    distribution objects.
    """
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2
        self.isdist1 = isinstance(dist1, Distribution)
        self.isdist2 = isinstance(dist2, Distribution)

    def __call__(self, *args, **kwargs):
        if self.isdist1 and self.isdist2:
            return self.dist1(*args, **kwargs)-self.dist2(*args, **kwargs)
        elif self.isdist1 and not self.isdist2:
            return self.dist1(*args, **kwargs)-self.dist2
        elif not self.isdist1 and self.isdist2:
            return self.dist1-self.dist2(*args, **kwargs)
        else:
            return self.dist1-self.dist2

    def __repr__(self):
        """Return string representation.
        """
        return '%s-%s' % (repr(self.dist1), repr(self.dist2))

    def __str__(self):
        """Return short string representation.
        """
        return '%s-%s' % (str(self.dist1), str(self.dist2))


#==================================#
# Define probability distributions #
#==================================#


class uniform(Distribution):
    """Generate random variates from a uniform distribution (box distribution)
    with the given parameters *loc* and *scale*.
    """
    def __init__(self, loc=0., scale=1., copies=1):
        """Initialize object.
        """
        self.loc = self.evalmath(loc)
        self.scale = self.evalmath(scale)
        self._copies = self.evalmath(copies, dtype=int)
        self.delivered = 0  # count how many copies of the same random numbers
                            # have been delivered
        ### evalmath should be used for all parameters of every distribution

        # needed for __repr__ only
        self.args = []
        self.kwargs = dict(loc=loc, scale=scale)

    def copies(self, copies):
        """Reset the copy counter and request the given number of copies.
        """
        self._copies = copies
        self.delivered = 0

    def __call__(self, *args, **kwargs):
        """Draw random variates.
        """
        # catch keyword arguments
        count_copies = kwargs.pop('count_copies', True)
        ### should be offered in all distribution classes

        # draw new random values
        if self.delivered == 0:
            self.variates = scipy.stats.uniform(loc=self.loc-self.scale/2.,
                                                scale=self.scale).rvs(*args,
                                                                      **kwargs)

        # increase counter (how many copies of the same random values have been
        # delivered)
        if count_copies:
            self.delivered += 1
            if self.delivered >= self._copies:
                # reset the counter
                self.delivered = 0

        # return random numbers
        return self.variates


def box(*args, **kwargs):
    """Alias for :class:`uniform`.
    """
    return uniform(*args, **kwargs)


class cauchy(Distribution):
    """Generate random variates from a Cauchy distribution (Lorentzian
    distribution) with the given parameters *loc* and *scale*.
    """
    def __init__(self, loc=0., scale=1., copies=1):
        """Initialize object."""
        # 2011-11-09
        self.loc = self.evalmath(loc)
        self.scale = self.evalmath(scale)
        self._copies = self.evalmath(copies, dtype=int)
        self.delivered = 0  # count how many copies of the same random numbers
                            # have been delivered

        # needed for __repr__ only
        self.args = []
        self.kwargs = dict(loc=self.loc, scale=self.scale)

    def __call__(self, *args, **kwargs):
        """Draw random variates.
        """
        # catch keyword arguments
        count_copies = kwargs.pop('count_copies', True)

        # draw new random values
        if self.delivered == 0:
            self.variates = \
                scipy.stats.cauchy(loc=self.loc,
                                   scale=self.scale/2.).rvs(*args, **kwargs)

        # increase counter (how many copies of the same random values have been
        # delivered)
        if count_copies:
            self.delivered += 1
            if self.delivered >= self._copies:
                # reset the counter
                self.delivered = 0

        # return random numbers
        return self.variates

    def copies(self, copies):
        """Reset the copy counter and request the given number of copies.
        """
        self._copies = copies
        self.delivered = 0


def lorentz(*args, **kwargs):
    """Alias for :class:`cauchy`."""
    return cauchy(*args, **kwargs)


class norm(Distribution):
    """Generate random variates from a normal distribution (Gaussian
    distribution) with the given parameters *loc* and *scale*.
    """
    def __init__(self, loc=0., scale=1., copies=1):
        """Initialize object.
        """
        self.loc = self.evalmath(loc)
        self.scale = self.evalmath(scale)
        self._copies = self.evalmath(copies, dtype=int)
        self.delivered = 0  # count how many copies of the same random numbers
                            # have been delivered

        # needed for __repr__ only
        self.args = []
        self.kwargs = dict(loc=self.loc, scale=self.scale)

    def __call__(self, *args, **kwargs):
        """Draw random variates.
        """
        # catch keyword arguments
        count_copies = kwargs.pop('count_copies', True)

        # draw new random values
        if self.delivered == 0:
            self.variates = \
                scipy.stats.norm(loc=self.loc,
                                 scale=self.scale/math.sqrt(12.)).rvs(*args,
                                                                      **kwargs)

        # increase counter (how many copies of the same random values have been
        # delivered)
        if count_copies:
            self.delivered += 1
            if self.delivered >= self._copies:
                # reset the counter
                self.delivered = 0

        # return random numbers
        return self.variates

    def copies(self, copies):
        """Reset the copy counter and request the given number of copies.
        """
        self._copies = copies
        self.delivered = 0


def gauss(*args, **kwargs):
    """Alias for :class:`norm`.
    """
    return norm(*args, **kwargs)


class triang(Distribution):
    """Return function object to generate random variates from a triangular
    distribution with the given parameters *loc* and *scale*.
    """
    def __init__(self, loc=0., scale=1., copies=1):
        self.loc = self.evalmath(loc)
        self.scale = self.evalmath(scale)
        self._copies = self.evalmath(copies, dtype=int)
        self.delivered = 0  # count how many copies of the same random numbers
                            # have been delivered

        # needed for __repr__ only
        self.args = []
        self.kwargs = dict(loc=self.loc, scale=self.scale)

    def __call__(self, *args, **kwargs):
        """Draw random variates.
        """
        # catch keyword arguments
        count_copies = kwargs.pop('count_copies', True)

        # draw new random values
        if self.delivered == 0:
            self.variates = \
                scipy.stats.triang(.5, loc=self.loc-self.scale/2,
                                   scale=self.scale).rvs(*args, **kwargs)

        # increase counter (how many copies of the same random values have been
        # delivered)
        if count_copies:
            self.delivered += 1
            if self.delivered >= self._copies:
                # reset the counter
                self.delivered = 0

        # return random numbers
        return self.variates

    def copies(self, copies):
        """Reset the copy counter and request the given number of copies.
        """
        self._copies = copies
        self.delivered = 0


class binary(Distribution):
    """Generate random variates from a discrete binary distribution with the
    given mixing ratio *mix* and the values *a* and *b*.
    """
    def __init__(self, mix=.5, a=0., b=1., copies=1):
        """Initialize object.
        """
        self.mix = self.evalmath(mix)
        self.a = self.evalmath(a)
        self.b = self.evalmath(b)
        self._copies = self.evalmath(copies, dtype=int)
        self.delivered = 0  # count how many copies of the same random numbers
                            # have been delivered

        self.args = []
        self.kwargs = dict(mix=self.mix, a=self.a, b=self.b)

    def __call__(self, *args, **kwargs):
        """Draw random variates.
        """
        # catch keyword arguments
        count_copies = kwargs.pop('count_copies', True)
        ### should be offered in all distribution classes

        if self.delivered == 0:
            # get size
            if len(args) > 0:
                size = args[0]
            else:
                size = kwargs.get('size', None)

            # generate random variates
            m = [1-self.kwargs['mix'], self.kwargs['mix']]
            r = range(2)
            values = scipy.array([self.kwargs['a'], self.kwargs['b']])
            inds = scipy.stats.rv_discrete(values=(r, m),
                                           name='binary').rvs(size=size)
            self.variates = values[inds]

        # increase counter (how many copies of the same random values have been
        # delivered)
        if count_copies:
            self.delivered += 1
            if self.delivered >= self._copies:
                # reset the counter
                self.delivered = 0

        # return random numbers
        return self.variates

    def copies(self, copies):
        """Reset the copy counter and request the given number of copies.
        """
        self._copiesb = copies
        self.deliveredb = 0


class multi(Distribution):
    """Generate random variates from a discrete distribution with the given
    values and mixing ratios, specified by arbitrary many value-ratio pairs
    (2-tuples).
    """
    def __init__(self, *values, **kwargs):
        """Initialize object.
        """
        self._copies = self.evalmath(kwargs.pop('copies', 1), dtype=int)
        self.delivered = 0  # count how many copies of the same random numbers
                            # have been delivered
        if len(kwargs) > 0:
            raise ValueError('there were unknown keyword arguments')

        self.args = values
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        """Draw random variates.
        """
        # catch keyword arguments
        count_copies = kwargs.pop('count_copies', True)
        ### should be offered in all distribution classes

        if self.delivered == 0:
            # get size
            if len(args) > 0:
                size = args[0]
            else:
                size = kwargs.get('size', None)

            # generate random variates
            val, mix = zip(*self.args)
            r = range(len(self.args))
            inds = scipy.stats.rv_discrete(values=(r, mix),
                                           name='multi').rvs(size=size)
            val = scipy.array(val)
            self.variates = val[inds]

        # increase counter (how many copies of the same random values have been
        # delivered)
        if count_copies:
            self.delivered += 1
            if self.delivered >= self._copies:
                # reset the counter
                self.delivered = 0

        # return random numbers
        return self.variates


# def copies(self, copies):
#     """Reset the copy counter and request the given number of copies."""
#     self._copies = copies
#     self.delivered = 0


#=========================================================#
# Select one of the distributions based on a string input #
#=========================================================#


def choose(string):
    """Select one of the probability distributions defined in this module
    according to a given string (e.g. to evaluate a command line parameter).
    """
    if 'uniform'.startswith(string.lower()) \
            or 'box'.startswith(string.lower()):
        return uniform
    elif 'cauchy'.startswith(string.lower()) \
            or 'lorentz'.startswith(string.lower()):
        return cauchy
    elif 'normal'.startswith(string.lower()) \
            or 'gauss'.startswith(string.lower()):
        return norm
    elif 'triangular'.startswith(string.lower()):
        return triang
    elif 'binary'.startswith(string.lower()):
        return binary
    elif 'multi'.startswith(string.lower()):
        return multi
    else:
        raise NameError('no probability distribution found for string %s'
                        % string)


def select(string):
    """Alias for :func:`choose`.
    """
    return choose(string)


#==================#
# Math definitions #
#==================#


def sin(x):
    """Return the sine of a distribution.
    """
    return _SinDistribution(x)


def cos(x):
    """Return the cosine of a distribution.
    """
    return _CosDistribution(x)


def tan(x):
    """Return the tangent of a distribution.
    """
    return _TanDistribution(x)


def arcsin(x):
    """Return the sine of a distribution.
    """
    return _ArcsinDistribution(x)


def arccos(x):
    """Return the cosine of a distribution.
    """
    return _ArccosDistribution(x)


def arctan(x):
    """Return the tangent of a distribution.
    """
    return _ArctanDistribution(x)


def exp(x):
    """Return the exponential of a distribution.
    """
    return _ExpDistribution(x)


class _Pi(Distribution):
    """Represent the number pi symbolically (technically a distribution
    object). Evaluate only on call.

    Just exists to allow expressions consisting of distribution objects and
    including the number pi, e.g. "dist = pi*binary(a=-1, b=1, mix=.1)" or
    similar expressions are possible in this way, being evaluated only on call
    ``dist()`` (in the above example, about 10 % of the resulting random
    numbers will be -pi, 90 % will be +pi).

    The above example is obviously equivalent to ``binary(a=-pi, b=pi,
    mix=.1)``. It merely serves as an example.
    """
    def __call__(self, *args, **kwargs):
        return math.pi

    def __repr__(self):
        """Return string representation.
        """
        return 'pi'

    def __str__(self):
        """Return short string representation.
        """
        return repr(self)


pi = _Pi()
