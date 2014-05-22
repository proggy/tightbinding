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
"""General collection of supplementary functions and classes that do not
really fit in anywhere else.

Performance of some of the functions is enhanced using Cython."""
# 2011-09-13 - 2012-05-08
import numpy
import os
import commands


def allscalar(seq):
    """Test if all elements of the sequence or set are scalar."""
    # 2011-09-13
    # former tb.allscalar from 2011-03-30
    for element in seq:
        if isiterable(element):
            return False
    return True


def anyobject(seq):
    """Test if any element of the sequence or set is an instance of a
    user-defined class."""
    # 2011-09-13
    # former tb.anyobject from 2011-03-30
    for element in seq:
        if isobject(element):
            return True
    return False


def basename(path, suffix=''):
    """My variation of os.path.basename, in which a given suffix can be
    canceled out (like in the php version of this function)."""
    # 06/01/2011-28/01/2011
    # former mytools.basename() from 05/01/2010
    basename = os.path.basename(path)
    suffixlen = len(suffix)
    if suffixlen > 0:
        if basename[-suffixlen:] == suffix:
            basename = basename[:-suffixlen]
    return basename


def capitalize(string):
    """Return string with first letter in uppercase and the rest of the
    characters in lowercase."""
    # 2011-09-13
    # former tb.capitalize from 2011-01-28
    if not typename(string) == 'str':
        raise TypeError('string expected')
    return string[:1].upper()+string[1:].lower()


def commonprefix(*args):
    """Finds the common prefix of all given strings."""
    # 2011-09-13
    # former tb.commonprefix from 2011-06-09

    # Handle two special cases
    if len(args) == 0:
        return ''
    if len(args) == 1:
        return args[0]

    # Start search
    firstarg = args[0]
    for cind in xrange(len(firstarg)):  # Character index
        for arg in args[1:]:
            if cind > len(arg)-1 or firstarg[cind] != arg[cind]:
                return firstarg[:cind]

    # Return common prefix
    return firstarg


def eqlist(**dictionary):
    """Print a dictionary to stdout as a list of elements in the form "key =
    value", aligning all equality signs. Also all nested contents are formatted
    nicely in this way. If the keyword "maxshape" is specified, display only
    the contents of those numpy arrays that do not have a dimension larger than
    the specified number. Instead, just note the shape of the array. For sparse
    matrices, maxshape gives the maximum number of nonzero elements. If the
    keyword "dense" is set to True, sparse matrices are converted to dense
    matrices before displaying."""
    # 2011-09-13
    # former tb.eqlist from 2011-02-10 - 2011-02-24

    # Handle special keyword arguments
    if 'maxshape' in dictionary:
        maxshape = dictionary.pop('maxshape')
    else:
        maxshape = 15  # default value
    if 'dense' in dictionary:
        dense = dictionary.pop('dense')
    else:
        dense = False  # default value

    # Local function definition
    def strrep(value, indent=0):
        """Return string representation of value, but in multirow format,
        with the given indentation after each line break."""

        # Is value a sparse matrix, and has the dense keyword been set to True?
        if typename(value) in ['csr_matrix', 'csc_matrix'] and dense:
            value = value.todense()

        # 10/02/2011
        if typename(value) == 'list':
            if 'list' in types(value) or 'tuple' in types(value) \
                    or 'dict' in types(value) or 'struct' in types(value) \
                    or 'ndarray' in types(value):
                return '[' + \
                       (',\n' +
                        ' '*(indent+1)).join(strrep(v, indent=indent+1)
                                             for v in value)+']'
            else:
                if len(value) <= maxshape:
                    return str(value)
                else:
                    return 'list of length '+str(len(value))
        elif typename(value) == 'tuple':
            if 'list' in types(value) or 'tuple' in types(value) \
                    or 'dict' in types(value) or 'struct' in types(value) \
                    or 'ndarray' in types(value):
                return '(' + \
                       (',\n' +
                        ' '*(indent+1)).join(strrep(v, indent=indent+1)
                                             for v in value)+')'
            else:
                if len(value) <= maxshape:
                    return str(value)
                else:
                    return 'tuple of length '+str(len(value))
        elif typename(value) == 'dict':
            # Build key-value pairs
            if len(value) > 0:
                maxkeylen = max([len(key) for key in value])
            else:
                maxkeylen = 0
            kvpairs = []
            for key in value:
                kvpairs.append(expandstr(key, length=maxkeylen) + ': ' +
                               strrep(value[key], indent=indent+1+maxkeylen+2))
            #maxpairlen = max([len(p) for p in kvpairs])
            return '{'+(',\n'+' '*(indent+1)).join(kvpairs)+'}'
        elif typename(value) == 'struct':
            # Build key-value pairs
            if len(value) > 0:
                maxkeylen = max([len(key) for key in value])
            else:
                maxkeylen = 0
            kvpairs = []
            for key in value:
                kvpairs.append(expandstr(key, length=maxkeylen) + ' = ' +
                               strrep(value[key], indent=indent+7+maxkeylen+3))
            #maxpairlen = max([len(p) for p in kvpairs])
            return 'struct('+(',\n'+' '*(indent+7)).join(kvpairs)+')'
        elif typename(value) in ['ndarray', 'matrix']:
            # Distinguish different dimensionalities
            #if len(value.shape) < 2:
            #  strvalue = str(value)
            #else:
            # Only print the contents of an array if it is not too big
            if numpy.all(numpy.array(value.shape) <= maxshape):
                strvalue = str(value)
            else:
                strvalue = 'ndarray with shape '+str(value.shape)

            # Make sure that the string representation of the array has
            # the right indentation at each line break
            if '\n' in strvalue:
                l = [s.strip() for s in strvalue.split('\n')]
                strvalue = ('\n'+' '*(indent+1)).join(l)

            # Return string representation of the array
            return strvalue
        elif typename(value) in ['csr_matrix', 'csc_matrix', 'lil_matrix',
                                 'dia_matrix', 'coo_matrix', 'bsc_matrix',
                                 'dok_matrix']:
            strvalue = repr(value)

            # show data if number of nonzero elements not bigger than maxshape
            if value.nnz <= maxshape:
                """lines = str(value).split('\n')
                strarray = []
                for line in lines:
                    strarray.append(line.split('\t'))
                strarray = numpy.array(strarray)
                maxlen0 = max([len(s) for s in strarray[:, 0]])
                maxlen1 = max([len(s) for s in strarray[:, 1]])
                for row in strarray:
                    str0 = expandstr(row[0], length=maxlen0)
                    str1 = expandstr(row[1], length=maxlen1, flip=True)
                    strvalue += '\n'+str0+' = '+str1"""
                value = value.tocoo()
                maxrowlen = max([len(str(r)) for r in value.row])
                maxcollen = max([len(str(c)) for c in value.col])
                maxdatalen = max([len(str(d)) for d in value.data])
                for row, col, data in zip(value.row, value.col, value.data):
                    rowstr = expandstr(row, length=maxrowlen, flip=True)
                    colstr = expandstr(col, length=maxcollen, flip=True)
                    datastr = expandstr(data, length=maxdatalen, flip=True)
                    strvalue += '\n(%s, %s) = %s' % (rowstr, colstr, datastr)

            # Make sure that the string representation of the array has
            # the right indentation at each line break
            if '\n' in strvalue:
                l = [s.strip() for s in strvalue.split('\n')]
                strvalue = ('\n'+' '*indent).join(l)

            # Return string representation of the array
            return strvalue
        else:
            return str(value)

    # Print to stdout
    if len(dictionary) == 0:
        return False
    maxkeylen = max([len(key) for key in dictionary])
    keys = dictionary.keys()
    keys.sort()
    for key in keys:
        print expandstr(key, length=maxkeylen)+' = '+strrep(dictionary[key],
                                                            indent=maxkeylen+3)


def equal(*objects):
    """My version of the function "all" with equality check (==), that
    works with all types of objects, even with scalars and nested lists."""
    # 2011-09-13
    # former tb.equal from 2011-02-09
    # former mytools.equal
    if len(objects) > 2:
        a = objects[0]
        for b in objects[1:]:
            if not equal(a, b):
                return False
        return True
    assert len(objects) == 2, 'at least two objects have to be specified'
    a, b = objects

    if not isiterable(a) and not isiterable(b):
        return a == b
    elif isiterable(a) and isiterable(b):
        if not isobject(a) and not isobject(b):
            if not len(a) == len(b):
                return False
            else:
                for i in xrange(len(a)):
                    if not equal(a[i], b[i]):
                        return False
                return True
        elif isobject(a) and isobject(b):
            return a.__dict__ == b.__dict__
        else:
            return False
    else:
        return False


def equaltype(seq):
    """Test if all elements of the sequence or set have the same type."""
    # 2011-09-13
    # former tb.equaltype from 2011-03-30
    seq = list(seq)
    first = seq.pop()
    for element in seq:
        if typename(element) != typename(first):
            return False
    return True


def expandstr(string, length=None, ch=' ', flip=False):
    """Return the string, but expand it to the given length with the given
    character.  If flip is True, fill characters in front of string instead of
    behind it."""
    # 2011-09-13 - 2012-05-03
    # former tb.expandstr from 2011-02-09
    # former mytools.expandstr
    string = str(string)
    lenght = int(length)

    l = len(string)
    if length is None:
        length = l

    output = ''

    if flip:
        output += ch*(length-l)
        output += string
    else:
        output += string
        output += ch*(length-l)

    return output


def typename(obj):
    """Just a shortcut to return the name of the type of the given object."""
    # 2011-09-13
    # former tb.typename from 2011-02-10
    return type(obj).__name__


def dump(*args, **kwargs):
    """Print values of the passed arguments to the screen, together with
    additional information about the value, like length or shape of a list or
    an array. The name of the value may be specified by using keyword
    arguments."""
    # 2011-09-13
    # former tb.dump from 2011-01-10 - 2011-04-06
    # former mytools.dump from 20/04/2010
    for arg in args:
        if typename(arg) == 'ndarray':
            print arg.shape, arg
        elif typename(arg) == 'list':
            print len(arg), arg
        else:
            print arg
    for key, value in kwargs.iteritems():
        if typename(value) == 'ndarray':
            print key, value.shape, value
        elif typename(value) == 'list':
            print key, len(value), value
        else:
            print key, value


def humanbytes(bytes):
    """Return given byte count (integer) in a human readable format (string).
    Example: 664070 --> 649Ki.  Supported binary prefixes: kibi, mebi, gibi,
    tebi, pebi, exbi, zebi, yobi."""
    # 2011-09-13
    # former tb.humanbytes from 2011-02-13
    # former mytools.humanbytes
    assert typename(bytes) in ['int', 'long'], \
        'integer expected, but value of type %s given' % typename(bytes)
    assert bytes >= 0, \
        'bad byte count: %i. Must be non-negative integer' % bytes

    # define units
    unittable = {0: '',
                 1: 'Ki',
                 2: 'Mi',
                 3: 'Gi',
                 4: 'Ti',
                 5: 'Pi',
                 6: 'Ei',
                 7: 'Xi',
                 8: 'Yi'}

    # calculate human readable string representation
    i = bytes
    u = 0
    while i > 1024:
        if not u+1 in unittable:
            break
        i = int(round(float(i)/1024))
        u += 1
    return str(i)+unittable[u]


def nicetime(seconds):
    """Return nice string representation of the given number of seconds in a
    human-readable format (approximated). Example: 3634 s --> 1 h."""
    # 2012-02-17
    from itertools import izip

    # create list of time units (must be sorted from small to large units)
    units = [{'factor': 1,  'name': 'sec'},
             {'factor': 60, 'name': 'min'},
             {'factor': 60, 'name': 'hrs'},
             {'factor': 24, 'name': 'dys'},
             {'factor': 7,  'name': 'wks'},
             {'factor': 4,  'name': 'mns'},
             {'factor': 12, 'name': 'yrs'}]

    value = int(seconds)
    for unit1, unit2 in izip(units[:-1], units[1:]):
        if value/unit2['factor'] == 0:
            return '%i %s' % (value, unit1['name'])
        else:
            value /= unit2['factor']
    return '%i %s' % (value, unit2['name'])


def isiterable(obj):
    """Check if an object is iterable. Return True for lists, tuples,
    dictionaries and numpy arrays (all objects that possess an __iter__
    method).  Return False for scalars (float, int, etc.), strings, bool and
    None."""
    # 2011-09-13
    # former tb.isiterable from 2011-01-27
    # former mytools.isiterable
    # Initial idea from
    # http://bytes.com/topic/python/answers/514838-how-test-if-object-sequence-
    # iterable:
    # return isinstance(obj, basestring) or getattr(obj, '__iter__', False)
    # I found this to be better:
    return not getattr(obj, '__iter__', False) is False


def isobject(obj):
    """Return True if obj possesses an attribute called "__dict__", otherwise
    return False."""
    # 2011-09-13
    # former tb.isobject from 2011-02-09
    # former mytools.isobject
    return not getattr(obj, '__dict__', False) is False


def nearest(a, v):
    """Return index of the nearest value in a to v."""
    # 2011-09-13
    # former tb.nearest from 2011-06-17
    # former mytools.nearest
    return numpy.argmin([numpy.abs(i-v) for i in a])


def npc():
    """Return number of processor cores on this machine. Supported operating
    systems: Linux/Unix, MacOS, Windows."""
    # 2011-09-13
    # former tb.npc from 2011-02-10
    # former mytools.detectCPUs
    # based on code from http://www.boduch.ca/2009/06/python-cpus.html

    # Linux, Unix and Mac OS
    if hasattr(os, 'sysconf'):
        if 'SC_NPROCESSORS_ONLN' in os.sysconf_names:
            # Linux and Unix
            npc = os.sysconf('SC_NPROCESSORS_ONLN')
            if isinstance(npc, int) and npc > 0:
                return npc
        else:
            # Mac OS:
            return int(os.popen2('sysctl -n hw.ncpu')[1].read())

    # Windows
    if 'NUMBER_OF_PROCESSORS' in os.environ:
        npc = int(os.environ['NUMBER_OF_PROCESSORS'])
        if npc > 0:
            return npc

    # Otherwise, return default value
    return 1


def opt2list(opt, dtype=int):
    """Return list of values as specified by an option string of the form
    "x1,x2,x3,x4,x5". dtype specifies the type of the values (int, float,
    str, etc.). Default: int."""
    # 2011-09-13 - 2011-11-29
    # former tb.opt2list from 2011-01-31
    if opt is None or opt == '':
        return []
    if typename(opt) != 'str':
        raise ValueError('string input required')
    else:
        if dtype == str:
            result = []
            for r in opt.split(','):
                if not r == '':
                    result.append(r)
            return result
        else:
            return [dtype(value) if value != '' else dtype(0)
                    for value in opt.split(',')]


def opt2mathlist(opt, dtype=int):
    """Return list of values as specified by an option string of the form
    "x1,x2,x3,x4,x5". dtype specifies the type of the values (int, float, str,
    etc.). If mathematical expressions are contained inside the option strings,
    they are evaluated. Default: int."""
    __created__ = '2012-02-19'
    __modified__ = '2012-10-09'
    if type(opt) in (int, long, float):
        return [dtype(opt)]
    if type(opt) in (list, tuple):
        opt = ','.join([str(item) for item in opt])
    if not opt:
        return []
    if typename(opt) != 'str':
        raise ValueError('string input required')
    result = []
    for r in opt.split(','):
        if r == '':
            if dtype == str:
                continue
            else:
                r = 0
        if ismath(r):
            r = evalmath(r, dtype=dtype)
        result.append(dtype(r))
    return result


def opt2range(opt, lower=None, upper=None):
    """Return list of integers as specified by an option string of the form
    "x:y:z". z has to be a positive integer, x and y have to be non-negative
    integers. Use lower and upper to define the lower and upper limits. In this
    way, the user can specify ranges by leaving x or y empty."""
    # 2011-09-13
    # former tb.opt2range from 2011-01-29 - 2011-02-24
    assert typename(opt) == 'str', 'string input required'
    strlist = opt.split(':')
    if len(strlist) > 0 and strlist[0] == '' and lower is not None:
        strlist[0] = int(lower)
    if len(strlist) > 1 and strlist[1] == '' and upper is not None:
        strlist[1] = int(upper)
    intlist = [int(m) if m != '' else 0 for m in strlist]
    assert len(intlist) in [1, 2, 3], \
        'bad order range: %s. Only one, two or three values ' % opt + \
        'may be given, separated by a colon (:)'
    if len(intlist) < 2:
        intlist.append(intlist[0]+1)
    if len(intlist) < 3:
        intlist.append(1)
    assert all([b >= 0 for b in intlist]), \
        'bad order range: Only positive integers or zero allowed.'
    result = range(*intlist)
    result.sort()
    return result


def opt2ranges(opt, lower=None, upper=None):
    """Return list of integers as specified by an option string of the form
    "x1:y1:z1,x2:y2:z2,...". Each z has to be a positive integer, x and y have
    to be positive integers or zero. Use lower and upper to define the lower
    and upper limits for the ranges. In this way, the user can specify ranges
    by leaving x or y empty."""
    # 2011-09-13
    # former tb.opt2ranges from 2011-01-29 - 2011-02-24
    assert typename(opt) == 'str', 'string input required'
    rangelist = opt.split(',')
    resultset = set()
    for rangestr in rangelist:
        resultset.update(opt2range(rangestr, lower=lower, upper=upper))
    result = list(resultset)
    result.sort()
    return result


def ismath(string):
    """Check if the given string is a mathematical expression (containing only
    mathematical operators like '+', '-', '*', or '/', and of course digits).
    Can be used before using "eval" on some string to evaluate a given
    expression.

    Note: This function does not check if the numerical expression is actually
    valid. It just gives a hint if the given string should be passed to eval or
    not."""
    # 2011-09-13 - 2011-10-12
    # former tb.mathexpr from 2011-06-12
    if '+' in string or '*' in string or '/' in string:
        return True

    # Special handling of minus sign
    if string.count('-') == 1 and string[0] == '-':
        return False
    if '-' in string:
        return True
    return False


def evalmath(value, dtype=float):
    """Cast value to the given data type (dtype). If value is a string, assume
    that it contains a mathematical expression, and evaluate it with eval
    before casting it to the specified type.

    The function could always use eval, but this is assumed to be slower for
    values that do not have to be evaluated."""
    # 2011-10-12
    if type(value).__name__ == 'str' and ismath(value):
        return dtype(eval(value))
    else:
        return dtype(value)


def printcols(strlist, ret=False):
    """Print the strings in the given list in column by column (similar the
    bash command "ls"), respecting the width of the shell window. If ret is
    True, give back the resulting string instead of printing it to stdout."""
    # 2011-09-13
    # former tb.printcols from 2011-02-13 - 2011-08-02
    # former mytools.printcols
    numstr = len(strlist)

    # Try to get the width of the shell window (will only work on Unix systems)
    try:
        cols = int(commands.getoutput('tput cols'))-1
    except ValueError:
        cols = 80

    # Determine the maximum string width
    maxwidth = max([len(s) for s in strlist])

    # Calculate number of columns
    numcols = cols/(maxwidth+2)

    # Calculate number of required rows
    numrows = int(numpy.ceil(1.*numstr/numcols))

    # Print
    result = ''
    for rind in xrange(numrows):  # row index
        for cind in xrange(numcols):  # column index
            sind = cind*numrows+rind  # string list index
            if sind < numstr:
                result += strlist[sind]+' '*(maxwidth-len(strlist[sind])+1)
                #print strlist[sind]+' '*(maxwidth-len(strlist[sind])+1),
        result += '\n'

    if ret:
        return result
    else:
        print result


def prod(seq):
    """Return the product of all elements of the given sequence.

    I use this function mainly to avoid importing the huge numpy module (which
    takes up to 1.5 seconds on my computer)."""
    # 2011-10-12
    if len(seq) == 0:
        return 0
    elif len(seq) == 1:
        return seq[0]
    else:
        prod = seq[0]
        for elem in seq[1:]:
            prod *= elem
        return prod


def sepnumstr(string):
    """Separate numeric values from characters within a string. Return
    resulting numeric values and strings as a list."""
    # 2011-09-13
    # former tb.sepnumstr from 2011-02-03 - 2011-04-06
    if not typename(string) == 'str':
        raise TypeError('string expected')

    # If string is empty, just return empty list
    if string == '':
        return []

    numchars = '-0123456789.'  # characters that belong to a numeric value
    result = []
    currval = ''  # current value
    currisnum = string[0] in numchars  # if current value is numeric
    for char in string:
        if (char in numchars) == currisnum:
            currval += char
        else:
            if currisnum:
                if currval == '.':
                    result.append(0.)
                elif '.' in currval:
                    result.append(float(currval))
                else:
                    result.append(int(currval))
            else:
                result.append(currval)
            currval = char
            currisnum = not currisnum

    # Add last value
    if currisnum:
        if currval == '.':
            result.append(0.)
        elif '.' in currval:
            result.append(float(currval))
        else:
            result.append(int(currval))
    else:
        result.append(currval)

    # Return result
    return result


class tic:
    """Implements a object-oriented Python version of the Matlab tic and toc
    functions."""
    # 2011-09-13
    # former tb.tic from 2011-02-27

    def __init__(self):
        from time import time
        self._start = time()

    def toc(self):
        from time import time
        return time()-self._start


def types(value):
    """Return list of typenames that the iterable value contains."""
    # 2011-11-15
    # former tb.types from 2011-02-10
    if not type(value).__name__ in ['list', 'tuple', 'dict', 'struct']:
        raise TypeError('expecting value of type list, tuple, dict or struct')
    # should be enabled for any iterable, also for set and frozenset
    types = []
    for v in value:
        if type(value).__name__ in ('dict', 'struct'):
            types.append(type(value[v]).__name__)
        else:
            types.append(type(v).__name__)
    return types


class Tab(object):
    """Display data in form of a table, print to stdout."""
    # 2011-12-18 - 2011-12-19

    def __init__(self, titles=None, sep='  ', head=False, width=None):
        """Initialize object to hold table data. Specify the columns in the
        order given by titles.Separate columns by sep. Show head row if head is
        True."""
        # 2011-12-18 - 2011-12-19
        if titles is not None and not isiterable(titles):
            self.titles = ['']*int(titles)
        else:
            self.titles = titles
        self.sep = sep
        self.rowdata = []
        self.head = head
        self.headdata = []
        self.width = width

    def add(self, **rowdata):
        """Add a data row to the table."""
        # 2011-12-18 - 2011-12-19

        # determine column titles, if not done already
        if self.titles is None:
            self.titles = rowdata.keys()
            self.titles.sort()
        elif len(self.titles) < len(rowdata):
            # add new titles
            for title in rowdata.keys():
                if title not in self.titles:
                    self.titles.append(title)

        self.rowdata.append(rowdata)

    def display(self):
        """Display the table (write to stdout)."""
        # 2011-12-18 - 2011-12-19

        # make sure all row data contain all columns
        for ind in xrange(len(self.rowdata)):
            for title in self.titles:
                if not title in self.rowdata[ind]:
                    self.rowdata[ind][title] = ''

        # determine width of the console (only on Unix)
        if self.width is None:
            try:
                console_width = int(os.environ['COLUMNS'])
            except (KeyError, ValueError):
                console_width = 80
        else:
            console_width = self.width

        # convert all data to string representation
        for ind in xrange(len(self.rowdata)):
            for key in self.rowdata[ind].keys():
                if type(self.rowdata[ind][key]).__name__ == 'matrix':
                    self.rowdata[ind][key] = '<matrix %ix%i %s>' \
                                             % self.rowdata[ind][key].shape \
                                             + (self.rowdata[ind][key].dtype,)
                elif type(self.rowdata[ind][key]).__name__ == 'ndarray':
                    self.rowdata[ind][key] = '<ndarray %s %s>' \
                        % ('x'.join(str(i)
                           for i in self.rowdata[ind][key].shape),
                           self.rowdata[ind][key].dtype)
                elif type(self.rowdata[ind][key]).__name__.endswith('_matrix')\
                        and hasattr(self.rowdata[ind][key], 'format'):
                    self.rowdata[ind][key] = '<%s %ix%i %s>' \
                        % ((self.rowdata[ind][key].format,)
                           + self.rowdata[ind][key].shape
                           + (self.rowdata[ind][key].dtype,))
                else:
                    self.rowdata[ind][key] = str(self.rowdata[ind][key])

        # determine maximal string width of each column
        widths = {}
        for ind in xrange(len(self.rowdata)):
            for key in self.rowdata[ind].keys():
                if key in widths:
                    widths[key] = max(widths[key], len(self.rowdata[ind][key]))
                else:
                    widths[key] = len(self.rowdata[ind][key])
        if self.head:
            for title in self.titles:
                if title in widths:
                    widths[title] = max(widths[title], len(title))
                else:
                    widths[title] = len(title)

        # print table to stdout
        if self.head:
            headline = self.sep.join(expandstr(title, length=widths[title])
                                     for title in self.titles)
            print headline[:(console_width-1)]
        for rd in self.rowdata:
            # build string line
            line = self.sep.join(expandstr(rd[title], length=widths[title])
                                 for title in self.titles)
            print line[:(console_width-1)]


def minima(a, include_boundaries=False):
    """Find indices of all local minima of the function given by the array
    a."""
    # 2012-02-29
    # based on:
    # http://stackoverflow.com/questions/4624970/finding-local-maxima-
    # minima-with-numpy-in-a-1d-numpy-array
    a = numpy.array(a)
    return numpy.flatnonzero(numpy.r_[include_boundaries, a[1:] < a[:-1]] &
                             numpy.r_[a[:-1] < a[1:], include_boundaries])


cpdef get_num_or_acc(string):
    """Interprete a string as either an interger number or a float representing
    an accuracy. Return the resulting number as well as the requested accuracy,
    of which one will be False and the other will carry the respective value.
    The accuracy can be given as a floating point number (including a "."), as
    a percentage (ending with "%"), as a parts-per-million value (ending with
    "ppm") or as a parts-per-billion value (ending with "ppb"). Additionally,
    return the number of digits that would be needed to represent the targeted
    accuracy."""
    # 2012-04-24
    # based on _Gdos.get_ssize from 2012-03-02 - 2012-04-16

    # force string input
    string = str(string)

    # initialize the variables with the default values
    num = False
    acc = False
    digits = 2

    # test cases
    if string[-1] == '%':
        percent = float(string[:-1])
        if percent < 0 or percent > 100:
            raise ValueError('bad percentage: %s. ' % string +
                             'Must be from interval [0, 100]')
        acc = percent/100
    elif string.endswith('ppm'):  # parts per million
        ppm = float(string[:-3])
        if ppm < 0 or ppm > 1000000:
            raise ValueError('bad parts-per-million value: %s. ' % string +
                             'Must be from interval [0, 1000000]')
        acc = ppm/1000000
        digits = 5
    elif string.endswith('ppb'):  # parts per billion
        ppb = float(string[:-3])
        if ppb < 0 or ppb > 1000000000:
            raise ValueError('bad parts-per-billion value: %s. ' % string +
                             'Must be from interval [0, 1000000000]')
        acc = ppb/1000000000
        digits = 8
    elif '.' in string:
        acc = float(string)
        if acc < 0 or acc > 1:
            raise ValueError('bad accuracy value: %s. ' % string +
                             'Must be from interval [0, 1]')
    else:
        num = int(string)
        digits = 0

    # return results
    return num, acc, digits


def get_ratio(string):
    """Interprete a string as a float representing a certain ratio. The ratio
    can be given as a floating point number (including a "."), as a percentage
    (ending with "%"), as a parts-per-million value (ending with "ppm") or as a
    parts-per-billion value (ending with "ppb"). If mathematical operators are
    found ("*", "/", "+", "-"), evaluate the expression."""
    # 2012-05-03 - 2012-05-08

    # force string input
    string = str(string)

    # evaluate arithmetical expression
    if "*" in string or "/" in string or "+" in string or "-" in string:
        string = str(eval(string))

    # test cases
    if string[-1] == '%':
        percent = float(string[:-1])
        if percent < 0 or percent > 100:
            raise ValueError('bad percentage: %s. ' % string +
                             'Must be from interval [0, 100]')
        return percent/100
    elif string.endswith('ppm'):  # parts per million
        ppm = float(string[:-3])
        if ppm < 0 or ppm > 1000000:
            raise ValueError('bad parts-per-million value: %s. ' % string +
                             'Must be from interval [0, 1000000]')
        return ppm/1000000
    elif string.endswith('ppb'):  # parts per billion
        ppb = float(string[:-3])
        if ppb < 0 or ppb > 1000000000:
            raise ValueError('bad parts-per-billion value: %s. ' % string +
                             'Must be from interval [0, 1000000000]')
        return ppb/1000000000
    else:
        ratio = float(string)
        if ratio < 0 or ratio > 1:
            raise ValueError('bad float ratio: %s. ' % string +
                             'Must be from interval [0, 1]')
        return ratio


def get_num_from_ratio(value, total=1, roundfunc=round):
    """Interprete the given value (possibly string) as a certain ratio. Treat
    the ratio as being a certain part of the given number "total" and return
    the resulting number (ratio*total, rounded).

    The ratio can be given as a floating point number (including a "."), as a
    percentage (ending with "%"), as a parts-per-million value (ending with
    "ppm") or as a parts-per-billion value (ending with "ppb"). If mathematical
    operators are found ("*", "/", "+", "-"), the expression is evaluated.

    If the given value is not a string, the following rules apply:
    Integers are treated as the requested number itself, floats as ratios
    from the total number.

    Always, an integer is returned. Note that also negative numbers may be
    returned (if the given ratio is negative). This could be useful in some
    situations."""
    # 2012-07-05 - 2012-07-13

    # check for None
    if value is None:
        return None

    if isinstance(value, (int, long)):
        # given value is already the part to return
        # just check that it is not overrunning total
        if abs(value) > abs(total):
            raise ValueError('given integer number (%i) is greater ' % value +
                             'than given total number (%i)' % total)
        part = value
    elif isinstance(value, float):
        # given value is a ratio
        if value < -1 or value > 1:
            raise ValueError('bad float ratio: %s. ' % value +
                             'Must be from interval [-1, 1]')
        part = value*total
    elif isinstance(value, basestring):
        # check for None
        if value == 'None':
            return None

        if '*' not in value and '/' not in value and '+' not in value \
                and value.count('-') == 1 and value[0] == '-':
            # in the case of a minus sign as first character, no mathematical
            # evaluation is needed
            pass
        elif '*' in value or '/' in value or '+' in value or '-' in value:
            # evaluate arithmetical expression
            value = str(eval(value))

        # test cases
        if value[-1] == '%':
            percent = float(value[:-1])
            if percent < -100 or percent > 100:
                raise ValueError('bad percentage: %s. ' % value +
                                 'Must be from interval [-100, 100]')
            part = percent/100*total
        elif value.endswith('ppm'):  # parts per million
            ppm = float(value[:-3])
            if ppm < -1e6 or ppm > 1e6:
                raise ValueError('bad parts-per-million value: %s. ' % value +
                                 'Must be from interval [-1000000, 1000000]')
            part = ppm/1e6*total
        elif value.endswith('ppb'):  # parts per billion
            ppb = float(value[:-3])
            if ppb < -1e9 or ppb > 1e9:
                raise ValueError('bad parts-per-billion value: %s. ' % value +
                                 'Must be from interval ' +
                                 '[-1000000000, 1000000000]')
            part = ppb/1e9*total
        else:
            ratio = float(value)
            if ratio < -1 or ratio > 1:
                raise ValueError('bad float ratio: %s. ' % value +
                                 'Must be from interval [-1, 1]')
            part = ratio*total

    # return rounded part
    return int(roundfunc(part))


def hasattrs(object, names):
    """Return whether the object has all attributes with the given names."""
    __created__ = '2012-07-13'
    for name in names:
        if not hasattr(object, name):
            return False
    return True
