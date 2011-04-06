import pickle
import time
import operator
import numpy as np

def fill_matrix_diagonal(matrix, value):
    """Implementation of fill_diagonal from numpy"""
    matrix.flat[::matrix.shape[1]+1] = value
    return matrix

def flatten(list):
    return [item for sublist in list for item in sublist]

def _sorted_centralities(cents):
    """Sort a centralities dictionary"""
    return sorted(cents.iteritems(), key = operator.itemgetter(1), reverse = True)

def load_words(path):
    """ Return words from a file path"""
    with open(path, 'r') as f:
        words = f.read().split()
    return words

def timed(f):
    """Decorator for measuring time used in functions"""
    def wrapper(*args):
        tic = time.time()
        result =  f(*args)
        toc = time.time()
        print '  -- time used in', f.__name__,':', (toc-tic)
        return result
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    wrapper.__dict__.update(f.__dict__)
    return wrapper

def interrogate(item):
    """Print useful information about item."""
    if hasattr(item, '__name__'):
        print "NAME:    ", item.__name__
    if hasattr(item, '__class__'):
        print "CLASS:   ", item.__class__.__name__
    print "ID:      ", id(item)
    print "TYPE:    ", type(item)
    print "VALUE:   ", repr(item)
    print "CALLABLE:",
    if callable(item):
        print "Yes"
    else:
        print "No"
    if hasattr(item, '__doc__'):
        doc = getattr(item, '__doc__')
        if not doc: return
        doc = doc.strip()   # Remove leading/trailing whitespace.
        firstline = doc.split('\n')[0]
        print "DOC:     ", firstline

def to_latex_table(data, write_file=True, prt=True, start=None, end=None):
    if write_file: f = open('output/experiment', 'a')
    if start:
        if prt:
            print start
        if write_file:
            f.write(start+'\n')
    for metric in data['forward'].keys():
        line = '        '+metric+' & '+'%1.3f' % data['forward'][metric]+' & '+'%1.3f' % data['undirected'][metric]+' & '+'%1.3f' % data['backward'][metric]+' \\\\'
        if prt:
            print line
        if write_file:
            f.write(line+'\n')
    if end:
        if prt:
            print end
        if write_file:
            f.write(end+'\n')

def test_unique(path='../data/reuters1000/'):
    """Check if there are any documents with multiple categories."""
    d = {}
    for root, cats, docs in os.walk(path):
        category = root.split(os.sep)[-1]
        if not category: continue
        d[category] = set(docs)
    for cat in d.keys():
        print
        print cat
        for other in d.keys():
            if cat!=other:
                print cat+' vs '+other
                print list(d[cat].intersection(other))
