import data
import graph
import freq_representation
import graph_representation
import classify
import evaluation
import pprint as pp
import plotter
import numpy
import networkx as nx
import scipy.spatial.distance

numpy.set_printoptions(linewidth = 1000, precision = 3)

def test():
    path = '../data/air/problem_descriptions_preprocessed'
    texts, labels = data.read_files(path)
    lengths = []
    for i, text in enumerate(texts):
        print str(i)+"/"+str(len(texts))
        g = graph_representation.construct_higher_order_cooccurrence_network(text, 3)

def test_vocabulary_size():
    path = '../data/air/problem_descriptions_preprocessed'
    texts, labels = data.read_files(path)
    lengths = []
    for text in texts:
        text = text.split(' ')
        l = len(list(set(text)))
        lengths.append(l)
        print '   ',l
    lengths = numpy.array(lengths)
    print 'avg', lengths.mean()
    print 'max', lengths.max()
    print 'min', lengths.min()

if __name__ == "__main__":
    test()
