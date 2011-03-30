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

def test_classification(orders=[1,2,3]):
    #~ path = '../data/air/problem_descriptions_preprocessed'
    path = '../data/tasa/TASA900_preprocessed'
    texts, labels = data.read_files(path)
    filenames = data.get_file_names(path)
    rep = []
    for i, text in enumerate(texts):
        print str(i)+"/"+str(len(texts))
        g = graph_representation.construct_cooccurrence_network(text, orders=orders, doc_id='output/higher_order/tasa/'+labels[i]+'/'+filenames[i])
        d = graph_representation.graph_to_dict(g, graph.GraphMetrics.WEIGHTED_DEGREE)
        rep.append(d)
    rep = graph_representation.dicts_to_vectors(rep)
    score = evaluation.evaluate_classification(rep, labels)
    print score
    return score

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
    test_classification()
