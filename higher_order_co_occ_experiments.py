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

def test_retrieval(orders=[1,2,3]):
    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_preprocessed'
    description_texts, labels = data.read_files(descriptions_path)
    filenames = data.get_file_names(descriptions_path)

    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)

    print '> Creating representations..'
    rep = []
    for i, text in enumerate(description_texts):
        print '    '+str(i)+"/"+str(len(description_texts))
        g = graph_representation.construct_cooccurrence_network(text, orders=orders, doc_id='output/higher_order/air/'+labels[i]+'/'+filenames[i])
        d = graph_representation.graph_to_dict(g, graph.GraphMetrics.WEIGHTED_DEGREE)
        rep.append(d)
    rep = graph_representation.dicts_to_vectors(rep)

    print '> Evaluating..'
    score = evaluation.evaluate_retrieval(rep, solution_vectors)
    print 'orders:', orders
    print 'score:', score
    fname = 'output/higher_order/results/retr'
    with open(fname, 'a') as f:
        s = reduce(lambda x,y:str(x)+str(y), orders)
        f.write(str(s)+' '+str(score)+'\n')
    return score

def test_classification(orders=[1,2,3]):
    print '> Reading cases..'
    path = '../data/tasa/TASA900_preprocessed'
    texts, labels = data.read_files(path)
    filenames = data.get_file_names(path)
    print '> Creating representations..'
    rep = []
    for i, text in enumerate(texts):
        print '    '+str(i)+"/"+str(len(texts))
        g = graph_representation.construct_cooccurrence_network(text, orders=orders, doc_id='output/higher_order/tasa/'+labels[i]+'/'+filenames[i])
        d = graph_representation.graph_to_dict(g, graph.GraphMetrics.WEIGHTED_DEGREE)
        rep.append(d)
    rep = graph_representation.dicts_to_vectors(rep)
    print '> Evaluating..'
    score = evaluation.evaluate_classification(rep, labels)
    print 'orders:', orders
    print 'score:', score
    fname = 'output/higher_order/results/class'
    with open(fname, 'a') as f:
        s = reduce(lambda x,y:str(x)+str(y), orders)
        f.write(str(s)+' '+str(score)+'\n')
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
    combinations = [[1],
                    [2],
                    [3],
                    [1,2],
                    [1,3],
                    [2,3],
                    [1,2,3]]
    for c in combinations:
        test_classification(c)
        #~ test_retrieval(c)
