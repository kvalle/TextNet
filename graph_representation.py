"""Construct graph representations from text.

The module contains functions from creating networks based on text documents,
and for converting the networks into feature-vectors.
Feature vectors are created based on node centrality in the text networks.

The following text representations are supported:

:random:
  Will create a network with all distinct terms in the provided document
  as nodes. Edges are created at random between the nodes, based on provided
  probabilities.

:co-occurrence:
  Distinct terms in the document are used as nodes. Edges are created
  between any terms that occurs closely together in the text.

:dependency:
  Words as nodes. Edges represent dependencies extracted from the text
  using the stanford dependency parser (see the 'stanford_parser' module).

The module makes heavy use of the :mod:`graph` module.

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

from random import random as rand
import numpy as np
import networkx as nx
import pickle
from scipy import sparse
import os.path
import math

import graph
import preprocess
import util
import data

######
##
##  Create graph representations
##
######

def create_graphs(documents, graph_type='co-occurrence', verbose=False):
    """Crate text networks of given type, using their default paramters"""
    graphs = []
    for i, text in enumerate(documents):
        if verbose and i%100==0:
            print '   ',i,'of',len(documents)
        if graph_type=='co-occurrence':
            g = construct_cooccurrence_network(text)
        elif graph_type=='dependency':
            g = construct_dependency_network(text)
        elif graph_type=='random':
            g = construct_random_network(text)
        else:
            raise Exception("unreccognized graph type: "+graph_type)
        graphs.append(g)
    return graphs

def _cooccurrence_preprocessing(doc, context, already_preprocessed):
    """Preprocess document as needed for co-occurrence network creation"""
    if context=='window':
        if already_preprocessed:
            doc = doc.split(' ')
        else:
            doc = preprocess.preprocess_text(doc)
    elif context=='sentence':
        doc = preprocess.tokenize_sentences(doc)
        for i, sentence in enumerate(doc):
            sentence = preprocess.preprocess_text(sentence)
            doc[i] = sentence
    return doc

def _window_cooccurrence_matrix(doc, direction='undirected', window_size=2, verbose=False):
    """Create co-occurrence matrix for *doc* using context window"""
    term_list = np.array(list(set(doc)))
    A = sparse.lil_matrix( (len(term_list), len(term_list)) )
    for i, word in enumerate(doc):
        if verbose:
            if i < 10 or i%1000==0: print '    word ',str(i)+' of '+str(len(doc))
        context = doc[i+1:i+1+window_size]
        for context_word in context:
            if word == context_word: continue
            x = np.where(term_list==word)[0][0]
            y = np.where(term_list==context_word)[0][0]
            if direction == 'forward' or direction == 'undirected':
                A[x,y] += 1
            if direction == 'backward' or direction == 'undirected':
                A[y,x] += 1
    return A, term_list

def _sentence_cooccurrence_matrix(doc, direction='undirected', verbose=False):
    """Create co-occurrence matrix for *doc* using sentence contexts"""
    term_list = np.array(list(set(util.flatten(doc))))
    A = sparse.lil_matrix( (len(term_list), len(term_list)) )
    for i, sentence in enumerate(doc):
        if verbose:
            if i < 10 or i%1000==0: print '    sentence ',str(i)+' of '+str(len(doc))
        for w, word in enumerate(sentence):
            for c, context_word in enumerate(sentence):
                if word == context_word: continue
                if direction=='forward' and w > c: continue
                if direction=='backward' and w < c: continue
                x = np.where(term_list==word)[0][0]
                y = np.where(term_list==context_word)[0][0]
                A[x,y] += 1
    return A, term_list

def construct_cooccurrence_network(doc, window_size=2, direction='undirected', context='sentence', already_preprocessed=False, orders=[], order_weights=[1.0,1.0,1.0],doc_id=None,verbose=False):
    """Construct co-occurrence network from text.

    *direction* must be 'forward', 'backward' or 'undirected', while  *context*
    can be 'window' or 'sentence'.

    If *context* is 'window', *already_preprocessed* indicate whether *doc*
    already have been processed. Sentence contexts require unpreocessed *doc*s.

    Any value for *window_size* is ignored if *context* is 'sentence'.

    A DiGraph is created regardless of direction parameter, but with 'undirected',
    edges are created in both directions.
    """
    doc = _cooccurrence_preprocessing(doc, context, already_preprocessed)
    if context is 'sentence':
        matrix, term_list = _sentence_cooccurrence_matrix(doc, direction, verbose)
    elif context is 'window':
        matrix, term_list = _window_cooccurrence_matrix(doc, direction, window_size, verbose)
    g = nx.DiGraph()
    g.add_nodes_from(term_list)
    if len(orders)==0:
        graph.add_edges_from_matrix(g, matrix, term_list)
    else:
        if doc_id is not None and os.path.exists(doc_id):
            first, second, third = data.pickle_from_file(doc_id)
        else:
            first, second, third = _higher_order_matrix(matrix.todense())
            if doc_id is not None:
                data.pickle_to_file((first,second,third), doc_id)
    if 1 in orders:
        graph.add_edges_from_matrix(g, first, term_list, rel_weight=order_weights[0])
    if 2 in orders:
        graph.add_edges_from_matrix(g, second, term_list, rel_weight=order_weights[1])
    if 3 in orders:
        graph.add_edges_from_matrix(g, third, term_list, rel_weight=order_weights[2])
    return g

def _higher_order_matrix(matrix, do_clip=False):
    """Construct higher order matrix from first order *matrix*.

    Values of *matrix* are clipped to range [0,1] depending on *do_clip*"""
    dim = matrix.shape[0]
    if do_clip: matrix = np.clip(matrix, 0, 1)
    first_order = util.fill_matrix_diagonal(matrix, 0.0)
    second_order = first_order**2
    third_order = first_order**3
    col_sums = np.sum(first_order, axis=0)
    discount_matrix = np.matrix(np.zeros((dim,dim), np.int8))
    for i in range(dim):
        for j in range(dim):
            d = col_sums[0,i] + col_sums[0,j] - 1
            if i!=j:
                discount_matrix[i,j] = d
    third_order = third_order - np.multiply(discount_matrix,first_order)
    return first_order, second_order, third_order

def construct_random_network(doc, p=0.2):
    """Construct random network for use as baseline.

    Create a random network based on *doc*, with words used for nodes.
    Edges are created between any given pair of nodes (a,b)  with probability *p*.

    All edges will have weight = 1.0
    """
    doc = preprocess.preprocess_text(doc)
    words = list(set(doc)) # list of unique words

    # create graph
    graph = nx.DiGraph()
    graph.add_nodes_from(words)

    # add edges
    for word_a in graph.nodes():
        for word_b in graph.nodes():
            if word_a != word_b and rand() < p:
                _update_edge_weight(graph, word_a, word_b)

    return graph

def construct_dependency_network(doc, weighted=False, direction='undirected',remove_stop_words=False, exclude=['agent', 'advcl','parataxis'],verbose=False, unpickle=True):
    """Construct a dependency network from *doc*.

    Creates a network form *doc* with distinct word used for nodes, and
    all dependency types defined by the stanford parser, except those listed
    in *exclude* used as edges.

    *direction* must be 'undirected', 'forward' or 'backward.
    Forward direction means head-dependent, while backward gives dependent-head relations.
    """
    graph = nx.DiGraph()
    if unpickle:
        deps = pickle.loads(doc)
    else:
        deps = doc
    for dep_type, dep in deps.iteritems():
        if verbose: print '    dep:',dep_type
        if dep_type in exclude:
            continue
        for tup in dep:
            g = tup[0][0]
            d = tup[1][0]
            g = preprocess.preprocess_token(g, do_stop_word_removal=remove_stop_words)
            d = preprocess.preprocess_token(d, do_stop_word_removal=remove_stop_words)
            if g is not None and d is not None:
                if direction=='forward':
                    _update_edge_weight(graph, g, d,labels=[dep_type],inc_weight=weighted)
                else:
                    _update_edge_weight(graph, d, g,labels=[dep_type],inc_weight=weighted)
    if direction=='undirected':
        graph = graph.to_undirected().to_directed()
    return graph

def _update_edge_weight(graph, node1, node2,labels=[],inc_weight=True):
    """Update or create weighed edge between two nodes"""
    if graph.has_edge(node1, node2):
        graph[node1][node2]['label'] += labels
        if inc_weight:
            graph[node1][node2]['weight'] += 1.0
    else:
        graph.add_edge(node1, node2, weight=1.0,label=labels)

def similarity_matrix_to_graph(distM):
    """Converts similarity matrix to weighted graph.

    :Author: Gleb Sizov <sizov@idi.ntnu.no>
    """
    lenM = distM
    weightM = 1 / lenM
    G = nx.Graph()
    for node1 in range(len(distM)):
        G.add_node(node1)
        for node2 in range(node1 + 1, len(distM)):
            if distM[node1, node2] > 0:
                G.add_edge(node1, node2, len=lenM[node1, node2], weight=weightM[node1, node2])
    return G

######
##
##  Create vector representations
##
######

def graphs_to_vectors(graphs, metric, verbose=False):
    """ Create centrality based feature-vectors from graph representations

    Takes a list of graphs and returns a numpy nd-matix of feature vectors,
    based on the provides *metric*.
    """
    all_tokens = graph.node_set(graphs)
    features = np.zeros((len(all_tokens), len(graphs)))
    for i, g in enumerate(graphs):
        if verbose and i%50==0: print str(i)+'/'+str(len(graphs))
        features[:,i] = graph_to_vector(g, metric, all_tokens)
    return features

def graph_to_vector(g, metric, all_tokens):
    """Create feature vector from a single graph.

    The list of *all_tokens* is used as basis for the feature vector, and
    value for each word in graph *g* according to *metric* is calculated.
    """
    cents = graph.centralities(g, metric)
    vector = [cents.get(token, 0.0) for token in all_tokens]
    return vector

def graph_to_dict(g, metric, icc=None):
    """Return node values as dictionary

    If `icc` is provided, values are TC-ICC, otherwise TC is calculated.
    """
    centralities = graph.centralities(g, metric)
    if icc:
        for term in centralities:
            centralities[term] = centralities[term] * icc[term]
    return centralities

def dicts_to_vectors(dicts, explicit_keys=None):
    """Convert a list of dictionaries to feature-vectors"""
    if not explicit_keys:
        node_set = set()
        for d in dicts:
            for node in d.keys():
                node_set.add(node)
        all_tokens = list(node_set)
    else:
        all_tokens = explicit_keys
    features = np.zeros((len(all_tokens), len(dicts)))
    for i, d in enumerate(dicts):
        if i%100==0: print '    vector',str(i)+'/'+str(len(dicts))
        features[:,i] = [d.get(token, 0.0) for token in all_tokens]
    return features

def calculate_icc_dict(centralities):
    icc = {}
    for term in centralities:
        icc[term] = 1.0/(1.0 + centralities[term])
    return icc

######
##
##  Term weighting metrics
##
######

from graph import GraphMetrics

def get_metrics(weighted=None, exclude_flow=False):
    """Return list of graph node evaluation metrics.

    If *weighted* is not specified, or `None`, all metrics are returned.
    Otherwise metrics suited for (un)*weighted* networks are returned."""
    metrics = None
    if weighted is None:
        metrics = graph.mapping.keys()
    elif weighted:
        metrics = [GraphMetrics.WEIGHTED_DEGREE, GraphMetrics.WEIGHTED_IN_DEGREE, GraphMetrics.WEIGHTED_OUT_DEGREE,
        GraphMetrics.WEIGHTED_CLOSENESS, GraphMetrics.CURRENT_FLOW_CLOSENESS,
        GraphMetrics.WEIGHTED_BETWEENNESS, GraphMetrics.CURRENT_FLOW_BETWEENNESS, GraphMetrics.WEIGHTED_LOAD,
        GraphMetrics.EIGENVECTOR, GraphMetrics.PAGERANK, GraphMetrics.HITS_HUBS, GraphMetrics.HITS_AUTHORITIES]
    else:
        metrics = [GraphMetrics.DEGREE, GraphMetrics.IN_DEGREE, GraphMetrics.OUT_DEGREE,
        GraphMetrics.CLOSENESS, GraphMetrics.CURRENT_FLOW_CLOSENESS,
        GraphMetrics.BETWEENNESS, GraphMetrics.CURRENT_FLOW_BETWEENNESS, GraphMetrics.LOAD,
        GraphMetrics.EIGENVECTOR, GraphMetrics.PAGERANK, GraphMetrics.HITS_HUBS, GraphMetrics.HITS_AUTHORITIES]
    if exclude_flow:
        metrics.remove(GraphMetrics.CURRENT_FLOW_BETWEENNESS)
        metrics.remove(GraphMetrics.CURRENT_FLOW_CLOSENESS)
    return metrics

######

def test_dependency_graph():
    (docs, labels) = data.read_files('../data/tasa/TASA900_dependencies')
    graphs = []
    for i, text in enumerate(docs):
        print i
        graphs.append(construct_dependency_network(text))

    g = graphs[0]
    print g.nodes()
    print g.edges()

    print '#graphs:', len(graphs)

    pos = nx.spring_layout(g)
    graph.draw_with_centrality(g, layout=pos)

def test_graph_to_dict():
    import pprint as pp
    g = nx.DiGraph()
    g.add_nodes_from(range(0,7))
    edge_list = [(0,1),(0,6),(0,5),(1,2),(1,6),(2,0),(2,1),(2,3),(3,4),(4,5),(4,6),(5,0),(5,3),(5,4)]
    g.add_edges_from(edge_list)
    pp.pprint(graph_to_dict(g,'PageRank'))

##
#  TESTS
##

def test_co_occurrences():
    doc1 = data.read_file('../data/tasa/TASATest/Science/Agatha09.07.03.txt')
    doc2 = data.read_file('../data/tasa/TASATest_preprocessed/Science/Agatha09.07.03.txt')
    g0 = construct_cooccurrence_network(doc1, context='window', already_preprocessed=False)
    g1 = construct_cooccurrence_network(doc2, context='window', already_preprocessed=True)
    g2 = construct_cooccurrence_network(doc1, context='sentence', already_preprocessed=False)
    graphs = data.pickle_from_file('output/testdata/co-occurrence-graphs.pkl')
    assert(graph.equal(g0,graphs[0]))
    assert(graph.equal(g1,graphs[1]))
    assert(graph.equal(g2,graphs[2]))

    doc = data.read_file('output/testdata/higher.order.testdoc.preprocessed.txt')
    g1 = construct_cooccurrence_network(doc, already_preprocessed=True, window_size=1, orders=[1])
    g12 = construct_cooccurrence_network(doc, already_preprocessed=True, window_size=1, orders=[1,2])
    g123 = construct_cooccurrence_network(doc, already_preprocessed=True, window_size=1, orders=[1,2,3])
    g13 = construct_cooccurrence_network(doc, already_preprocessed=True, window_size=1, orders=[1,3])
    assert(('foo','bar') in g1.edges())
    assert(('foo','baz') not in g1.edges())
    assert(('foo','cake') not in g1.edges())
    assert(('foo','bar') in g12.edges())
    assert(('foo','baz') in g12.edges())
    assert(('foo','cake') not in g12.edges())
    assert(('foo','bar') in g123.edges())
    assert(('foo','baz') in g123.edges())
    assert(('foo','cake') in g123.edges())
    assert(('foo','baz') not in g13.edges())
    print 'ok'

def test_higher_order():
    matrix = np.matrix([[3,1,1,2],
                        [1,2,0,2],
                        [2,0,2,1],
                        [2,2,1,3]])
    first, second, third = _higher_order_matrix(matrix)
    ref_1 = np.matrix([[0,1,1,1],
                       [1,0,0,1],
                       [1,0,0,1],
                       [1,1,1,0]])
    assert(np.equal(first,ref_1).all())
    ref_2 = np.matrix([[3,1,1,2],
                       [1,2,2,1],
                       [1,2,2,1],
                       [2,1,1,3]])
    assert(np.equal(second,ref_2).all())
    ref_3 = np.matrix([[4,1,1,0],
                       [1,2,2,1],
                       [1,2,2,1],
                       [0,1,1,4]])
    assert(np.equal(third,ref_3).all())
    print 'ok'

def run_tests():
    test_co_occurrences()
    test_higher_order()

if __name__=="__main__":
    run_tests()
