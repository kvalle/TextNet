"""Toolbox module for working with networkx graphs.

Module contains functions for calculating graph centrality, visualizing
graphs and finding various network properties, in addition to various
other useful functions.

Graph centralities are accessed using the :func:`centralities` function, which
takes as arguments a graph and the metric to use as a constant of the
GraphMetrics class.

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy
from scipy import sparse

import plotter

def normalize(A):
    """Normalize a numpy nd-array"""
    A = numpy.array(A, float)
    A -= A.min()
    return A / A.max()

def add_edges_from_matrix(graph, matrix, nodes, rel_weight=1.0):
    """Add edges to *graph* based on adjacency *matrix*.

    The *nodes* list corresponds to each row/column in *matrix*.
    The *rel_weight* scales the edge weights."""
    if sparse.isspmatrix(matrix):
        matrix = matrix.todense()
    for i, node in enumerate(nodes):
        for j, other in enumerate(nodes):
            if matrix[i,j]>0:
                w = 0.0
                if graph.has_edge(node, other):
                    w = graph.get_edge_data(node,other)['weight']
                graph.add_edge(node, other, weight=w+matrix[i,j]*rel_weight)

def equal(g1, g2):
    """Check if two graphs are identical.

    The graphs are considered equal if they contain the same set of nodes
    and edges. Edge weights are not considered.
    """
    if sorted(g1.edges()) != sorted(g2.edges()):
        return False
    if sorted(g1.nodes()) != sorted(g2.nodes()):
        return False
    return True

def reduce_edge_set(graph, remove_label):
    """Return new graph with some edges removed.

    Those edges that have the *remove_label*, and no other labels, are removed.
    """
    new_graph = nx.DiGraph()
    for u, v, data in graph.edges_iter(data=True):
        if remove_label not in data['label'] or len(data['label'])>1:
            new_graph.add_edge(u,v,data)
    return new_graph

def node_set(graphs):
    """Return list of unique nodes from list of graphs"""
    nodes = set()
    for g in graphs:
        nodes.update(g.nodes())
    return list(nodes)

def get_hubs(graph, n=10):
    """Return the *n* most important hubs from the graph"""
    from operator import itemgetter
    degrees = nx.degree_centrality(graph)
    degrees = sorted(degrees.iteritems(), key=itemgetter(1))
    return degrees[:n]

def network_properties(graph, plot_distribution=False, verbose=False):
    """Returns information about the graph."""
    props = {}
    num_nodes = graph.number_of_nodes()
    props['# nodes'] = num_nodes
    if verbose:
        print '# nodes:', props['# nodes']

    props['# edges'] = len(graph.edges())
    if verbose:
        print '# edges:', props['# edges']

    props['# selfloops'] = graph.number_of_selfloops()
    if verbose:
        print '# selfloops:', props['# selfloops']

    props['mean degree'] = float(sum(graph.degree().values())) / num_nodes
    if verbose:
        print 'mean degree:', props['mean degree']

    cc = nx.clustering(graph.to_undirected()).values()
    props['average clustering coefficient (undirected)'] = sum(cc) / num_nodes
    if verbose:
        print 'average clustering coefficient (undirected):', props['average clustering coefficient (undirected)']

    props['# connected components'] = nx.number_connected_components(graph.to_undirected())
    if verbose:
        print '# connected components:', props['# connected components']

    props['connected (if undirected)?'] = nx.is_connected(graph.to_undirected())
    if verbose:
        print 'connected (if undirected)?:', props['connected (if undirected)?']

    props['characteristic path length'] = nx.average_shortest_path_length(graph)
    if verbose:
        print 'characteristic path length:', props['characteristic path length']

    props['characteristic path length (undirected)'] = nx.average_shortest_path_length(graph.to_undirected())
    if verbose:
        print 'characteristic path length (undirected):', props['characteristic path length (undirected)']

    if plot_distribution:
        plotter.plot_degree_distribution(graph.to_undirected())

    return props

######
##
##  Centrality measures
##
######

def invert_edge_weights(G):
    """Returns a graph with all edge weights inverted"""
    for u,v in G.edges():
        G[u][v]['weight'] = 1.0 / G[u][v]['weight']
    return G

def weighted_degree(G, normalize=True):
    """Weighted degree centralities.

    Counts both incomming and outgoing links.
    This is the same as the sum of the weighted in-degree and weighted out-degree.
    Assumes digraph.
    """
    D = {}
    I = weighted_in_degree(G, normalize)
    O = weighted_out_degree(G, normalize)
    for node in G.nodes():
        D[node] = I[node] + O[node]
    return D

def weighted_in_degree(G, normalize=True):
    """Weighted in-degree centralities"""
    D = {}
    nodes = G.nodes()
    for node1 in nodes:
        D[node1] = sum([G[node2][node1]['weight'] for node2 in G.predecessors_iter(node1)])
        if normalize:
            D[node1] = float(D[node1])/(len(nodes)-1)
    return D

def weighted_out_degree(G, normalize=True):
    """Weighted out-degree centralities"""
    D = {}
    nodes = G.nodes()
    for node1 in nodes:
        D[node1] = sum([G[node1][node2]['weight'] for node2 in G.successors_iter(node1)])
        if normalize:
            D[node1] = float(D[node1])/(len(nodes)-1)
    return D

def hits_hubs(G):
    """The HITS hubs centralities"""
    return nx.hits_numpy(G)[0]

def hits_authorities(G):
    """The HITS authorities centralities"""
    return nx.hits_numpy(G)[1]

def clustering_degree(G):
    """Clustering degree ''centrality''

    Measure of centrality based on the clustering coefficient of a node
    multiplied with its weighted degree centrality.
    """
    DC = {}
    D = weighted_degree(G)
    C = nx.clustering(G.to_undirected())
    for node in G.nodes():
        DC[node] = D[node]*C[node]
    return DC

def closeness(G):
    """Closeness centrality"""
    G = invert_edge_weights(G)
    return nx.closeness_centrality(G)

def weighted_closeness(G):
    """Weighted version of the closeness centrality"""
    G = invert_edge_weights(G)
    return nx.closeness_centrality(G, weighted_edges=True)

def betweenness(G):
    """Betweenness centrality"""
    G = invert_edge_weights(G)
    return nx.betweenness_centrality(G)

def weighted_betweenness(G):
    """Weighted version of the betweenness centrality"""
    G = invert_edge_weights(G)
    return nx.betweenness_centrality(G, weighted_edges=True)

def load(G):
    """Load centrality"""
    G = invert_edge_weights(G)
    return nx.load_centrality(G)

def weighted_load(G):
    """Weighted version of the load centrality"""
    G = invert_edge_weights(G)
    return nx.load_centrality(G, weighted_edges=True)

def current_flow_betweenness(G):
    """Current-flow betweenness centrality"""
    G = G.to_undirected()
    G = invert_edge_weights(G)
    if nx.is_connected(G):
        return nx.current_flow_betweenness_centrality(G)
    else:
        return _aggregate_for_components(G, nx.current_flow_betweenness_centrality)

def current_flow_closeness(G):
    """Current-flow closeness centrality"""
    G = G.to_undirected()
    G = invert_edge_weights(G)
    if nx.is_connected(G):
        return nx.current_flow_closeness_centrality(G)
    else:
        return _aggregate_for_components(G, nx.current_flow_closeness_centrality)

def _aggregate_for_components(G, aggr_fun):
    aggr = {}
    for subgraph in nx.connected_component_subgraphs(G):
        if len(subgraph.nodes())==1: # only node in subgraph: centrality 0
            a = {subgraph.nodes()[0]: 0}
        else:
            a = aggr_fun(subgraph)
        aggr.update(a)
    return aggr

def pagerank(G):
    """PageRank values for nodes in graph *G*"""
    return nx.pagerank_numpy(G)

# Enumeration with available centrality measures.
class GraphMetrics:
    """Class holding constants for the different graph centrality metrics"""
    DEGREE = 'Degree'
    WEIGHTED_DEGREE = 'Degree (weighted)'
    IN_DEGREE = 'In-degree'
    WEIGHTED_IN_DEGREE = 'In-degree (weighted)'
    OUT_DEGREE = 'Out-degree'
    WEIGHTED_OUT_DEGREE = 'Out-degree (weighted)'

    CLOSENESS = 'Closeness'
    WEIGHTED_CLOSENESS = 'Closeness (weighted)'
    CURRENT_FLOW_CLOSENESS = 'Current-flow closeness'

    BETWEENNESS = 'Betweenness'
    WEIGHTED_BETWEENNESS = 'Betweenness (weighted)'
    CURRENT_FLOW_BETWEENNESS = 'Current-flow betweenness'
    LOAD = 'Load'
    WEIGHTED_LOAD = 'Load (weighted)'

    EIGENVECTOR = 'Eigenvector'
    PAGERANK =  'PageRank'
    HITS_HUBS = 'HITS (hubs)'
    HITS_AUTHORITIES = 'HITS (authorities)'
    CLUSTERING_DEGREE = 'Degree-weighted clustering coefficient'

# Dictionary with mapping between constants and functions
mapping = {
    GraphMetrics.DEGREE : nx.degree_centrality,
    GraphMetrics.WEIGHTED_DEGREE: weighted_degree,
    GraphMetrics.IN_DEGREE : nx.in_degree_centrality,
    GraphMetrics.WEIGHTED_IN_DEGREE: weighted_in_degree,
    GraphMetrics.OUT_DEGREE : nx.out_degree_centrality,
    GraphMetrics.WEIGHTED_OUT_DEGREE: weighted_out_degree,

    GraphMetrics.CLOSENESS : closeness,
    GraphMetrics.WEIGHTED_CLOSENESS : weighted_closeness,
    GraphMetrics.CURRENT_FLOW_CLOSENESS : current_flow_closeness,

    GraphMetrics.BETWEENNESS : betweenness,
    GraphMetrics.WEIGHTED_BETWEENNESS : weighted_betweenness,
    GraphMetrics.CURRENT_FLOW_BETWEENNESS : current_flow_betweenness,
    GraphMetrics.LOAD : load,
    GraphMetrics.WEIGHTED_LOAD : weighted_load,

    GraphMetrics.EIGENVECTOR : nx.eigenvector_centrality_numpy,
    GraphMetrics.PAGERANK : pagerank,
    GraphMetrics.HITS_HUBS : hits_hubs,
    GraphMetrics.HITS_AUTHORITIES : hits_authorities,
    #~ GraphMetrics.CLUSTERING_DEGREE : clustering_degree
}

# Obtains centrality vector.
def centralities(graph, method):
    """Return centralities for nodes in *graph* using the given centrality *method*"""
    return mapping[method](graph)

######
##
##  Draw / display graph
##
######

def draw(graph):
    """Draw the *graph*"""
    nx.draw(graph, pos = nx.pydot_layout(graph), node_size = 1, node_color = 'w', font_size = 8)
    plt.show()

def draw_with_centrality(G, sizeV = None, min_size = 500, max_size = 4000, default_size = 500, layout=None):
    """Visualizes a graph preserving edge length and demonstrating centralities as node sizes."""
    if sizeV is None:
        sizeV = default_size
    else:
        sizeV = normalize(sizeV) * (max_size - min_size) + min_size

    if not layout:
        #~ layout = nx.pydot_layout(G)
        #~ layout = nx.spring_layout(G)
        layout = nx.graphviz_layout(G)
        #~ layout = nx.circular_layout(G)
        #~ layout = nx.shell_layout(G)
        #~ layout = nx.spectral_layout(G)

    nx.draw(G, pos = layout, node_size = sizeV, node_color = 'w', font_size = 20)
    plt.show()

######
##
##  Demo functions
##
##  (Functions intended to test/demonstrate various functionaliyt of the module)
##
######

def demo_centralities():
    print '> Hello, Dave.'
    #~ g = nx.Graph()
    #~ g.add_nodes_from(range(1, 6))
    #~ g.add_edges_from([(1,2),(1,3),(1,4),(1,5)])
    #~ cents = centralities(g, Metrics.DEGREE)
    #~ cents = graph.centralities(g, graph.Metrics.EIGENVECTOR)
    #~ cents = graph.centralities(g, graph.Metrics.PAGERANK)
    #~ cents = graph.centralities(g, graph.Metrics.CLOSENESS)
    #~ cents = graph.centralities(g, graph.Metrics.BETWEENNESS)
    #~ print nx.info(g)
    #~ draw_with_centrality(g, cents.values())
    print "> I'm sorry, Dave. I can't do that, Dave."

def demo_graph_generators():
    g = nx.wheel_graph(6)
    #~ g = nx.cycle_graph(10)
    #~ g = nx.star_graph(4)
    #~ g = nx.balanced_tree(3,3)
    #~ g = nx.complete_graph(5)
    #~ g = nx.dorogovtsev_goltsev_mendes_graph(2)
    #~ g = nx.hypercube_graph(2)
    #~ g = nx.ladder_graph(10)
    #~ g = nx.lollipop_graph(5, 4)
    #~ g = nx.house_graph()
    #~ g = nx.house_x_graph()
    draw_with_centrality(g)

def demo_create_clustering_example():
    g = nx.wheel_graph(6)
    pos = nx.spring_layout(g)
    g.add_edge(1,3)
    #~ g.add_edge(1,4)
    g.add_edge(2,4)
    #~ g.add_edge(2,5)
    #~ g.add_edge(3,1)
    #~ g.add_edge(3,5)
    g.remove_edge(4,5)
    # remove node 0 edges
    #~ g.remove_edge(0,1)
    #~ g.remove_edge(0,2)
    #~ g.remove_edge(0,3)
    #~ g.remove_edge(0,4)
    #~ g.remove_edge(0,5)
    #~ print nx.clustering(g)
    #~ print nx.average_clustering(g)
    draw_with_centrality(g, layout=pos)

def demo_create_centrality_example():
    g = nx.wheel_graph(6)
    #~ pos = nx.spring_layout(g)
    pos = None
    g = nx.Graph()
    g.add_edge(1,3, weight=1.0)
    g.add_edge(1,4, weight=1.0)
    g.add_edge(2,4, weight=3.0)
    g.add_edge(2,5, weight=1.0)
    g.add_edge(3,1, weight=1.0)
    g.add_edge(3,5, weight=1.0)
    draw_with_centrality(g, layout=pos)

def pagerank_example():
    n = 7
    g = nx.wheel_graph(n)
    pos = nx.spring_layout(g)

    g = nx.DiGraph()
    g.add_nodes_from(range(0,n))

    g.add_edge(0,1)
    g.add_edge(0,6)
    g.add_edge(0,5)
    g.add_edge(1,2)
    g.add_edge(1,6)
    g.add_edge(2,0)
    g.add_edge(2,1)
    g.add_edge(2,3)
    g.add_edge(3,4)
    g.add_edge(4,5)
    g.add_edge(4,6)
    g.add_edge(5,0)
    g.add_edge(5,3)
    g.add_edge(5,4)

    ranks = nx.pagerank(g)
    for n in range(0,n):
        print 'node',n
        print '  rank:',ranks[n]
        print '  out edges:',g.neighbors(n)
        if g.neighbors(n):
            print '  per edge:',ranks[n]/len(g.neighbors(n))
        else:
            print '  per edge: null'

    draw_with_centrality(g, layout=pos)

def hits_example():
    n = 7
    #~ g = nx.wheel_graph(n)
    #~ pos = nx.spring_layout(g)
    g = nx.DiGraph()
    g.add_nodes_from(range(0,n))
    edge_list = [(0,1),(0,6),(0,5),(1,2),(1,6),(2,0),(2,1),(2,3),(3,4),(4,5),(4,6),(5,0),(5,3),(5,4)]
    g.add_edges_from(edge_list)
    hubs,auts = nx.hits(g)
    for n in range(0,n):
        print 'node',n
        print '  authority:',auts[n]
        print '  hubness:', hubs[n]
        print '  out:',g.successors(n)
        print '  in:',g.predecessors(n)
    #~ draw_with_centrality(g, layout=pos)

def closeness_example():
    n = 5
    g = nx.wheel_graph(n)
    pos = nx.spring_layout(g)
    g = nx.DiGraph()
    g.add_nodes_from(range(0,n))
    edge_list = [(1,0),(2,1),(0,2),(0,3),(3,4)]
    for a,b in edge_list:
        g.add_edge(b,a,weight=1.0)
    cent = closeness(g)
    for n in range(0,n):
        print 'node',n
        print '  centrality:',cent[n]
        print '  out:',g.successors(n)
        print '  in:',g.predecessors(n)
    draw_with_centrality(g, layout=pos)

def demo_network_properties():
    import pprint as pp
    g = nx.DiGraph()
    g.add_nodes_from(range(0,7))
    edge_list = [(0,1),(0,6),(0,5),(1,2),(1,6),(2,0),(2,1),(2,3),(3,4),(4,5),(4,6),(5,0),(5,3),(5,4)]
    g.add_edges_from(edge_list)
    pp.pprint(network_properties(g, True))

def test_reduce_edge_set():
    g = nx.DiGraph()
    g.add_edge(1,2,label=['foo','bar'])
    g.add_edge(1,3,label=['baz'])
    g.add_edge(1,4,label=['foo'])
    g.add_edge(1,5,label=['bar'])
    f = reduce_edge_set(g, 'bar')
    for a,b,data in f.edges_iter(data=True):
        print a,b,data

if __name__ == "__main__":
    #~ demo_centralities()
    #~ demo_graph_generators()
    #~ demo_create_clustering_example()
    #~ demo_create_centrality_example()
    #~ hits_example()
    #~ demo_network_properties()
    #~ closeness_example()
    test_reduce_edge_set()
    pass
