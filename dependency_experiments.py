"""
Experiments with various aspects of the dependency network representation.
"""
import pprint as pp
import numpy
import scipy.spatial.distance
import pickle
import networkx as nx
import nltk

import data
import graph
import freq_representation
import graph_representation
import classify
import evaluation
import plotter
import stanford_parser
import preprocess

numpy.set_printoptions(linewidth = 1000, precision = 3)

def edge_direction_evaluation(direction):
    """
    Evaluate impact of using different edge directions on dependency networks.

    Values for *direction*: ``forward``, ``backward``, and ``undirected``.
    """
    results = {'_edge-direction':direction}

    print '------ CLASSIFICATION EVALUATION --------'

    print '> Reading cases..'
    descriptions_path = '../data/tasa/TASA900_dependencies'
    texts, labels = data.read_files(descriptions_path)

    print '> Creating representations..'
    rep = []
    for i, text in enumerate(texts):
        if i%100==0: print '   ',str(i)+'/'+str(len(texts))
        g = graph_representation.construct_dependency_network(text, direction=direction)
        metric  = graph.GraphMetrics.CLOSENESS
        d = graph_representation.graph_to_dict(g, metric)
        rep.append(d)
        g = None # just to make sure..
    rep = graph_representation.dicts_to_vectors(rep)

    print '> Evaluating..'
    score = evaluation.evaluate_classification(rep, labels)
    print '   score:', score
    results['classification'] = score

    print '------ RETRIEVAL EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_dependencies'
    description_texts, labels = data.read_files(descriptions_path)
    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)

    print '> Creating representations..'
    rep = []
    for i, text in enumerate(description_texts):
        if i%100==0: print '   ',str(i)+'/'+str(len(description_texts))
        g = graph_representation.construct_dependency_network(text, direction=direction)
        metric = graph.GraphMetrics.EIGENVECTOR
        d = graph_representation.graph_to_dict(g, metric)
        rep.append(d)
        g = None # just to make sure..
    rep = graph_representation.dicts_to_vectors(rep)

    print '> Evaluating..'
    score = evaluation.evaluate_retrieval(rep, solution_vectors)
    print '   score:', score
    results['retrieval'] = score

    data.pickle_to_file(results, 'output/dependencies/stop_words_retr_'+direction)

    pp.pprint(results)
    return results

def stop_word_evaluation(rem_stop_words):
    """
    Experiment for determining what effect removing stop words have on
    dependency networks.
    """
    results = {'_removing stop-words':rem_stop_words}

    print '------ CLASSIFICATION EVALUATION --------'

    print '> Reading cases..'
    descriptions_path = '../data/tasa/TASA900_dependencies'
    texts, labels = data.read_files(descriptions_path)

    print '> Creating representations..'
    rep = []
    total_nodes = 0
    for i, text in enumerate(texts):
        if i%100==0: print '   ',str(i)+'/'+str(len(texts))
        g = graph_representation.construct_dependency_network(text, remove_stop_words=rem_stop_words)
        total_nodes += len(g.nodes())
        metric  = graph.GraphMetrics.CLOSENESS
        d = graph_representation.graph_to_dict(g, metric)
        rep.append(d)
        g = None # just to make sure..
    rep = graph_representation.dicts_to_vectors(rep)

    print '> Evaluating..'
    score = evaluation.evaluate_classification(rep, labels)
    print '   score:', score
    print '(the networks had a total of',total_nodes,'nodes)'
    results['classification'] = score

    print '------ RETRIEVAL EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_dependencies'
    description_texts, labels = data.read_files(descriptions_path)
    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)

    print '> Creating representations..'
    rep = []
    total_nodes = 0
    for i, text in enumerate(description_texts):
        if i%100==0: print '   ',str(i)+'/'+str(len(description_texts))
        g = graph_representation.construct_dependency_network(text, remove_stop_words=rem_stop_words)
        total_nodes += len(g.nodes())
        metric = graph.GraphMetrics.EIGENVECTOR
        d = graph_representation.graph_to_dict(g, metric)
        rep.append(d)
        g = None # just to make sure..
    rep = graph_representation.dicts_to_vectors(rep)

    print '> Evaluating..'
    score = evaluation.evaluate_retrieval(rep, solution_vectors)
    print '   score:', score
    print '(the networks had a total of',total_nodes,'nodes)'
    results['retrieval'] = score

    if rem_stop_words:
        postfix = '_without'
    else:
        postfix = '_with'
    data.pickle_to_file(results, 'output/dependencies/stop_words_retr'+postfix)

    pp.pprint(results)
    return results

def centrality_weights_retrieval(weighted=True):
    """
    Evaluate whether edge weights are beneficial to the depdendency
    network represenation for the retrieval task.
    """
    results = {'_is_weighted':weighted, '_evaluation':'retrieval'}
    graph_metrics = graph_representation.get_metrics(weighted)

    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_dependencies'
    description_texts, labels = data.read_files(descriptions_path)

    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)

    rep = {}
    for metric in graph_metrics:
        rep[metric] = []

    print '> Creating graph representations..'
    for i, text in enumerate(description_texts):
        if i%10==0: print '   ',str(i)+'/'+str(len(description_texts))
        g = graph_representation.construct_dependency_network(text, weighted=weighted)
        for metric in graph_metrics:
            d = graph_representation.graph_to_dict(g, metric)
            rep[metric].append(d)
        g = None # just to make sure..
        if i%100==0:
            if weighted:
                postfix = '_weighted'
            else:
                postfix = '_unweighted'
            data.pickle_to_file(rep, 'output/dependencies/exp1_retr_tmp_'+str(i)+'_'+postfix)

    print '> Creating vector representations..'
    for metric in graph_metrics:
        rep[metric] = graph_representation.dicts_to_vectors(rep[metric])

    print '> Evaluating..'
    for metric in graph_metrics:
        vectors = rep[metric]
        score = evaluation.evaluate_retrieval(vectors, solution_vectors)
        print '   ', metric, score
        results[metric] = score

    if weighted:
        postfix = '_weighted'
    else:
        postfix = '_unweighted'
    data.pickle_to_file(results, 'output/dependencies/exp1_retr'+postfix)

    pp.pprint(results)
    return results

def centrality_weights_classification(weighted=True):
    """
    Evaluate whether edge weights are beneficial to the depdendency
    network represenation for the classification task.
    """
    results = {'_is_weighted':weighted, '_evaluation':'classification'}
    graph_metrics = graph_representation.get_metrics(weighted)

    print '> Reading cases..'
    descriptions_path = '../data/tasa/TASA900_dependencies'
    texts, labels = data.read_files(descriptions_path)

    rep = {}
    for metric in graph_metrics:
        rep[metric] = []

    print '> Creating graph representations..'
    for i, text in enumerate(texts):
        if i%10==0: print '   ',str(i)+'/'+str(len(texts))
        g = graph_representation.construct_dependency_network(text, weighted=weighted)
        for metric in graph_metrics:
            d = graph_representation.graph_to_dict(g, metric)
            rep[metric].append(d)
        g = None # just to make sure..
        if i%100==0:
            if weighted:
                postfix = '_weighted'
            else:
                postfix = '_unweighted'
            data.pickle_to_file(rep, 'output/dependencies/exp1_class_tmp_'+str(i)+'_'+postfix)

    print '> Creating vector representations..'
    for metric in graph_metrics:
        rep[metric] = graph_representation.dicts_to_vectors(rep[metric])

    print '> Evaluating..'
    for metric in graph_metrics:
        vectors = rep[metric]
        score = evaluation.evaluate_classification(vectors, labels)
        print '   ', metric, score
        results[metric] = score

    if weighted:
        postfix = '_weighted'
    else:
        postfix = '_unweighted'
    data.pickle_to_file(results, 'output/dependencies/exp1_class'+postfix)

    pp.pprint(results)
    return results

def plot_exp1():
    """
    Plotting the results of the weight evaluation experiment.
    """
    colors = ['#3C54FF','#EF4C32','#27A713']

    # classification
    bar_names = ['Betweenness','Closeness','Current-flow betweenness',
                'Current-flow closeness','Degree','Eigenvector',
                'HITS authorities','HITS hubs','Load', 'PageRank']
    data = {
            'unweighted': [0.36388888888888887,0.57499999999999996,0.23333333333333334,0.56944444444444442,0.52500000000000002,0.49722222222222223,0.49722222222222223,0.49722222222222223,0.35555555555555557,0.52777777777777779],
            'weighted': [0.36944444444444446,0.57499999999999996,0.20833333333333334,0.58333333333333337,0.49444444444444446,0.45555555555555555,0.45555555555555555,0.45555555555555555,0.36666666666666664,0.51111111111111107]
        }
    #~ plotter.bar_graph(data, bar_names, x_label='classification accuracy',colors=colors,legend_place=None)

    # retrieval
    bar_names = ['Betweenness','Closeness','Current-flow betweenness','Current-flow closeness','Degree','Eigenvector','HITS authorities','HITS hubs','Load','PageRank']
    data = {
            'unweighted':[0.14606637651984622,0.17184314735361236,0.042019078720146409,0.17399729543537901,0.18149811054435275,0.19854658693196564,0.19854658693196564,0.19854658693196564,0.14700372822743263,0.17725358882165362],
            'weighted':[0.13586098100141117,0.18216618328598347,0.042019078720146409,0.17613717518129621,0.18821229318222113,0.17540014008712554,0.17540014008712554,0.17540014008712554,0.15104493506838745,0.17252331100724849]
        }
    plotter.bar_graph(data, bar_names, x_label='retrieval accuracy',colors=colors,legend_place=None)

def stanford_example():
    """
    Example/test of the stanford parser.
    """
    sentence = "Immediately after the second touchdown, the pilot decided to perform a go-around."
    pos, tree, dependencies = stanford_parser.parse(sentence)
    print '\nsentence:\n"'+sentence+'"'
    print '\ndependencies:\n',dependencies
    print '\ntree:\n',tree
    print '\npos:'
    for p in pos: print p.word()+'/'+p.tag(),
    print '\nTikZ pos tree:'
    print _pos_tree(tree)

    deps = preprocess.extract_dependencies(sentence)
    g = _create_dep_network(deps, True)
    nx.write_dot(g, 'report_imgs/stanford-example/graph.dot')

def _pos_tree(node):
    label = str(node.label())
    if node.isLeaf():
        result = label
    else:
        result = "[."+label+" "
        for c in node.children():
            result += _pos_tree(c)+" "
        result += " ]"
    return result

def _create_dep_network(deps, filter_tokens=False):
    graph = nx.DiGraph()
    for dep in deps:
        for tup in deps[dep]:
            g = tup[0][0]
            d = tup[1][0]
            if filter_tokens:
                g = preprocess.preprocess_token(g)
                d = preprocess.preprocess_token(d)
            if g is not None and d is not None:
                graph.add_edge(g, d, weight=1.0, label=dep) # add label
    return graph

def corpus_dependency_properties(dataset = 'air/problem_descriptions'):
    """
    Identify and pickle to file various properties of the given dataset.
    These can alter be converted to pretty tables using
    :func:`~experiments.print_network_props`.
    """
    print '> Reading data..', dataset
    corpus_path = '../data/'+dataset+'_dependencies'
    (documents, labels) = data.read_files(corpus_path)

    props = {}
    giant = nx.DiGraph()
    print '> Building networks..'
    for i, text in enumerate(documents):
        if i%10==0: print '   ',str(i)+'/'+str(len(documents))
        g = graph_representation.construct_dependency_network(text,remove_stop_words=True)
        giant.add_edges_from(g.edges())
        p = graph.network_properties(g)
        for k,v in p.iteritems():
            if i==0: props[k] = []
            props[k].append(v)
        g = None # just to make sure..

    print '> Calculating means and deviations..'
    props_total = {}
    for key in props:
        props_total[key+'_mean'] = numpy.mean(props[key])
        props_total[key+'_std'] = numpy.std(props[key])

    data.pickle_to_file(giant, 'output/properties/dependency/corpus_network_air_all_no_stop_words')
    data.pickle_to_file(props, 'output/properties/dependency/docs_air_all_no_stop_words')
    data.pickle_to_file(props_total, 'output/properties/dependency/docs_air_all_no_stop_words_total')

def evaluate_dep_types():
    """
    Leave-one-out evaluation of the various dependency types from the stanford parser.
    """
    exclude_list = ['dep', 'aux', 'auxpass', 'cop', 'agent', 'acomp',
                    'attr', 'ccomp', 'xcomp', 'complm', 'dobj', 'iobj',
                    'pobj', 'mark', 'rel', 'nsubj', 'nsubjpass', 'csubj',
                    'csubjpass', 'cc', 'conj', 'expl', 'abbrev', 'amod',
                    'appos', 'advcl', 'purpcl', 'det', 'predet', 'preconj',
                    'infmod', 'mwe', 'partmod', 'advmod', 'neg', 'rcmod',
                    'quantmod', 'tmod', 'nn', 'npadvmod', 'num', 'number',
                    'prep', 'poss', 'possessive', 'prt', 'parataxis',
                    'punct', 'ref', 'xsubj', 'pcomp', 'prepc']
    results = {'classification':[], 'retrieval':[]}

    print '------ CLASSIFICATION EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/tasa/TASA900_dependencies'
    texts, labels = data.read_files(descriptions_path)
    print '> Creating representations..'
    rep = {}
    for exclude_label in exclude_list:
        rep[exclude_label] = []
    metric  = graph.GraphMetrics.CLOSENESS
    for i, text in enumerate(texts):
        if i%10==0: print '   ',str(i)+'/'+str(len(texts))
        full_graph = graph_representation.construct_dependency_network(text)
        for exclude_label in exclude_list:
            g = graph.reduce_edge_set(full_graph, exclude_label)
            d = graph_representation.graph_to_dict(g, metric)
            rep[exclude_label].append(d)
            g = None # just to make sure..
        full_graph = None
    for exclude_label in exclude_list:
        rep[exclude_label] = graph_representation.dicts_to_vectors(rep[exclude_label])
    print '> Evaluating..'
    for exclude_label in exclude_list:
        score = evaluation.evaluate_classification(rep[exclude_label], labels)
        print '  ', exclude_label, score
        results['classification'].append(score)

    data.pickle_to_file(results, 'output/dependencies/types_eval_tmp')

    print '------ RETRIEVAL EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_dependencies'
    description_texts, labels = data.read_files(descriptions_path)
    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)
    print '> Creating representations..'
    rep = {}
    for exclude_label in exclude_list:
        rep[exclude_label] = []
    metric = graph.GraphMetrics.EIGENVECTOR
    for i, text in enumerate(description_texts):
        if i%1==0: print '   ',str(i)+'/'+str(len(description_texts))
        full_graph = graph_representation.construct_dependency_network(text)
        for exclude_label in exclude_list:
            g = graph.reduce_edge_set(full_graph, exclude_label)
            d = graph_representation.graph_to_dict(g, metric)
            rep[exclude_label].append(d)
            g = None # just to make sure..
        full_graph = None
        #~ if i%100==0: data.pickle_to_file(rep, 'output/dependencies/types_eval_rep_'+str(i))
    for exclude_label in exclude_list:
        rep[exclude_label] = graph_representation.dicts_to_vectors(rep[exclude_label])
    print '> Evaluating..'
    for exclude_label in exclude_list:
        score = evaluation.evaluate_retrieval(rep[exclude_label], solution_vectors)
        print '  ', exclude_label, score
        results['retrieval'].append(score)

    pp.pprint(results)
    data.pickle_to_file(results, 'output/dependencies/types_eval')

    return results

def evaluate_dep_type_sets():
    """
    Evaluation of various sets of dependency relations.

    Each set is excluded from the representation, and the performance recorded.
    The best strategy is to exclude those dependencies which removal lead to the
    greatest imporovement for the representation.
    """
    strategies = {
            'defensive': ['agent', 'advcl', 'parataxis'],
            'aggressive': ['agent', 'advcl', 'parataxis', 'dep', 'aux', 'ccomp', 'xcomp', 'dobj', 'pobj', 'nsubj', 'nsubjpass', 'cc', 'abbrev', 'purpcl', 'predet', 'preconj', 'advmod', 'neg', 'rcmod', 'tmod', 'poss', 'prepc'],
            'compromise_1': ['agent', 'advcl', 'parataxis', 'aux', 'xcomp', 'pobj', 'nsubjpass', 'cc', 'abbrev', 'purpcl', 'predet', 'neg', 'tmod', 'poss', 'prepc'],
            'compromise_2': ['agent', 'advcl', 'parataxis', 'aux', 'xcomp', 'pobj', 'nsubjpass', 'cc', 'abbrev', 'purpcl', 'predet', 'neg', 'tmod', 'poss', 'prepc', 'attr', 'csubj', 'csubjpass', 'number', 'possessive', 'punct', 'ref']
        }
    results = {'classification':{}, 'retrieval':{}}

    print '------ CLASSIFICATION EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/tasa/TASA900_dependencies'
    texts, labels = data.read_files(descriptions_path)
    print '> Creating representations..'
    rep = {}
    for strategy in strategies:
        rep[strategy] = []
    metric  = graph.GraphMetrics.CLOSENESS
    for i, text in enumerate(texts):
        if i%10==0: print '   ',str(i)+'/'+str(len(texts))
        for strategy in strategies:
            g = graph_representation.construct_dependency_network(text, exclude=strategies[strategy])
            d = graph_representation.graph_to_dict(g, metric)
            rep[strategy].append(d)
            g = None # just to make sure. I don't trust this damn garbage collector...
    for strategy in strategies:
        rep[strategy] = graph_representation.dicts_to_vectors(rep[strategy])
    print '> Evaluating..'
    for strategy in strategies:
        score = evaluation.evaluate_classification(rep[strategy], labels)
        print '  ', strategy, score
        results['classification'][strategy] = score

    data.pickle_to_file(results, 'output/dependencies/types_set_eval_tmp')

    print '------ RETRIEVAL EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_dependencies'
    description_texts, labels = data.read_files(descriptions_path)
    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)
    print '> Creating representations..'
    rep = {}
    for strategy in strategies:
        rep[strategy] = []
    metric = graph.GraphMetrics.EIGENVECTOR
    for i, text in enumerate(description_texts):
        if i%1==0: print '   ',str(i)+'/'+str(len(description_texts))
        full_graph = graph_representation.construct_dependency_network(text)
        for strategy in strategies:
            g = graph_representation.construct_dependency_network(text, exclude=strategies[strategy])
            d = graph_representation.graph_to_dict(g, metric)
            rep[strategy].append(d)
            g = None # just to make sure..
        full_graph = None
        #~ if i%100==0: data.pickle_to_file(rep, 'output/dependencies/types_eval_rep_'+str(i))
    for strategy in strategies:
        rep[strategy] = graph_representation.dicts_to_vectors(rep[strategy])
    print '> Evaluating..'
    for strategy in strategies:
        score = evaluation.evaluate_retrieval(rep[strategy], solution_vectors)
        print '  ', strategy, score
        results['retrieval'][strategy] = score

    pp.pprint(results)
    data.pickle_to_file(results, 'output/dependencies/types_set_eval')

    return results

def plot_type_sets_evaluation():
    """
    Plot results from the :func:`evaluate_dep_type_sets` experiment.
    """
    strategies = ['defensive', 'aggressive', 'compromise 1', 'compromise 2', 'all types']
    #~ results = data.pickle_from_file('output/dependencies/types_set_eval')
    results = { 'retrieval': {
                   'aggressive':   0.0,
                   'defensive':    0.0,
                   'compromise_1': 0.0,
                   'compromise_2': 0.0},
                'classification': {
                   'aggressive':   0.505555555556,
                   'defensive':    0.588888888889,
                   'compromise_1': 0.563888888889,
                   'compromise_2': 0.558333333333}}
    class_data = [results['classification']['defensive'],
                  results['classification']['aggressive'],
                  results['classification']['compromise_1'],
                  results['classification']['compromise_2'],
                  0.5750]
    retr_data =  [results['retrieval']['defensive'],
                  results['retrieval']['aggressive'],
                  results['retrieval']['compromise_1'],
                  results['retrieval']['compromise_2'],
                  0.1985]
    colors = ['#3C54FF','#EF4C32','#27A713']
    #~ plotter.bar_graph({'_':class_data}, strategies, x_label='classification accuracy',colors=colors,legend_place=None)
    plotter.bar_graph({'_':retr_data}, strategies, x_label='retrieval accuracy',colors=colors,legend_place=None)


def plot_type_evaluation():
    """
    Plot results from the :func:`evaluate_dep_types` experiment.
    """
    l = ['dep', 'aux', 'auxpass', 'cop', 'agent', 'acomp',
        'attr', 'ccomp', 'xcomp', 'complm', 'dobj', 'iobj',
        'pobj', 'mark', 'rel', 'nsubj', 'nsubjpass', 'csubj',
        'csubjpass', 'cc', 'conj', 'expl', 'abbrev', 'amod',
        'appos', 'advcl', 'purpcl', 'det', 'predet', 'preconj',
        'infmod', 'mwe', 'partmod', 'advmod', 'neg', 'rcmod',
        'quantmod', 'tmod', 'nn', 'npadvmod', 'num', 'number',
        'prep', 'poss', 'possessive', 'prt', 'parataxis',
        'punct', 'ref', 'xsubj', 'pcomp', 'prepc']
    d = data.pickle_from_file('output/dependencies/types_eval_class')
    diffs  = []
    print '--- Classification ---'
    for i, dep_type in enumerate(l):
        val = d['classification'][i]
        diff = val - 0.5750
        diffs.append(diff)
        print "\t\t\t"+dep_type+"  &  "+'%1.4f'%val+"  &  "+'%1.4f'%diff +"\\\\"

    d = data.pickle_from_file('output/dependencies/types_eval_retr')
    diffs  = []
    print '--- Retrieval ---'
    for i, dep_type in enumerate(l):
        val = d['retrieval'][i]
        diff = val - 0.1985
        diffs.append(diff)
        print "\t\t\t"+dep_type+"  &  "+'%1.4f'%val+"  &  "+'%1.4f'%diff +"\\\\"

def print_common_hub_words(rem_stop_words):
    """
    Print a list of the most common hub words in the created networks.
    Purpose of experiment was to show that hub words typically are stop words.

    The *rem_stop_words* determine whether stop words are removed before creating
    the networks.
    """
    results = {'_removing stop-words':rem_stop_words}

    print '------ CLASSIFICATION EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/tasa/TASA900_dependencies'
    texts, labels = data.read_files(descriptions_path)

    print '> Creating representations..'
    fd = nltk.probability.FreqDist()
    for i, text in enumerate(texts):
        if i%100==0: print '   ',str(i)+'/'+str(len(texts))
        g = graph_representation.construct_dependency_network(text, remove_stop_words=rem_stop_words)
        hubs = graph.get_hubs(g, 10)
        for h in hubs:
            fd.inc(h[0])
        g = None # just to make sure..

    results['tasa'] = fd.keys()

    print '------ RETRIEVAL EVALUATION --------'
    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_dependencies'
    description_texts, labels = data.read_files(descriptions_path)

    print '> Creating representations..'
    fd = nltk.probability.FreqDist()
    for i, text in enumerate(description_texts):
        if i%100==0: print '   ',str(i)+'/'+str(len(description_texts))
        g = graph_representation.construct_dependency_network(text, remove_stop_words=rem_stop_words)
        hubs = graph.get_hubs(g, 10)
        for h in hubs:
            fd.inc(h[0])
        g = None # just to make sure..

    results['air'] = fd.keys()

    if rem_stop_words:
        modifier = 'without'
    else:
        modifier = 'with'
    data.pickle_to_file(results, 'output/dependencies/common_hubs_'+modifier+'stop_words')

    pp.pprint(results)
    return results

def print_hubs():
    """
    Print results from :func:`print_common_hub_words` as latex table.
    """
    w = data.pickle_from_file('output/dependencies/common_hubs_withstop_words')
    wo = data.pickle_from_file('output/dependencies/common_hubs_withoutstop_words')

    tasa_w = [term.encode('ascii','ignore') for term in w['tasa'][:10]]
    air_w = [term.encode('ascii','ignore') for term in w['air'][:10]]
    tasa_wo = [term.encode('ascii','ignore') for term in wo['tasa'][:10]]
    air_wo = [term.encode('ascii','ignore') for term in wo['air'][:10]]

    for i in range(10):
        print tasa_w[i],' & ',air_w[i],' & ',tasa_wo[i],' & ',air_wo[i],'\\\\'

def corpus_properties(dataset):
    """
    Identify and pickle to file various properties of the given dataset.
    These can alter be converted to pretty tables using
    :func:`~experiments.print_network_props`.
    """
    print '> Reading data..', dataset
    corpus_path = '../data/'+dataset+'_dependencies'
    (documents, labels) = data.read_files(corpus_path)

    props = {}
    #~ giant = nx.DiGraph()
    print '> Building networks..'
    for i, deps in enumerate(documents):
        if i%10==0: print '   ',str(i)+'/'+str(len(documents))
        g = graph_representation.construct_cooccurrence_network(deps)
        #~ giant.add_edges_from(g.edges())
        p = graph.network_properties(g)
        for k,v in p.iteritems():
            if i==0: props[k] = []
            props[k].append(v)
        g = None # just to make sure..

    print '> Calculating means and deviations..'
    props_total = {}
    for key in props:
        print '   ',key
        props_total[key+'_mean'] = numpy.mean(props[key])
        props_total[key+'_std'] = numpy.std(props[key])

    data_name = dataset.replace('/','.')
    #~ data.pickle_to_file(giant, 'output/properties/cooccurrence/giant_'+data_name)
    data.pickle_to_file(props, 'output/properties/dependency/stats_'+data_name)
    data.pickle_to_file(props_total, 'output/properties/dependency/stats_tot_'+data_name)

def print_degree_distributions(dataset):
    """
    Extracts degree distribution values from networks, and print them to
    cvs-file.

    **warning** overwrites if file exists.
    """
    print '> Reading data..', dataset
    corpus_path = '../data/'+dataset+'_dependencies'
    (documents, labels) = data.read_files(corpus_path)

    degsfile = open('output/properties/dependency/degrees_docs_'+dataset.replace('/','.'), 'w')

    giant = nx.DiGraph()
    print '> Building networks..'
    for i, text in enumerate(documents):
        if i%10==0: print '   ',str(i)+'/'+str(len(documents))
        g = graph_representation.construct_dependency_network(text)
        giant.add_edges_from(g.edges())
        degs = nx.degree(g).values()
        degs = [str(d) for d in degs]
        degsfile.write(','.join(degs)+'\n')
    degsfile.close()

    print '> Writing giant\'s distribution'
    with open('output/properties/dependency/degrees_giant_'+dataset.replace('/','.'), 'w') as f:
        ds = nx.degree(giant).values()
        ds = [str(d) for d in ds]
        f.write(','.join(ds))

def compare_stats_to_random(dataset):
    dataset = dataset.replace('/','.')
    stats = data.pickle_from_file('output/properties/dependency/stats_tot_'+dataset)
    n = stats['# nodes_mean']
    p = stats['mean degree_mean']/(2*n)
    g = nx.directed_gnp_random_graph(int(n), p)
    props = graph.network_properties(g)
    pp.pprint(props)

def evaluate_tc_icc_classification():
    graph_metrics = graph_representation.get_metrics(False)

    print '> Reading cases..'
    path = '../data/tasa/TASA900_dependencies'
    #~ path = '../data/tasa/TASATest_dependencies'
    texts, labels = data.read_files(path)

    print '> Building corpus graph..'
    gdeps = {}
    for i, text in enumerate(texts):
        if i%10==0: print '   ',str(i)+'/'+str(len(texts))
        d = pickle.loads(text)
        for dep in d.keys():
            gdeps[dep] = gdeps.get(dep, []) + d[dep]
    giant = graph_representation.construct_dependency_network(pickle.dumps(gdeps),verbose=True)
    data.pickle_to_file(giant, 'output/giants/dependency/classification.net')

    rep = {}
    icc = {}
    print '> Calculating ICCs..'
    for metric in graph_metrics:
        print
        print metric
        rep[metric] = []
        try:
            icc[metric] = graph_representation.calculate_icc_dict(giant, metric)
            data.pickle_to_file(giant, 'output/output/tc_icc/dependency/classification.icc')
        except:
            print "GOD FUCKING DAMN IT. FUCKING TOO LITTLE MEMORY DAMN IT. FUCK."
            icc[metric] = None

    print '> Creating graph representations..'
    for i, text in enumerate(texts):
        if i%10==0: print '   ',str(i)+'/'+str(len(texts))
        g = graph_representation.construct_dependency_network(text)
        for metric in graph_metrics:
            if not icc[metric]: continue
            d = graph_representation.graph_to_dict(g, metric, icc[metric])
            rep[metric].append(d)
        g = None # just to make sure..

    print '> Creating vector representations..'
    for metric in graph_metrics:
        if not icc[metric]: continue
        rep[metric] = graph_representation.dicts_to_vectors(rep[metric])

    print '> Evaluating..'
    results = {}
    for metric in graph_metrics:
        if not icc[metric]:
            results[metric] = None
            continue
        vectors = rep[metric]
        score = evaluation.evaluate_classification(vectors, labels)
        print '   ', metric, score
        results[metric] = score

    pp.pprint(results)
    data.pickle_to_file(results, 'output/tc_icc/dependency/classification.res')
    return results

if __name__ == "__main__":
    #~ centrality_weights_classification(True)
    #~ centrality_weights_classification(False)
    #~ centrality_weights_retrieval(True)
    #~ centrality_weights_retrieval(False)
    #~ plot_exp1()
    #~ plot_type_evaluation()
    #~ pp.pprint(data.pickle_from_file('output/dependencies/exp1_retr_weighted'))
    #~ stanford_example()
    #~ stop_word_evaluation(True)
    #~ stop_word_evaluation(False)
    #~ edge_direction_evaluation('forward')
    #~ edge_direction_evaluation('backward')
    #~ evaluate_dep_types()
    #~ print_hubs()
    #~ corpus_dependency_properties()
    #~ print_common_hub_words(False)
    #~ evaluate_dep_type_sets()
    #~ plot_type_sets_evaluation()
    #~ centrality_weights_retrieval(False)

    #~ corpus_properties('tasa/TASA900')
    #~ corpus_properties('air/problem_descriptions')
    #~ compare_stats_to_random('tasa/TASA900')
    #~ compare_stats_to_random('air/problem_descriptions')

    #~ print_degree_distributions('tasa/TASA900')
    #~ print_degree_distributions('air/problem_descriptions')

    evaluate_tc_icc_classification()
