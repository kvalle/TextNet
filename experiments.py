"""
Module containing methods for experimenting with the various graph representations.
Experiments not particular to any single representation is put here,
e.g. comparisons of the representations , or tests of properties of the datasets.
"""
import pprint as pp
import numpy
import scipy.spatial.distance

import data
import graph
import freq_representation
import graph_representation
import evaluation
import plotter
import preprocess

numpy.set_printoptions(linewidth = 1000, precision = 3)

def classification_comparison_graph(dataset='reuters', graph_type='co-occurrence', icc=None):
    """
    Experiment used for comparative evaluation of different network
    representations on classification.

    graph_type = 'co-occurrence' | 'dependency'

    `icc` determines whether to use _inverse corpus centrality_ in the vector representations.
    """
    def make_dicts(docs, icc):
        rep = []
        for i, doc in enumerate(docs):
            if i%100==0: print '    graph',str(i)+'/'+str(len(docs))
            g = gfuns[graph_type](doc)
            d = graph_representation.graph_to_dict(g, metrics[graph_type], icc)
            rep.append(d)
        return rep

    postfix = {'co-occurrence':'_text', 'dependency':'_dependencies'}
    gfuns = {'co-occurrence':graph_representation.construct_cooccurrence_network,
                'dependency':graph_representation.construct_dependency_network}
    metrics = {'co-occurrence':graph.GraphMetrics.WEIGHTED_DEGREE,
                'dependency':graph.GraphMetrics.CLOSENESS}

    print '--', graph_type
    print '> Reading data..', dataset
    training_path = '../data/'+dataset+'/training'+postfix[graph_type]
    training_docs, training_labels = data.read_files(training_path)
    test_path = '../data/'+dataset+'/test'+postfix[graph_type]
    test_docs, test_labels = data.read_files(test_path)

    icc_training = None
    icc_test = None
    if icc:
        print '> Calculating ICC..'
        print '    training giant'
        gdoc_training = ' '.join(training_docs)
        giant_training = gfuns[graph_type](gdoc_training)
        print '    training icc'
        icc_training = graph_representation.calculate_icc_dict(giant_training, metrics[graph_type])
        print '    test giant'
        gdoc_test = ' '.join(test_docs)
        giant_test = gfuns[graph_type](gdoc_test)
        print '    test icc'
        icc_test = graph_representation.calculate_icc_dict(giant_test, metrics[graph_type])

    print '> Creating representations..'
    training_dicts = make_dicts(training_docs, icc_training)
    test_dicts = make_dicts(test_docs, icc_test)

    print '    dicts -> vectors'
    keys = set()
    for d in training_dicts + test_dicts:
        keys = keys.union(d.keys())
    keys = list(keys)
    print '    vocabulary size:', len(keys)

    training_rep = graph_representation.dicts_to_vectors(training_dicts, keys)
    test_rep = graph_representation.dicts_to_vectors(test_dicts, keys)

    print '> Evaluating..'
    reps = {'training':training_rep, 'test':test_rep}
    labels = {'training':training_labels, 'test':test_labels}
    results = evaluation.evaluate_classification(reps, labels, mode='split')
    print results
    s = 'classification comparison '
    if icc: s += 'USING TC-ICC'
    s += '\nrepresentation: '+graph_type+'\nresult: '+str(results)+'\n\n\n'
    data.write_to_file(s, 'output/comparison/classification')
    return results

def classification_comparison_freq(dataset='reuters'):
    print '> Reading data..', dataset
    training_path = '../data/'+dataset+'/training_preprocessed'
    training_docs, training_labels = data.read_files(training_path)
    test_path = '../data/'+dataset+'/test_preprocessed'
    test_docs, test_labels = data.read_files(test_path)

    results = {}
    for metric in freq_representation.get_metrics():
        print '   ', metric,
        training_dicts = freq_representation.text_to_dict(training_docs, metric)
        test_dicts = freq_representation.text_to_dict(test_docs, metric)
        print '    dicst -> vectors'
        keys = set()
        for d in training_dicts + test_dicts:
            keys = keys.union(d.keys())
        print '    vocabulary size:', len(keys)
        training_rep = graph_representation.dicts_to_vectors(training_dicts, keys)
        test_rep = graph_representation.dicts_to_vectors(test_dicts, keys)
        reps = {'training':training_rep, 'test':test_rep}
        labels = {'training':training_labels, 'test':test_labels}
        score = evaluation.evaluate_classification(reps, labels, mode='split')
        results[metric] = score
        print score
    pp.pprint(results)
    s = 'classification comparison \nrepresentation: frequency\nresult:\n'+str(results)+'\n\n\n'
    data.write_to_file(s, 'output/comparison/classification')
    return results

def retrieval_comparison_graph(dataset='air', graph_type='co-occurrence', icc=None):
    """
    Experiment used for comparative evaluation of different network
    representations on retrieval.

    graph_type = 'co-occurrence' | 'dependency'

    `icc` determines whether to use _inverse corpus centrality_ in the vector representations.
    """
    def make_dicts(docs):
        rep = []
        for i, doc in enumerate(docs):
            if i%100==0: print '    graph',str(i)+'/'+str(len(docs))
            g = gfuns[graph_type](doc)
            d = graph_representation.graph_to_dict(g, metrics[graph_type], icc)
            rep.append(d)
        return rep

    postfix = {'co-occurrence':'_text', 'dependency':'_dependencies'}
    gfuns = {'co-occurrence':graph_representation.construct_cooccurrence_network,
                'dependency':graph_representation.construct_dependency_network}
    metrics = {'co-occurrence':graph.GraphMetrics.WEIGHTED_DEGREE,
                'dependency':graph.GraphMetrics.EIGENVECTOR}

    print '--', graph_type
    print '> Reading data..', dataset
    path = '../data/'+dataset+'/problem_descriptions'+postfix[graph_type]
    docs, labels = data.read_files(path)

    print '> Creating solution representations..'
    solutions_path = '../data/'+dataset+'/solutions_preprocessed'
    solutions_texts, labels = data.read_files(solutions_path)
    solutions_rep = freq_representation.text_to_vector(solutions_texts, freq_representation.FrequencyMetrics.TF_IDF)

    if icc:
        print '> Constructing giant and calculating ICC..'
        gdoc = ' '.join(docs)
        giant = gfuns[graph_type](gdoc)
        icc = graph_representation.calculate_icc_dict(giant_training, metrics[graph_type])

    print '> Creating problem description representations..'
    dicts = make_dicts(docs)
    descriptions_rep = graph_representation.dicts_to_vectors(dicts)

    print '> Evaluating..'
    results = evaluation.evaluate_retrieval(descriptions_rep, solutions_rep)
    print results
    s = 'retrieval comparison '
    if icc: s += 'USING TC-ICC'
    s += '\nrepresentation: '+graph_type+'\nresult: '+str(results)+'\n\n\n'
    data.write_to_file(s, 'output/comparison/retrieval')
    return results

def retrieval_comparison_freq(dataset='mir'):
    print '> Reading data..', dataset
    path = '../data/'+dataset+'/problem_descriptions_preprocessed'
    docs, _ = data.read_files(path)

    print '> Creating solution representations..'
    solutions_path = '../data/'+dataset+'/solutions_preprocessed'
    solutions_docs, _ = data.read_files(solutions_path)
    solutions_rep = freq_representation.text_to_vector(solutions_docs, freq_representation.FrequencyMetrics.TF_IDF)

    print '> Evaluating..'
    results = {}
    for metric in freq_representation.get_metrics():
        print '   ', metric,
        descriptions_rep = freq_representation.text_to_vector(docs, metric)
        score = evaluation.evaluate_retrieval(descriptions_rep, solutions_rep)
        results[metric] = score
        print score
    pp.pprint(results)
    s = 'retrieval comparison \nrepresentation: frequency\nresult:\n'+str(results)+'\n\n\n'
    data.write_to_file(s, 'output/comparison/retrieval')
    return results

def do_classification_experiments(dataset='tasa/TASA900',
                                    graph_types = ['co-occurrence','dependency','random'],
                                    use_frequency = True):
    """
    Experiment used for comparative evaluation of different network
    representations on classification.

    Toggle comparison with frequency-based methods using *use_frequency*.
    """
    results = {'_dataset':dataset,
                '_evaluation':'classification'}
    print '> Evaluation type: classification'
    print '> Reading data..', dataset
    corpus_path = '../data/'+dataset
    docdata = data.read_data(corpus_path, graph_types)

    print '> Evaluating..'
    for gtype in graph_types:
        print '   ',gtype
        documents, labels = docdata[gtype]
        graphs = graph_representation.create_graphs(documents, gtype)
        results[gtype] = {}
        for metric in graph_representation.get_metrics():
            print '    -', metric
            vectors = graph_representation.graphs_to_vectors(graphs, metric)
            results[gtype][metric] = evaluation.evaluate_classification(vectors, labels)
    if use_frequency:
        print '    frequency'
        results['freq'] = {}
        for metric in freq_representation.get_metrics():
            print '    -', metric
            documents, labels = data.read_files(corpus_path+'_preprocessed')
            vectors = freq_representation.text_to_vector(documents, metric)
            results['freq'][metric] = evaluation.evaluate_classification(vectors, labels)

    print
    pp.pprint(results)
    return results

def do_retrieval_experiments(descriptions='air/problem_descriptions',
                                solutions='air/solutions',
                                graph_types=['co-occurrence','dependency','random'],
                                use_frequency=True):
    """
    Experiment used for comparative evaluation of different network
    representations on the retrieval task.

    Toggle comparison with frequency-based methods using *use_frequency*.
    """
    results = {'_solutions':solutions,
                '_descriptions':descriptions,
                '_evaluation':'retrieval'}

    print '> Evaluation type: retrieval'
    print '> Reading cases..'
    descriptions_path = '../data/'+descriptions
    descriptiondata = data.read_data(descriptions_path, graph_types)

    solutions_path = '../data/'+solutions+'_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)

    print '> Evaluating..'
    for gtype in graph_types:
        print '   ',gtype
        docs, labels = descriptiondata[gtype]
        graphs = graph_representation.create_graphs(docs, gtype)
        results[gtype] = {}
        for metric in graph_representation.get_metrics():
            print '    -', metric
            vectors = graph_representation.graphs_to_vectors(graphs, metric)
            results[gtype][metric] = evaluation.evaluate_retrieval(vectors, solution_vectors)
    if use_frequency:
        print '    frequency'
        results['freq'] = {}
        for metric in freq_representation.get_metrics():
            print '    -', metric
            docs, labels = data.read_files(descriptions_path+'_preprocessed')
            vectors = freq_representation.text_to_vector(docs, metric)
            results['freq'][metric] = evaluation.evaluate_retrieval(vectors, solution_vectors)

    print
    pp.pprint(results)
    return results

def plot_sentence_lengths(datafile=None):
    """
    Function for plotting histogram of sentence lengths within a given dataset.
    """
    if datafile is None:
        import preprocess
        print '> reading data..'
        path = '../data/tasa/TASA900_text'
        texts, labels = data.read_files(path)
        sentence_lengths = []
        print '> counting lengths..'
        for text in texts:
            sentences = preprocess.tokenize_sentences(text)
            for sentence in sentences:
                tokens = preprocess.tokenize_tokens(sentence)
                sentence_lengths.append(len(tokens))
        data.pickle_to_file(sentence_lengths, 'output/tasa_sentence_lengths.pkl')
    else:
        sentence_lengths = data.pickle_from_file(datafile)
    plotter.histogram(sentence_lengths, 'sentence length (tokens)', '# sentences', bins=70)

def print_network_props():
    """
    Prints latex table with various properties for networks created from
    texts in the datasets.
    """
    print '-- Co-occurrence'
    tasa = data.pickle_from_file('output/properties/cooccurrence/stats_tot_tasa.TASA900')
    air = data.pickle_from_file('output/properties/cooccurrence/stats_tot_air.problem_descriptions')
    for key in air.keys():
        prop, sep, mod = key.partition('_')
        if mod!='std':
            print prop,' & ',
            print '%2.3f'%tasa[prop+sep+'mean'],' & ','%2.3f'%tasa[prop+sep+'std'],' & ',
            print '%2.3f'%air[prop+sep+'mean'],' & ','%2.3f'%air[prop+sep+'std'],'\\\\'
    print
    print '-- Dependency, all types'
    air = data.pickle_from_file('output/properties/dependency/stats_tot_air.problem_descriptions')
    tasa = data.pickle_from_file('output/properties/dependency/stats_tot_tasa.TASA900')
    for key in air.keys():
        prop, sep, mod = key.partition('_')
        if mod!='std':
            print prop,' & ',
            print '%2.3f'%tasa[prop+sep+'mean'],' & ','%2.3f'%tasa[prop+sep+'std'],' & ',
            print '%2.3f'%air[prop+sep+'mean'],' & ','%2.3f'%air[prop+sep+'std'],'\\\\'

def dataset_stats(dataset):
    """
    Print and plot statistics for a given dataset.
    A histogram is plotted with the document length distribution of the data.
    """
    print '> Reading data..', dataset
    corpus_path = '../data/'+dataset
    (documents, labels) = data.read_files(corpus_path)
    file_names = data.get_file_names(corpus_path)
    lengths = []
    empty = 0
    for i,d in enumerate(documents):
        d = preprocess.tokenize_tokens(d)
        lengths.append(len(d))
        if len(d)==0:
            print file_names[i],'is empty'
            empty += 1
    lengths = numpy.array(lengths)
    print '# documents:',len(documents)
    print '# empty documents:',empty
    print '# words:',sum(lengths)
    print 'length avg:',lengths.mean()
    print 'length stddev:',lengths.std()
    print
    print 'document lengths (sorted):',sorted(lengths)
    plotter.histogram(lengths,'# tokens','# documents','',bins=80)

def solution_similarity_stats(dataset='air/solutions_preprocessed'):
    """
    Plots histogram of solution-solution similarity distribution of a dataset.
    """
    print '> Reading data..', dataset
    corpus_path = '../data/'+dataset
    (documents, labels) = data.read_files(corpus_path)

    print '> Creating vector representations..'
    vectors = freq_representation.text_to_vector(documents, freq_representation.FrequencyMetrics.TF_IDF)

    print '> Calculating similarities..'
    distances = scipy.spatial.distance.cdist(vectors.T, vectors.T, 'cosine')
    diag = numpy.diag([2.0]*len(distances),0) # move similarities of "self" to -1
    distances = distances + diag
    similarities = 1.0 - distances
    similarities = similarities.ravel()
    similarities = [s for s in similarities if s >= 0]
    print plotter.histogram(similarities,'similarity','# matches','',bins=150)
    print
    print max(similarities)
    print min(similarities)
    print float(sum(similarities))/len(similarities)
    num = len([sim for sim in similarities if sim < 0.23])
    print 'fraction sims < .23:', float(num)/len(similarities)

def test_document_lengths(dataset='mir'):
    print '> Reading data..', dataset
    path = '../data/'+dataset+'/problem_descriptions_preprocessed'
    docs, _ = data.read_files(path)
    names = data.get_file_names(path)
    print "PROBLEM DESCRIPTIONS"
    for i, d in enumerate(docs):
        if not d:
            print names[i], "is empty"
    path = '../data/'+dataset+'/solutions_preprocessed'
    docs, _ = data.read_files(path)
    names = data.get_file_names(path)
    print "SOLUTIONS"
    for i, d in enumerate(docs):
        if not d:
            print names[i], "is empty"

def term_centrality_study(doc='air/reports_text/2005/a05a0059.html', num=20):
    def _print_terms(cents, rep, num):
        ts = _top_cents(cents, num)
        terms = []
        for t in ts:
            terms.append(t[0])
        print rep + ' & ' + ', '.join(terms) + ' \\\\'
    def _top_cents(cents,num):
        return sorted(cents.iteritems(), key = operator.itemgetter(1), reverse = True)[0:num]
    def _calc_cents(g, metric, gcents=None):
        if gcents: icc = graph_representation.calculate_icc_dict(gcents)
        else: icc = None
        return graph_representation.graph_to_dict(g, metric, icc)

    import operator
    import dependency_experiments
    import co_occurrence_experiments

    dataset = 'air/problem_descriptions'
    path = '../data/'+doc
    doc = data.read_file(path)

    metric = graph.GraphMetrics.DEGREE
    context = 'window'
    g = graph_representation.construct_cooccurrence_network(doc, context=context)
    cents = _calc_cents(g, metric)
    _print_terms(cents, 'Co-occurrence TC', num)
    gcents = co_occurrence_experiments.retrieve_centralities(dataset, context, metric)
    cents = _calc_cents(g, metric, gcents)
    _print_terms(cents, 'Co-occurrence TC-ICC', num)

    metric = graph.GraphMetrics.EIGENVECTOR
    deps = data._text_to_dependencies(doc)
    g = graph_representation.construct_dependency_network(deps)
    cents = _calc_cents(g, metric)
    _print_terms(cents, 'Dependency TC', num)
    #~ gcents = dependency_experiments.retrieve_centralities(dataset, metric)
    #~ cents = _calc_cents(g, metric, gcents)
    #~ _print_terms(cents, 'Dependency TC-ICC', num)

    fdict = freq_representation.text_to_dict([doc], freq_representation.FrequencyMetrics.TF_IDF)[0]
    _print_terms(fdict, 'TF-IDF', num)

    fdict = freq_representation.text_to_dict([doc], freq_representation.FrequencyMetrics.TF)[0]
    _print_terms(fdict, 'TF', num)

if __name__ == "__main__":
    #~ do_classification_experiments('tasa/TASA900',[])
    #~ do_retrieval_experiments('air/problem_descriptions', 'air/solutions',[])
    #~ plot_sentence_lengths('output/tasa_sentence_lengths.pkl')
    #~ print_network_props()
    #~ dataset_stats('tasa/TASA900_text')
    #~ solution_similarity_stats()

    #~ classification_comparison_graph(graph_type='co-occurrence', icc=True)
    #~ classification_comparison_graph(graph_type='dependency', icc=True)
    #~ classification_comparison_freq()

    #~ retrieval_comparison_graph(dataset='mir', graph_type='co-occurrence', icc=True)
    #~ retrieval_comparison_graph(dataset='mir', graph_type='dependency', icc=True)
    #~ retrieval_comparison_freq()

    #~ test_document_lengths()
    #~ solution_similarity_stats(dataset='mir/solutions_preprocessed')

    #~ solution_similarity_stats()
    term_centrality_study()
