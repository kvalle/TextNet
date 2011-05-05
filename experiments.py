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

def classification_comparison_graph(dataset='reuters', graph_type='co-occurrence'):
    """
    Experiment used for comparative evaluation of different network
    representations on classification.

    graph_type = 'co-occurrence' | 'dependency'

    Toggle comparison with frequency-based methods using *use_frequency*.
    """
    def make_dicts(docs):
        rep = []
        for i, doc in enumerate(docs):
            if i%100==0: print '    graph',str(i)+'/'+str(len(docs))
            g = gfuns[graph_type](doc)
            d = graph_representation.graph_to_dict(g, metrics[graph_type])
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

    print '> Creating representations..'
    training_dicts = make_dicts(training_docs)
    test_dicts = make_dicts(test_docs)

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
    s = 'classification comparison \nrepresentation: '+graph_type+'\nresult: '+str(results)+'\n\n\n'
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

if __name__ == "__main__":
    #~ do_classification_experiments('tasa/TASA900',[])
    #~ do_retrieval_experiments('air/problem_descriptions', 'air/solutions',[])
    #~ plot_sentence_lengths('output/tasa_sentence_lengths.pkl')
    #~ print_network_props()
    #~ dataset_stats('tasa/TASA900_text')
    #~ solution_similarity_stats()

    #~ classification_comparison_graph(graph_type='co-occurrence')
    #~ classification_comparison_graph(graph_type='dependency')
    classification_comparison_freq()
