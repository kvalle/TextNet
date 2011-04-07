"""
Module containing methods for experimenting with the various graph representations.
Experiments not particular to any single representation is put here,
e.g. comparisons of the representations , or tests of properties of the datasets.

Warning: This module probably contain a lot of redundant code and is a mess most of the time.
This is because it contains experiments constructed for specific purposes that are hard
to predict ahead of time. When done, the experiments are left as is, to be available for
re-runs later if needed.
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

def do_classification_experiments(dataset='tasa/TASA900',
                                    graph_types = ['co-occurrence','dependency','random'],
                                    use_frequency = True):
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
    print '-- Co-occurrence'
    tasa = data.pickle_from_file('output/properties/cooccurrence/docs_tasa_sentence_total')
    air = data.pickle_from_file('output/properties/cooccurrence/docs_air_2_total')
    for key in air.keys():
        prop, sep, mod = key.partition('_')
        if mod!='std':
            print prop,' & ',
            print '%2.3f'%tasa[prop+sep+'mean'],' & ','%2.3f'%tasa[prop+sep+'std'],' & ',
            print '%2.3f'%air[prop+sep+'mean'],' & ','%2.3f'%air[prop+sep+'std'],'\\\\'
    print
    print '-- Dependency, all types'
    air = data.pickle_from_file('output/properties/dependency/docs_air_all_total')
    tasa = data.pickle_from_file('output/properties/dependency/docs_tasa_all_total')
    for key in air.keys():
        prop, sep, mod = key.partition('_')
        if mod!='std':
            print prop,' & ',
            print '%2.3f'%tasa[prop+sep+'mean'],' & ','%2.3f'%tasa[prop+sep+'std'],' & ',
            print '%2.3f'%air[prop+sep+'mean'],' & ','%2.3f'%air[prop+sep+'std'],'\\\\'

def dataset_stats(dataset):
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
    solution_similarity_stats()
