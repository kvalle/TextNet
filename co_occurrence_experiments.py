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

def do_context_size_evaluation_retrieval():
    """Experiment intended to evaluate the performance with different
    context sizes for co-occurrence networks in retrieval."""
    results = {}
    graph_metrics = graph_representation.get_metrics()
    for metric in graph_metrics:
        results[metric] = []

    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_preprocessed'
    description_texts, labels = data.read_files(descriptions_path)

    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)

    for window_size in range(1,11)+[20,40,80]:
        print '-- window size:',window_size

        rep = {}
        for metric in graph_metrics:
            rep[metric] = []
        print '> Creating representations..'

        # creating graphs and finding centralities
        for i, text in enumerate(description_texts):
            if i%10==0: print i
            g = graph_representation.construct_cooccurrence_network(text, window_size=window_size, already_preprocessed=True)
            for metric in graph_metrics:
                d = graph_representation.graph_to_dict(g, metric)
                rep[metric].append(d)
            g = None # just to make sure..

        # creating representation vectors
        for metric in graph_metrics:
            rep[metric] = graph_representation.dicts_to_vectors(rep[metric])

        print '> Evaluating..'
        for metric in graph_metrics:
            vectors = rep[metric]
            score = evaluation.evaluate_retrieval(vectors, solution_vectors)
            print '   ', metric, score
            results[metric].append(score)

        data.pickle_to_file(results, 'output/retr_context_'+str(window_size))

    pp.pprint(results)
    return results

def do_context_size_evaluation_classification():
    results = {}
    graph_metrics = graph_representation.get_metrics()
    for metric in graph_metrics:
        results[metric] = []

    print '> Reading cases..'
    path = '../data/tasa/TASA900_preprocessed'
    texts, labels = data.read_files(path)

    for window_size in range(1,11)+[20,40,80]:
        print '-- window size:',window_size

        rep = {}
        for metric in graph_metrics:
            rep[metric] = []
        print '> Creating representations..'

        # creating graphs and finding centralities
        for text in texts:
            g = graph_representation.construct_cooccurrence_network(text, window_size=window_size, already_preprocessed=True)
            for metric in graph_metrics:
                d = graph_representation.graph_to_dict(g, metric)
                rep[metric].append(d)
            g = None # just to make sure..

        # creating representation vectors
        for metric in graph_metrics:
            rep[metric] = graph_representation.dicts_to_vectors(rep[metric])

        print '> Evaluating..'
        for metric in graph_metrics:
            vectors = rep[metric]
            score = evaluation.evaluate_classification(vectors, labels)
            print '   ', metric, score
            results[metric].append(score)

        data.pickle_to_file(results, 'output/class_context_'+str(window_size))

    pp.pprint(results)
    return results

def do_context_sentence_evaluation_retrieval():
    results = {}
    graph_metrics = graph_representation.get_metrics()
    for metric in graph_metrics:
        results[metric] = []

    print '> Reading cases..'
    descriptions_path = '../data/air/problem_descriptions_text'
    description_texts, labels = data.read_files(descriptions_path)

    solutions_path = '../data/air/solutions_preprocessed'
    solution_texts, labels = data.read_files(solutions_path)
    solution_vectors = freq_representation.text_to_vector(solution_texts, freq_representation.FrequencyMetrics.TF_IDF)

    rep = {}
    for metric in graph_metrics:
        rep[metric] = []
    print '> Creating representations..'

    # creating graphs and finding centralities
    for i, text in enumerate(description_texts):
        if i%10==0: print str(i)+'/'+str(len(description_texts))
        g = graph_representation.construct_cooccurrence_network(text, context='sentence', already_preprocessed=False)
        for metric in graph_metrics:
            d = graph_representation.graph_to_dict(g, metric)
            rep[metric].append(d)
        g = None # just to make sure..

    # creating representation vectors
    for metric in graph_metrics:
        rep[metric] = graph_representation.dicts_to_vectors(rep[metric])

    print '> Evaluating..'
    for metric in graph_metrics:
        vectors = rep[metric]
        score = evaluation.evaluate_retrieval(vectors, solution_vectors)
        print '   ', metric, score
        results[metric].append(score)

    data.pickle_to_file(results, 'output/retr_context_sentence_take2')

    pp.pprint(results)
    return results

def do_context_sentence_evaluation_classification():
    print '> Reading cases..'
    path = '../data/tasa/TASA900_text'
    texts, labels = data.read_files(path)

    print '> Evaluating..'
    graphs = []
    results = {}
    for text in texts:
        g = graph_representation.construct_cooccurrence_network(text, context='sentence')
        graphs.append(g)
    for metric in graph_representation.get_metrics():
        print '   ', metric
        vectors = graph_representation.graphs_to_vectors(graphs, metric, verbose=True)
        score = evaluation.evaluate_classification(vectors, labels)
        results[metric+' (sentence)'] = score

    data.pickle_to_file(results, 'output/class_context_sentence')

    pp.pprint(results)
    return results

def complete_network():
    print '> Reading cases..'
    path = '../data/air/problem_descriptions_text'
    texts, labels = data.read_files(path)

    print '> Creating graph..'
    g = None
    for i, text in enumerate(texts):
        if i%10==0: print str(i)+'/'+str(len(texts))
        tmp = graph_representation.construct_cooccurrence_network(text, context='sentence', already_preprocessed=False)
        if g is None:
            g = tmp
        else:
            g.add_nodes_from(tmp.nodes())
            g.add_edges_from(tmp.edges())

    data.pickle_to_file(g, 'output/complete_networks/air_descriptions.pkl')

    pp.pprint(g)
    return g

def plot_results():
    #~ retr_results = data.pickle_from_file('output/retr_context_10')
    retr_results = {'Weighted degree (window)': [0.22290305491606582,
                       0.2239404496699994,
                       0.22351183191703122,
                       0.22293583927185456,
                       0.2216027852882311,
                       0.22232860216650002,
                       0.22230162622918934,
                       0.22287683186704185,
                       0.22266252053221772,
                       0.22237418794670616],
                 'PageRank (window)': [0.21772129149181993,
                              0.21884861149427587,
                              0.22063142971295358,
                              0.21893898241891538,
                              0.21973766615441442,
                              0.22054672890564322,
                              0.22099589130745473,
                              0.22129686184085004,
                              0.22148942934157456,
                              0.22147928890310792],
                    'Weighted degree (sentence)': [0.21784622825075944]*10,
                    'PageRank (sentence)': [0.22056586008664569]*10}
                    #~ 'PageRank (sentence)':[0.223649757653]*10,
                    #~ 'Weighted degree (sentence)':[0.223449136101]*10}
    pp.pprint(retr_results)
    plotter.plot(range(1,11),retr_results,'retrieval score','n, context size','',[1,10,.216,.225], legend_place="lower right")

    #~ class_results = {'Weighted degree (window)': [0.52777777777777779,
                       #~ 0.53333333333333333,
                       #~ 0.53611111111111109,
                       #~ 0.53333333333333333,
                       #~ 0.53888888888888886,
                       #~ 0.54166666666666663,
                       #~ 0.53611111111111109,
                       #~ 0.52777777777777779,
                       #~ 0.53055555555555556,
                       #~ 0.53055555555555556],
             #~ 'PageRank (window)': [0.55833333333333335,
                          #~ 0.55000000000000004,
                          #~ 0.55277777777777781,
                          #~ 0.54166666666666663,
                          #~ 0.5444444444444444,
                          #~ 0.54722222222222228,
                          #~ 0.54722222222222228,
                          #~ 0.53888888888888886,
                          #~ 0.53888888888888886,
                          #~ 0.53611111111111109],
                #~ 'PageRank (sentence)':[0.56666666666666665]*10,
                #~ 'Weighted degree (sentence)':[0.57499999999999996]*10}
    #~ pp.pprint(class_results)
    #~ plotter.plot(range(1,11),class_results,'classification score','n, context size','',[1,10,.515,.58], legend_place=None)

def corpus_cooccurrence_properties():
    dataset = 'air/problem_descriptions'
    #~ dataset = 'tasa/TASA900'
    print '> Reading data..', dataset
    corpus_path = '../data/'+dataset+'_text'
    (documents, labels) = data.read_files(corpus_path)

    props = {}
    giant = nx.DiGraph()
    print '> Building networks..'
    for i, text in enumerate(documents):
        if i%10==0: print '   ',str(i)+'/'+str(len(documents))
        g = graph_representation.construct_cooccurrence_network(text,context='sentence')
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

    data.pickle_to_file(giant, 'output/properties/cooccurrence/corpus_network_air_sentence')
    data.pickle_to_file(props, 'output/properties/cooccurrence/docs_air_sentence')
    data.pickle_to_file(props_total, 'output/properties/cooccurrence/docs_air_sentence_total')

if __name__ == "__main__":
    #~ pp.pprint(data.pickle_from_file('output/retr_context_sentence_take2'))
    #~ plot_results()

    #~ print "------------------------------------- CLASSIFICATION - context window"
    #~ do_context_size_evaluation_classification()
    #~ print "------------------------------------- CLASSIFICATION - context sentence"
    #~ do_context_sentence_evaluation_classification()

    #~ print "------------------------------------- RETRIEVAL - context window"
    #~ do_context_size_evaluation_retrieval()
    #~ print "------------------------------------- RETRIEVAL - context sentence"
    #~ do_context_sentence_evaluation_retrieval()

    corpus_cooccurrence_properties()
