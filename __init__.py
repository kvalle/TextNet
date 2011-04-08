import data
import graph
import graph_representation
import freq_representation
import evaluation

def classification_demo():
    """Function intended to illustrate classification in the experimental framework.

    Intended as a basis for new experiments for those not intimately
    familiar with the code.
    """
    print 'Evaluation type: Classification'
    print 'Graph type:      Co-occurrence w/2-word window context'
    print 'Centrality:      Weighted degree'
    print
    print '> Reading data..'
    corpus_path = '../data/tasa/TASA900_preprocessed'
    docs, labels = data.read_files(corpus_path)

    print '> Creating representations..'
    dicts = []
    for i, doc in enumerate(docs):
        print '   ',str(i)+'/'+str(len(docs))
        g = graph_representation.construct_cooccurrence_network(doc)
        d = graph_representation.graph_to_dict(g, graph.GraphMetrics.WEIGHTED_DEGREE)
        dicts.append(d)
    vectors = graph_representation.dicts_to_vectors(dicts)

    print '> Evaluating..'
    score = evaluation.evaluate_classification(vectors, labels)
    print '    score:', score
    print

def retrieval_demo():
    """Function intended to illustrate retrieval in the experimental framework.

    Intended as a basis for new experiments for those not intimately
    familiar with the code.
    """
    print 'Evaluation type: Retrieval'
    print 'Graph type:      Dependency'
    print 'Centrality:      PageRank'
    print
    print '> Reading data..'
    desc_path = '../data/air/problem_descriptions_dependencies'
    sol_path = '../data/air/solutions_preprocessed'
    problems, _ = data.read_files(desc_path)
    solutions, _ = data.read_files(sol_path)

    print '> Creating solution representations..'
    metric = freq_representation.FrequencyMetrics.TF_IDF
    sol_vectors = freq_representation.text_to_vector(solutions, metric)

    print '> Creating problem description representations..'
    dicts = []
    for i, doc in enumerate(problems):
        print '   ',str(i)+'/'+str(len(problems))
        g = graph_representation.construct_dependency_network(doc)
        d = graph_representation.graph_to_dict(g, graph.GraphMetrics.PAGERANK)
        dicts.append(d)
    desc_vectors = graph_representation.dicts_to_vectors(dicts)

    print '> Evaluating..'
    score = evaluation.evaluate_retrieval(desc_vectors, sol_vectors)
    print '    score:', score
    print

if __name__=='__main__':
    classification_demo()
    retrieval_demo()
