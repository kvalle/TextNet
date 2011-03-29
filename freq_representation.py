import numpy as np
import math
import networkx as nx
import selector
import preprocess

#~ from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist

def text_to_vector(docs, metric, feature_selection=False):
    """ Create frequency based feature-vector from text """
    doc_freqs = FreqDist() # Distribution over how many documents each word appear in.
    tf_dists = [] # List of TF distributions per document

    # Create freq_dist for each document
    for doc in docs:
        doc = preprocess.preprocess_text(doc)
        fd = FreqDist()
        for word in doc: fd.inc(word)
        doc_freqs.update(fd.samples())
        tf_dists.append(fd)


    all_tokens = doc_freqs.keys()
    num_docs = len(docs)
    num_features = len(all_tokens)


    # Build feature x document matrix
    matrix = np.zeros((num_features, num_docs))
    for i, fd in enumerate(tf_dists):
        if metric == FrequencyMetrics.TF:
            v = [fd.freq(word) for word in all_tokens]
        elif metric == FrequencyMetrics.TF_IDF:
            v = [fd.freq(word) * math.log(float(num_docs)/doc_freqs[word]) for word in all_tokens]
        else:
            raise ValueError("No such feature type: %s" % feature_type);
        matrix[:,i] = v

    if feature_selection:
        sel = selector.MaxSelector(matrix)
        indices = sel.select_features(100)
        matrix = matrix[indices,:]

    return matrix

######
##
##  Term weighting metrics
##
######

class FrequencyMetrics:
    TF = 'tf'
    TF_IDF = 'tf-idf'

def get_metrics():
    return [FrequencyMetrics.TF,
            FrequencyMetrics.TF_IDF]


######
##
##  Test methods
##
######

def test():
    import data
    dataset = 'test/freq'
    corpus_path = '../data/'+dataset
    (documents, labels) = data.read_files(corpus_path)

    vectors = {}
    print '> Building vector representations..'
    for metric in get_metrics():
        print '   ', metric
        vectors[metric] = text_to_vector(documents, metric)

if __name__=='__main__':
    test()
